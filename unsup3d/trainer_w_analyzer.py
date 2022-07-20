import os
import glob
from datetime import datetime
import numpy as np
import torch
from . import meters
from . import utils
from tqdm import tqdm
from .dataloaders import get_data_loaders


IS_CAMERA = True

class Trainer_():
    def __init__(self, cfgs, model, is_colab, root_dir):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.is_colab = is_colab
        self.cfgs = cfgs

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs, is_colab, root_dir)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader, trainds, valds, testds = get_data_loaders(cfgs)

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_name is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        return epoch

    def save_checkpoint(self, epoch, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

        ## resume from checkpoint
        if self.resume:
            start_epoch = self.load_checkpoint(optim=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            if self.is_colab:
                self.logger = None
            else:
                from tensorboardX import SummaryWriter
                self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

            ## cache one batch for visualization
            if IS_CAMERA:
                self.viz_input, _ = self.val_loader.__iter__().__next__()
            else:
                self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch(self.train_loader, epoch)
            self.metrics_trace.append("train", metrics)


            if (epoch+1) % self.save_checkpoint_freq == 0:    #############################################################################################################################################
                with torch.no_grad():
                    metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                    self.metrics_trace.append("val", metrics)

                self.save_checkpoint(epoch+1, optim=True)
                #self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
                #self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False, is_camera=IS_CAMERA, img_paths = None):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()


        if is_camera:
            # generate view directory
            view_dir = os.path.join(self.checkpoint_dir,'views')
            os.makedirs(view_dir, exist_ok=True)
            
            # generate sub-directories
            train_view_dir = os.path.join(view_dir, 'train')
            val_view_dir = os.path.join(view_dir, 'val')
            os.makedirs(train_view_dir, exist_ok=True)
            os.makedirs(val_view_dir, exist_ok=True)


        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train() 

        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()
        

        for iter, input in enumerate(loader):
            if is_camera:
                cids = input[1].numpy()
                input = input[0]

                if iter == 0:
                    cid_list = cids
                else:
                    cid_list = np.append(cid_list, cids, axis = 0)

            m = self.model.forward(input)
            if is_train:
                # when it's training get the model estimation
                if is_camera:
                    views, avg_depth = self.model.get_extrinsic()

                    if iter == 0:
                        depth_sum = avg_depth
                        all_views = views

                        SAMPLE_PCS_PATH = os.path.join(view_dir, 'train_sample_pcs.npy')
                        sample_pcs = self.model.get_canon_pc()
                        sample_pcs = sample_pcs.detach().cpu().numpy()

                        np.save(SAMPLE_PCS_PATH, sample_pcs)

                    else:
                        depth_sum += avg_depth
                        all_views = np.append(all_views, views, axis=0)


                self.model.backward()
            
            elif is_test:
                self.model.save_results(self.test_result_dir)
            
            else:
                # validation case
                if is_camera:
                    views, avg_depth = self.model.get_extrinsic()

                    if iter == 0:
                        depth_sum = avg_depth
                        all_views = views

                        SAMPLE_PCS_PATH = os.path.join(view_dir, 'val_sample_pcs.npy')
                        sample_pcs = self.model.get_canon_pc()
                        sample_pcs = sample_pcs.detach().cpu().numpy()

                        np.save(SAMPLE_PCS_PATH, sample_pcs)

                    else:
                        depth_sum += avg_depth
                        all_views = np.append(all_views, views, axis=0)

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.forward(self.viz_input)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)


        # when every epeoch is finished, save result.
        n_case = all_views.shape[0]
        depth_avg = depth_sum / n_case
        cid_list = cid_list.squeeze()

        if is_camera and is_train:
            train_view_path = os.path.join(train_view_dir, 'view_train_%d.npy'%(epoch))
            train_dep_path = os.path.join(train_view_dir, 'avg_dep_train_%d.npy'%(epoch))
            cids_path = os.path.join(train_view_dir, 'cids_%d.npy'%(epoch))

            np.save(train_view_path, all_views)
            np.save(train_dep_path, depth_avg)
            np.save(cids_path, cid_list)
        elif is_camera and is_validation:
            val_view_path = os.path.join(val_view_dir, 'view_val_%d.npy'%(epoch))
            val_dep_path = os.path.join(val_view_dir, 'avg_dep_val_%d.npy'%(epoch))
            cids_path = os.path.join(val_view_dir, 'cids_%d.npy'%(epoch))

            np.save(val_view_path, all_views)
            np.save(val_dep_path, depth_avg)
            np.save(cids_path, cid_list)

        return metrics



######################################################################################

    def result_analyze(self):
        """Perform testing."""
        VIEW_PATH = "/home/unsup3d/view_dir/views_90.npy"
        AVG_DEPTH_PATH = "/home/unsup3d/view_dir/avg_depth_90.npy"
        SAMPLE_PCS_PATH = "/home/unsup3d/view_dir/sample_pcs_90.npy"


        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))


        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            print(f"Starting analyzing")
            self.model.set_eval()

            for iter, input in tqdm(enumerate(self.test_loader)):
                m = self.model.forward(input)
                views, avg_depth = self.model.get_extrinsic()

                if iter == 0:
                    depth_sum = avg_depth
                    all_views = views

                    sample_pcs = self.model.get_canon_pc()
                    sample_pcs = sample_pcs.detach().cpu().numpy()

                    np.save(SAMPLE_PCS_PATH, sample_pcs)
                    self.model.visualize(self.logger, total_iter=1, max_bs=25)


                else:
                    depth_sum += avg_depth
                    all_views = np.append(all_views, views, axis=0)
                
            n_case = all_views.shape[0]
            depth_avg = depth_sum / n_case


            print("all views shape: ", all_views.shape)
            print("avg depth shape: ", depth_sum.shape)

            np.save(VIEW_PATH, all_views)
            np.save(AVG_DEPTH_PATH, depth_avg)

