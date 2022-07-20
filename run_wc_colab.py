# run with camera

import argparse
import torch
from unsup3d import setup_runtime, Trainer_, Unsup3D
import os

IS_COLAB = True
ROOT_DIR = '/content/'



## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default='experiments/train_celeba_64_b16.yml', type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('--save_dir', default='/content/drive/MyDrive/Unsup3D', type = str, help = 'directory, that result will be saved')
parser.add_argument('--exp_name', default='debug_wc', type = str, help = 'name of folder, will be generated in save_dir')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = Trainer_(cfgs, Unsup3D, IS_COLAB, ROOT_DIR)

run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

## run
if run_train:
    trainer.train()
elif run_test:
    trainer.test()
