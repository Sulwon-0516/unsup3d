import argparse
import torch
from unsup3d import setup_runtime, Trainer, Unsup3D
import os



EXP_NAME = '05_31_colab_test'
ROOT_DIR = '/content/'
IS_COLAB = True

os.makedirs(os.path.join(ROOT_DIR, EXP_NAME), exist_ok=True)

## runtime arguments
parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
args = parser.parse_args()

## set up
cfgs = setup_runtime(args)
trainer = Trainer(cfgs, Unsup3D, IS_COLAB, ROOT_DIR)
run_train = cfgs.get('run_train', False)
run_test = cfgs.get('run_test', False)

## run
if run_train:
    trainer.train()
if run_test:
    trainer.test()
