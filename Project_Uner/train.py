import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import lib.data as datasets
import lib.trainer as trainer
from lib.utils import Logger


def main(config, resume, save_dir, log_dir):
    # DATA LOADERS
    train_set = getattr(datasets, config.datamanager.type)(
        config.datamanager.root, config.datamanager.train_set,
        load_size=config.datamanager.load_size,
        crop_size=config.datamanager.crop_size,
        mean=config.datamanager.norm_mean,
        std=config.datamanager.norm_std,
        mode="train")
    val_set = getattr(datasets, config.datamanager.type)(
        config.datamanager.root, config.datamanager.val_set,
        load_size=config.datamanager.crop_size,
        crop_size=config.datamanager.crop_size,
        mean=config.datamanager.norm_mean,
        std=config.datamanager.norm_std,
        mode="val")

    train_loader = DataLoader(dataset=train_set, num_workers=config.datamanager.workers,
                              batch_size=config.datamanager.batch_size_train, shuffle=True,
                              pin_memory=config.datamanager.pin_memory, drop_last=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=config.datamanager.batch_size_test, shuffle=False,
                            pin_memory=config.datamanager.pin_memory)

    print(f'\n{train_loader.dataset}\n')
    print(f'\n{val_loader.dataset}\n')

    # TRAINING
    runner = getattr(trainer, config.trainer.type)(
        config=config,
        resume=resume,
        save_dir=save_dir,
        log_dir=log_dir,
        train_loader=train_loader,
        val_loader=val_loader)

    runner.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Training Options')

    parser.add_argument('-c', '--config', default='./configs/ITTR.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume',
                        default="",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # Setup seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    # cudnn.benchmark = True

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = edict(config)

    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    save_dir = Path('.').resolve() / config.trainer.save_dir / config.name / start_time
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(log_dir / 'train.txt')

    sys.stdout.description(f'Config file: {args.config}\n')
    print(f'Config file: {args.config}\n')

    sys.stdout.description(str(config))
    print(str(config))

    main(config, args.resume, save_dir, log_dir)
