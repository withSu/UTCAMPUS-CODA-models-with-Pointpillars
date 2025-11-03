#!/usr/bin/env python3
"""Simple evaluation script for trained models"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add tools path
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_yaml_file
from eval_utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--cfg_file', type=str, required=True, help='config file')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    # Set split
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = args.split

    # Create output directory
    output_dir = Path('output') / cfg.EXP_GROUP_PATH / cfg.TAG / 'eval' / f'eval_{args.split}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    log_file = output_dir / 'eval_log.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info('**********************Start Evaluation**********************')
    logger.info(f'Config file: {args.cfg_file}')
    logger.info(f'Checkpoint: {args.ckpt}')
    logger.info(f'Split: {args.split}')

    # Build dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    logger.info(f'Total samples: {len(test_set)}')

    # Build model
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )

    # Load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # Evaluate
    with torch.no_grad():
        eval_utils.eval_one_epoch(
            cfg, model, test_loader, 0, logger,
            dist_test=False,
            result_dir=output_dir,
            save_to_file=True
        )

    logger.info('**********************Evaluation Finished**********************')


if __name__ == '__main__':
    main()
