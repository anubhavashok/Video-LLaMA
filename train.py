"""
Adapted from salesforce@LAVIS and Vision-CAIR@MiniGPT-4. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import multiprocessing

import numpy as np
import torch
print(torch.cuda.device_count())

import cv2
cv2.setNumThreads(0)

import torch.backends.cudnn as cudnn

import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank, init_distributed_mode, is_main_process
from video_llama.common.logger import setup_logger
from video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from video_llama.common.registry import registry
from video_llama.common.utils import now

# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default=None)#str(now()))

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.exp_name is None:
        args.exp_name = os.path.splitext(os.path.basename(args.cfg_path))[0]
        print(f'Exp name not set, setting it to: {args.exp_name}')

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    if is_main_process():
        wandb.init(project="video_embeddings", name=args.exp_name)

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    # datasets['webvid']['train'][0]
    # datasets
    model = task.build_model(cfg)
    device = torch.device("cuda")
    model = model.cuda()

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    multiprocessing.set_start_method('forkserver')
    main()
