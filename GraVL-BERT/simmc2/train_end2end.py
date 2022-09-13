# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import _init_paths
import os
import argparse
import torch
import subprocess

from simmc2.function.config import config, update_config
from simmc2.function.train import train_net
from simmc2.function.test import test_net


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file', default='/home/ec2-user/VL-BERT/cfgs/simmc/base_qa2r_4x16G_fp32.yaml')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint', default='/home/ec2-user/VL-BERT/model/simmc2')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir', default='/dev/shm/simmc2_logdir')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set',
                        default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')

    # easy test pretrain model
    parser.add_argument('--partial-pretrain', type=str)
    parser.add_argument('--evalonly', type=bool, default=False)
    parser.add_argument('--savepath', type=str, default=None)
    parser.add_argument('--nolabel', type=bool, default=False)

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    if args.partial_pretrain is not None:
        config.NETWORK.PARTIAL_PRETRAIN = args.partial_pretrain

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    if args.evalonly:
        config.GPUS = '0'
        config.TRAIN.AUTO_RESUME = False

    return args, config


def main():
    args, config = parse_args()
    rank, model = train_net(args, config)
    if args.do_test and (rank is None or rank == 0):
        test_net(args, config)


if __name__ == '__main__':
    main()


