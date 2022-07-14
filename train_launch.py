"""
use torch.distributed.launch to start training
modified from: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/train_multi_GPU

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import init_distributed_mode

from model import CHORE
from trainer.trainer import Trainer
from data.data_paths import DataPaths
from data.train_data import BehaveDataset


def launch_train(args):
    world_size = torch.cuda.device_count()
    init_distributed_mode(args)

    rank = args.rank
    device = torch.device(args.device)

    # prepare model
    model = CHORE(args, rank=rank).to(device)
    ddp_mp_model = DDP(model, device_ids=[rank], find_unused_parameters=True) # this is required

    # prepare data
    train_paths, val_paths = DataPaths.load_splits(args.split_file)
    train_dataset = BehaveDataset(train_paths, args.batch_size, 'train',
                                  num_workers=args.num_workers,
                                  total_samplenum=args.num_samples_train,
                                  image_size=args.net_img_size,
                                  ratios=args.ratios,
                                  sigmas=args.sigmas,
                                  input_type=args.input_type,
                                  random_flip=args.random_flip,
                                  aug_blur=args.aug_blur,
                                  crop_size=args.loadSize,)
    val_dataset = BehaveDataset(val_paths, args.batch_size*2, 'val',
                                  num_workers=args.num_workers,
                                  total_samplenum=args.num_samples_train,
                                  image_size=args.net_img_size,
                                  ratios=args.ratios,
                                  sigmas=args.sigmas,
                                  input_type=args.input_type,
                                  random_flip=False,
                                  aug_blur=0.,
                                  crop_size=args.loadSize, )

    trainer = Trainer(ddp_mp_model, device,
                          train_dataset,
                          val_dataset,
                          args.exp_name,
                          multi_gpus=True,
                          rank=rank,
                          world_size=world_size,
                          threshold=args.clamp_thres,
                          input_type=args.input_type,
                          lr=args.learning_rate,
                          ck_period=60 if 'ck_period' not in args else args.ck_period,
                          milestones=args.milestones)
    # add barrier to make sure the weights are loaded/initialized properly
    dist.barrier()
    # start training
    trainer.train_model(args.num_epochs)
    # clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    """
    launch with: python -m torch.distributed.launch --nproc_per_node=3 --use_env recon/train_launch.py -en=zcat_segmask_launch
    where nproc_per_node is the number of gpus in one machine 
    """
    from argparse import ArgumentParser
    from config.config_loader import load_configs
    parser = ArgumentParser()
    parser.add_argument('-en', '--exp_name')

    # multi-gpu arguments
    # device will be set by system sutomatically
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # number of processes, i.e. number of GPUs
    parser.add_argument('-w', '--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # for pycharm debug
    parser.add_argument('-d1', )
    parser.add_argument('--multiproc')
    parser.add_argument('--qt-support')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    assert args.exp_name==configs.exp_name

    # add command line configs
    configs.device = args.device
    configs.world_size = args.world_size
    configs.dist_url = args.dist_url

    launch_train(configs)