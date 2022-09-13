# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import torch.utils.data

from .datasets import *
from . import samplers
from .transforms.build import build_transforms
from .collate_batch import BatchCollator
import pprint

DATASET_CATALOGS = {'SIMMC2': SIMMC2Dataset, 'mturk': mturkDataset}


def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)


def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, batch_size):

    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False)
    return batch_sampler


def make_dataloader(cfg, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None,
                    expose_sampler=False):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        ann_file = cfg.DATASET.TRAIN_ANNOTATION_FILE
        adj_path = cfg.DATASET.TRAIN_ADJ_PATH if cfg.DATASET.ADD_ADJ else None
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
    elif mode == 'val':
        ann_file = cfg.DATASET.VAL_ANNOTATION_FILE
        adj_path = cfg.DATASET.VAL_ADJ_PATH if cfg.DATASET.ADD_ADJ else None
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.VAL.BATCH_IMAGES * num_gpu
        shuffle = cfg.VAL.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu
    else:
        ann_file = cfg.DATASET.TEST_ANNOTATION_FILE
        adj_path = cfg.DATASET.TEST_ADJ_PATH if cfg.DATASET.ADD_ADJ else None
        num_gpu = len(cfg.GPUS.split(','))
        batch_size = cfg.TEST.BATCH_IMAGES * num_gpu
        shuffle = cfg.TEST.SHUFFLE
        num_workers = cfg.NUM_WORKERS_PER_GPU * num_gpu

    transform = build_transforms(cfg, mode)

    if dataset is None:

        dataset = build_dataset(dataset_name=cfg.DATASET.DATASET, ann_file=ann_file,
                                root_path=cfg.DATASET.ROOT_PATH, data_path=cfg.DATASET.DATASET_PATH,
                                img_path=cfg.DATASET.IMAGE_PATH,
                                test_mode=(mode == 'test'), 
                                adj_path=adj_path, transform=transform,
                                add_image_as_a_box=cfg.DATASET.ADD_IMAGE_AS_A_BOX,
                                mask_size=(cfg.DATASET.MASK_SIZE, cfg.DATASET.MASK_SIZE),
                                pretrained_model_name=cfg.NETWORK.BERT_MODEL_NAME,
                                use_mentioned_lbl=cfg.DATASET.USE_MENTIONED_LBL,
                                use_3d_coords=cfg.DATASET.USE_3D_COORDS,
                                use_turn_lbl=cfg.DATASET.USE_TURN_LBL)

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size)
    collator = BatchCollator(dataset=dataset, append_ind=cfg.DATASET.APPEND_INDEX)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collator)
    if expose_sampler:
        return dataloader, sampler

    return dataloader
