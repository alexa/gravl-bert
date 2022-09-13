# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import pprint
import shutil
import inspect

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from common.utils.create_logger import create_logger
from common.utils.misc import summary_parameters, bn_fp16_half_eval
from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from common.trainer import train
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import simmc_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from simmc2.data.build import make_dataloader, build_dataset, build_transforms
from simmc2.modules import *
from simmc2.function.val import do_validation

try:
		from apex import amp
		from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
		pass
		#raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def test_net(args, config):

		# cudnn
		torch.backends.cudnn.benchmark = False
		if args.cudnn_off:
				torch.backends.cudnn.enabled = False


		#os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
		model = eval(config.MODULE)(config)
		num_gpus = len(config.GPUS.split(','))
		assert num_gpus <= 1 or (not config.TRAIN.FP16), "Not support fp16 with torch.nn.DataParallel. " \
																										 "Please use amp.parallel.DistributedDataParallel instead."
		total_gpus = num_gpus

		# model
		torch.cuda.set_device(int(config.GPUS))
		model.cuda()

		# loader
		val_loader = make_dataloader(config, mode='val', distributed=False)

		batch_size = num_gpus * (sum(config.TRAIN.BATCH_IMAGES) if isinstance(config.TRAIN.BATCH_IMAGES, list)
														 else config.TRAIN.BATCH_IMAGES)

		# partial load pretrain state dict
		pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
		prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]

		pretrain_state_dict_parsed = {}
		for k, v in pretrain_state_dict.items():
				no_match = True
				for pretrain_prefix, new_prefix in prefix_change:
						if k.startswith(pretrain_prefix):
								k = new_prefix + k[len(pretrain_prefix):]
								pretrain_state_dict_parsed[k] = v
								no_match = False
								break
				if no_match:
						pretrain_state_dict_parsed[k] = v
		if 'module.vlbert.relationsip_head.caption_image_relationship.weight' in pretrain_state_dict \
						and config.NETWORK.LOAD_REL_HEAD:
				pretrain_state_dict_parsed['module.final_mlp.1.weight'] \
						= pretrain_state_dict['module.vlbert.relationsip_head.caption_image_relationship.weight'][1:2].float() \
						- pretrain_state_dict['module.vlbert.relationsip_head.caption_image_relationship.weight'][0:1].float()
				pretrain_state_dict_parsed['module.final_mlp.1.bias'] \
						= pretrain_state_dict['module.vlbert.relationsip_head.caption_image_relationship.bias'][1:2].float() \
							- pretrain_state_dict['module.vlbert.relationsip_head.caption_image_relationship.bias'][0:1].float()
		if config.NETWORK.PARTIAL_PRETRAIN_SEGMB_INIT:
				if isinstance(pretrain_state_dict_parsed['module.vlbert._module.token_type_embeddings.weight'],
											torch.HalfTensor):
						pretrain_state_dict_parsed['module.vlbert._module.token_type_embeddings.weight'] = \
								pretrain_state_dict_parsed['module.vlbert._module.token_type_embeddings.weight'].float()
				pretrain_state_dict_parsed['module.vlbert._module.token_type_embeddings.weight'][1] = \
						pretrain_state_dict_parsed['module.vlbert._module.token_type_embeddings.weight'][0]
		pretrain_state_dict = pretrain_state_dict_parsed

		smart_partial_load_model_state_dict(model, pretrain_state_dict)

		# metrics
		# do not use it since we do not have labels
		# val_metrics_list = [simmc_metrics.Accuracy(allreduce=args.dist, num_replicas=world_size if args.dist else 1),
		# 		simmc_metrics.Precision(allreduce=args.dist, num_replicas=world_size if args.dist else 1),
		# 		simmc_metrics.Recall(allreduce=args.dist, num_replicas=world_size if args.dist else 1)]
		# val_metrics = CompositeEvalMetric()
		# for child_metric in val_metrics_list:
		# 		val_metrics.add(child_metric)
		val_metrics = None

		# epoch end callbacks
		# epoch_end_callbacks = []
		# if (rank is None) or (rank == 0):
		# 		epoch_end_callbacks = [Checkpoint(model_prefix, config.CHECKPOINT_FREQUENT)]


		# batch
		batch_size = len(config.GPUS.split(',')) * config.TRAIN.BATCH_IMAGES

		# setup lr step and lr scheduler



		# apex: amp fp16 mixed-precision training
		if config.TRAIN.FP16:
				# model.apply(bn_fp16_half_eval)
				model, optimizer = amp.initialize(model, optimizer,
																					opt_level='O2',
																					keep_batchnorm_fp32=False,
																					loss_scale=config.TRAIN.FP16_LOSS_SCALE,
																					min_loss_scale=128.0)
				if args.dist:
						model = Apex_DDP(model, delay_allreduce=True)
		with torch.no_grad():
			do_validation(model, val_loader, val_metrics, config.DATASET.LABEL_INDEX_IN_BATCH, result_save_path=args.savepath, testmode=True)

		# train(model, optimizer, lr_scheduler, train_loader, train_sampler, train_metrics,
		# 			config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH, logger,
		# 			rank=rank, batch_end_callbacks=batch_end_callbacks, epoch_end_callbacks=epoch_end_callbacks,
		# 			writer=writer, validation_monitor=validation_monitor, fp16=config.TRAIN.FP16,
		# 			clip_grad_norm=config.TRAIN.CLIP_GRAD_NORM,
		# 			gradient_accumulate_steps=config.TRAIN.GRAD_ACCUMULATE_STEPS,
		# 			eval_only=args.evalonly)

