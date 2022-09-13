import os
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.metrics.save_false_cases import save_false_cases
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from simmc2.data.build import make_dataloader
from simmc2.modules import *

from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import simmc_metrics
from simmc2.function.val import do_validation

@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    # pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()

    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='val', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    q_ids = []
    answer_logits = []
    labels = []
    model.eval()
    cur_id = 0
    tp = 0


    val_metrics_list = [simmc_metrics.Accuracy(allreduce=False, num_replicas=1),
        simmc_metrics.Precision(allreduce=False, num_replicas=1),
        simmc_metrics.Recall(allreduce=False, num_replicas=1)]
    val_metrics = CompositeEvalMetric()
    for child_metric in val_metrics_list:
        val_metrics.add(child_metric)

    # f_saver = save_false_cases(os.path.join(save_path, 'simmc_false_cased.json'))
    # for nbatch, batch in enumerate(test_loader):
    # # for nbatch, batch in tqdm(enumerate(test_loader)):
    #     bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
    #     batch = to_cuda(batch)
    #     label = batch[config.DATASET.LABEL_INDEX_IN_BATCH]
    #     batch = [batch[i] for i in range(len(batch)) if i != config.DATASET.LABEL_INDEX_IN_BATCH]
    #     output = model(*batch)
    #     output['label'] = label[:, 1:]
    #     val_metrics.update(output)
    #     # f_saver.update(output)
    #     labels.append(output['label'].view(-1).detach().cpu().tolist())
    #     answer_logits.append(F.softmax(output['label_logits'], dim=1)[:, 1].detach().cpu().tolist())
    #     cur_id += bs
    do_validation(model, test_loader, val_metrics, config.DATASET.LABEL_INDEX_IN_BATCH)
    print(val_metrics.get())
    # f_saver.save()
    # result = [{'answer_logits': answer_logits, 'label': labels}]

    # cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    # result_json_path = os.path.join(save_path, '{}_simmc2.json'.format(cfg_name if save_name is None else save_name))
    # with open(result_json_path, 'w') as f:
    #     json.dump(result, f)
    # print('result json saved to {}.'.format(result_json_path))
    # return result_json_path
    return 0
