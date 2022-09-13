# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import namedtuple
import torch
from common.trainer import to_cuda
import torch.nn.functional as F
import json

@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch, result_save_path=None, testmode=False):
    net.eval()
    if metrics is not None:
        metrics.reset()
    logits = []
    labels = []
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch] if not testmode else None
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        outputs = net(*datas)
        if not testmode:
            outputs.update({'label': label[:, 1:]})
            metrics.update(outputs)
            labels.append(outputs['label'].view(-1).detach().cpu().tolist())
            
        logits.append(F.softmax(outputs['label_logits'], dim=1)[:, 1].detach().cpu().tolist())

    if result_save_path is not None:
        result = [{'answer_logits': logits, 'label': labels}]
        with open(result_save_path, 'w') as f:
            json.dump(result, f)