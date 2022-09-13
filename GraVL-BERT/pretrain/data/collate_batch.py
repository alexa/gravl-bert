# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import torch
from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        if batch[0][self.data_names.index('image')] is not None:
            max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
            image_none = False
        else:
            image_none = True
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'answer' in self.data_names:
            max_answers = max([len(data[self.data_names.index('answer')]) for data in batch])
        if 'question' in self.data_names:
            max_question_length = max([len(data[self.data_names.index('question')]) for data in batch])
        if 'obj_meta' in self.data_names:
            max_meta_length = max([data[self.data_names.index('obj_meta')].shape[1] for data in batch])
        if 'mlm_text' in self.data_names:
            max_mlm_text_length = max([len(data[self.data_names.index('mlm_text')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if image_none:
                out['image'] = None
            else:
                image = ibatch[self.data_names.index('image')]
                out['image'] = clip_pad_images(image, max_shape, pad=0)

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            if 'objects' in self.data_names:
                objects = ibatch[self.data_names.index('objects')]
                out['objects'] = clip_pad_1d(objects, max_boxes, pad=0)

            if 'question' in self.data_names:
                question = ibatch[self.data_names.index('question')]
                out['question'] = clip_pad_2d(question, (max_question_length, len(question[0])), pad=-2)

            if 'answer' in self.data_names:
                answer = ibatch[self.data_names.index('answer')]
                out['answer'] = clip_pad_1d(answer, max_answers, pad=-1)

            if 'label' in self.data_names:
                label = ibatch[self.data_names.index('label')]
                out['label'] = clip_pad_1d(label, max_boxes, pad=0)

            if 'obj_meta' in self.data_names:
                obj_meta = ibatch[self.data_names.index('obj_meta')]
                out['obj_meta'] = clip_pad_2d(obj_meta, (max_boxes, max_meta_length), pad=0)


            if 'adj_mat' in self.data_names:
                adj_mat = ibatch[self.data_names.index('adj_mat')]
                out['adj_mat'] = clip_pad_2d(adj_mat, (max_boxes, max_boxes), pad=0)

            if 'mentioned_lbl' in self.data_names:
                mentioned_lbl = ibatch[self.data_names.index('mentioned_lbl')]
                out['mentioned_lbl'] = clip_pad_1d(mentioned_lbl, max_boxes, pad=0)

            if 'mlm_text' in self.data_names:
                mlm_text = ibatch[self.data_names.index('mlm_text')]
                out['mlm_text'] = clip_pad_2d(mlm_text, (max_mlm_text_length, len(mlm_text[0])), pad=-2)

            if 'mlm_label' in self.data_names:
                mlm_label = ibatch[self.data_names.index('mlm_label')]
                out['mlm_label'] = clip_pad_1d(mlm_label, max_mlm_text_length, pad=0)

            if 'obj_cat_id' in self.data_names:
                obj_cat_id = ibatch[self.data_names.index('obj_cat_id')]
                out['obj_cat_id'] = clip_pad_1d(obj_cat_id, max_boxes, pad=-1)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple

