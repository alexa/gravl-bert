# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert_mlm import VisualLinguisticBertForPretraining
import random

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT_mop(Module):
    def __init__(self, config):

        super(ResNetVLBERT_mop, self).__init__(config)

        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=self.enable_cnn_reg_loss)
            # if config.NETWORK.VLBERT.object_word_embed_mode == 1:
            #     self.object_linguistic_embeddings = nn.Embedding(81, config.NETWORK.VLBERT.hidden_size)
            # elif config.NETWORK.VLBERT.object_word_embed_mode == 2:
            #     self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            # elif config.NETWORK.VLBERT.object_word_embed_mode == 3:
            #     self.object_linguistic_embeddings = None
            # else:
            #     raise NotImplementedError
            self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        # self.object_linguistic_embeddings = None

        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBertForPretraining(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path,
                                         with_rel_head=False, with_mlm_head=False, with_mvrc_head=True)

        # self.hm_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        # self.hi_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)

        dim = config.NETWORK.VLBERT.hidden_size

        # self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]))
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.CrossEntropyLoss()

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        # self.hm_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hm_out.bias.data.zero_()
        # self.hi_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hi_out.bias.data.zero_()
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        if self.config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            language_pretrained = torch.load(self.language_pretrained_model_path)
            mlm_transform_state_dict = {}
            pretrain_keys = []
            for k, v in language_pretrained.items():
                if k.startswith('cls.predictions.transform.'):
                    pretrain_keys.append(k)
                    k_ = k[len('cls.predictions.transform.'):]
                    if 'gamma' in k_:
                        k_ = k_.replace('gamma', 'weight')
                    if 'beta' in k_:
                        k_ = k_.replace('beta', 'bias')
                    mlm_transform_state_dict[k_] = v
            print("loading pretrained classifier transform keys: {}.".format(pretrain_keys))
            self.final_mlp[0].load_state_dict(mlm_transform_state_dict)

    def train(self, mode=True):
        super(ResNetVLBERT_mop, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        for n, p in self.named_parameters():
            if 'image_feature' in n:
                p.requires_grad = False


    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def prepare_text_from_qa(self, question):

        batch_size, max_len, _ = question.shape
        question = torch.clamp(question, min=0)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.uint8, device=question.device)
        text_input_ids = question[:, :, 0].long()
        text_tags = question[:, :, 1].long()
        text_token_type_ids = question[:, :, 2].long()

        return text_input_ids, text_token_type_ids, text_tags, input_mask

    def forward(self,
                  image,
                  boxes,
                  object_ids,
                  mlm_text,
                  obj_cat_id,
                  label,
                  im_info,
                  obj_meta,
                  adj_mat=None
                  ):
        ###########################################


        
        # visual feature extraction
        images = image
        x_inds, y_inds = torch.where(label == 1)

        # box_mask = (boxes[:, :, 0] > - 1.5)
        box_mask = boxes.new_ones((boxes.size(0), boxes.size(1))).long()


        max_len = int(box_mask.sum(1).max().item())
        # box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        boxes = boxes[:, y_inds]
        box_mask = box_mask[:, y_inds]
        object_ids = object_ids[:, y_inds]
        obj_meta = obj_meta[:, y_inds, :]
        label = label[:, y_inds]
        obj_cat_id = obj_cat_id[:, y_inds]
        n_batch, n_obj, _ = boxes.size()

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        kept_obj = []
        for i in range(obj_cat_id.size(1)):
            if torch.rand([1])[0] > 0.5 and (i != 0):
                obj_reps['obj_reps'][:, i, :].fill_(0)
            else:
                kept_obj.append(i)


        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa(mlm_text)

        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])


        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2

        object_linguistic_embeddings = self.object_linguistic_embeddings(boxes.new_zeros(n_batch, n_obj).long())
        ###########################################

        # Visual Linguistic BERT


        relationship_logits, mlm_logits, mvrc_logits = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_linguistic_embeddings,
                                      obj_reps['obj_reps'],
                                      box_mask,
                                      output_all_encoded_layers=False,
                                      output_text_and_object_separately=True)


        # encoded_layers_text [batch, text_len, 768]
        # encoded_layers_obj [batch, obj_len, 768]
        # pooled_output [batch, 768]
        mvrc_loss = im_info.new_zeros(())
        mvrc_label = obj_cat_id.long()[:, kept_obj].view(-1)
        mvrc_logits = mvrc_logits.view((-1, mvrc_logits.shape[-1]))[kept_obj]
        # mvrc_logits_padded = mvrc_logits.new_zeros((*mvrc_label.shape, mvrc_logits.shape[-1])).fill_(-10000.0)
        # mvrc_logits_padded[:, :mvrc_logits.shape[1]] = mvrc_logits
        # mlm_logits = mlm_logits_padded
        mvrc_loss = self.loss_func(mvrc_logits,
                                   mvrc_label)
        ###########################################
        outputs = {}

        if torch.isnan(mvrc_loss):
            raise ValueError()

        outputs.update({'mvrc_logits': mvrc_logits,
                        'mvrc_label': mvrc_label,
                        'mvrc_loss': mvrc_loss})

        return outputs, mvrc_loss
