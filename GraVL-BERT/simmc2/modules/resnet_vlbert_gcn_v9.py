import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn_v6 import FastRCNN
from common.visual_linguistic_bert_gcn_v8 import VisualLinguisticBert
import random

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERTgcn9(Module):
    def __init__(self, config):

        super(ResNetVLBERTgcn9, self).__init__(config)

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
            # self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            self.object_linguistic_embeddings = None
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

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        # self.hm_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        # self.hi_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)

        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, 2),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 2)
            )
        elif config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            transform = BertPredictionHeadTransform(config.NETWORK.VLBERT)
            linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, 2)
            self.final_mlp = nn.Sequential(
                transform,
                nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                linear
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        self.turn_pred_head = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 1))

        # self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]))
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.CrossEntropyLoss()
        self.turn_loss = nn.L1Loss()
        # init weights
        # self.init_weight()
        # self.fix_params()

    def init_weight(self):
        # self.hm_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hm_out.bias.data.zero_()
        # self.hi_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hi_out.bias.data.zero_()
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
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
        super(ResNetVLBERTgcn9, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass


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
        text_input_ids = question[:, :, 0]
        text_tags = question[:, :, 1]
        text_token_type_ids = question[:, :, 2]

        return text_input_ids, text_token_type_ids, text_tags, input_mask

    def sample_objs(self, label):
        x_pos, y_pos = torch.where(label == 1)
        x_neg, y_neg = torch.where(label != 1)
        inds = random.sample(range(x_neg.size(0)), min((x_pos.size(0) - 1) * 4, x_neg.size(0)))
        x_neg = x_neg[inds]
        y_neg = y_neg[inds]
        x_inds = torch.cat([x_pos, x_neg], dim=0)
        y_inds = torch.cat([y_pos, y_neg], dim=0)
        return x_inds, y_inds

    def train_forward(self,
                      image,
                      boxes,
                      object_ids, 
                      question,
                      answer,
                      label,
                      im_info,
                      obj_meta,
                      adj_mat,
                      coord_3d=None,
                      mentioned_lbl=None,
                      turn_lbl=None,
                      ):
        ###########################################

        # visual feature extraction
        images = image

        x_inds, y_inds = self.sample_objs(label)

        # box_mask = (boxes[:, :, 0] > - 1.5)
        box_mask = boxes.new_ones((boxes.size(0), boxes.size(1))).long()


        max_len = int(box_mask.sum(1).max().item())
        # box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]


        boxes = boxes[:, y_inds]
        coord_3d = coord_3d[:, y_inds]
        box_mask = box_mask[:, y_inds]
        object_ids = object_ids[:, y_inds]
        obj_meta = obj_meta[:, y_inds, :]
        label = label[:, y_inds]
        adj_mat = adj_mat[:, y_inds, :][:, :, y_inds]
        mentioned_lbl = mentioned_lbl[:, y_inds] if mentioned_lbl is not None else None


        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                coord_3d=coord_3d,
                                                classes=None,
                                                segms=None)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa(question)


        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])


        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2


        # object_linguistic_embeddings = self.object_linguistic_embeddings(object_ids.long().clamp(min=0, max=0))
        object_linguistic_embeddings = None
        ###########################################

        # Visual Linguistic BERT


        encoded_layers, pooled_output = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_linguistic_embeddings,
                                      obj_reps['obj_reps'],
                                      obj_reps['obj_reps_su'],
                                      box_mask,
                                      obj_meta,
                                      adj_mat,
                                      mentioned_lbl,
                                      output_all_encoded_layers=False,
                                      output_text_and_object_separately=False)


        # encoded_layers_text [batch, text_len, 768]
        # encoded_layers_obj [batch, obj_len, 768]
        # pooled_output [batch, 768]


        ###########################################
        outputs = {}

        logits = self.final_mlp(pooled_output)
        logits = logits[1:]
        label = label[:, 1:]
        

        # loss
        # loss = self.loss_func(logits[y_inds].view(-1, 2), label[x_inds, y_inds].view(-1).long())
        loss = self.loss_func(logits.view(-1, 2), label.view(-1).long())
        if torch.isnan(loss):
            raise ValueError()

        # _, turn_inds = torch.where(text_input_ids == 30522)
        turn_logits = self.turn_pred_head(pooled_output)
        loss += 0.05 * self.turn_loss(turn_logits.view(-1,), mentioned_lbl.view(-1).long())


        outputs.update({'label_logits': logits,
                        'label': label,
                        'ans_loss': loss})

        return outputs, loss

    def inference_forward(self,
                      image,
                      boxes,
                      object_ids, 
                      question,
                      answer,
                      im_info,
                      obj_meta,
                      adj_mat,
                      coord_3d=None,
                      mentioned_lbl=None,
                      turn_lbl=None):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = boxes.new_ones((boxes.size(0), boxes.size(1))).long()
        max_len = int(box_mask.sum(1).max().item())
        boxes = boxes[:, :max_len]


        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                coord_3d=coord_3d,
                                                classes=None,
                                                segms=None)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask = self.prepare_text_from_qa(question)
        
        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])


        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2

        # object_linguistic_embeddings = self.object_linguistic_embeddings(object_ids.long().clamp(min=0, max=0))
        object_linguistic_embeddings = None
        ###########################################

        # Visual Linguistic BERT


        encoded_layers, pooled_output = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_linguistic_embeddings,
                                      obj_reps['obj_reps'],
                                      obj_reps['obj_reps_su'],
                                      box_mask,
                                      obj_meta,
                                      adj_mat,
                                      mentioned_lbl,
                                      output_all_encoded_layers=False,
                                      output_text_and_object_separately=False)


        # hm = F.tanh(self.hm_out(hidden_states[_batch_inds, ans_pos]))
        # hi = F.tanh(self.hi_out(hidden_states[_batch_inds, ans_pos + 2]))
        ###########################################
        outputs = {}
        logits = self.final_mlp(pooled_output)
        logits = logits[1:].view(-1, 2)

        outputs.update({'label_logits': logits})

        return outputs
