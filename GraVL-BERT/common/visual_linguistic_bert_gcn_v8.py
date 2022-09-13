# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from external.pytorch_pretrained_bert.modeling import BertLayerNorm, BertEncoder, BertPooler, ACT2FN, BertOnlyMLMHead
from common.gcn.GCN import GCN
import copy
# todo: add this to config
NUM_SPECIAL_WORDS = 1000


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class VisualLinguisticBert(BaseModel):
    def __init__(self, config, language_pretrained_model_path=None):
        super(VisualLinguisticBert, self).__init__(config)

        self.config = config
        
        # embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.end_embedding = nn.Embedding(1, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.obj_su_embeddings = nn.Embedding(1, config.hidden_size)

        
        self.mentioned_lbl_embeddings = nn.Linear(1, config.hidden_size)

        self.embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.gcn_token_embeddings = nn.Embedding(1, config.hidden_size)


        self.gcn = GCN(config.hidden_size, 5)
        self.gcn_2 = GCN(config.hidden_size, 5)
        self.gcn_3 = GCN(config.hidden_size, 5)
        # for compatibility of roberta
        self.position_padding_idx = config.position_padding_idx

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.visual_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.visual_size, config.hidden_size)
        if config.visual_ln:
            self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            visual_scale_text = nn.Parameter(torch.as_tensor(self.config.visual_scale_text_init, dtype=torch.float),
                                             requires_grad=True)
            self.register_parameter('visual_scale_text', visual_scale_text)
            visual_scale_object = nn.Parameter(torch.as_tensor(self.config.visual_scale_object_init, dtype=torch.float),
                                               requires_grad=True)
            self.register_parameter('visual_scale_object', visual_scale_object)

        self.encoder = BertEncoder(config)

        if self.config.with_pooler:
            self.pooler = BertPooler(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False
            self.special_word_embeddings = nn.Embedding(NUM_SPECIAL_WORDS, config.hidden_size)
            self.special_word_embeddings.weight.data.copy_(self.word_embeddings.weight.data[:NUM_SPECIAL_WORDS])

    def word_embeddings_wrapper(self, input_ids):
        if self.config.word_embedding_frozen:
            word_embeddings = self.word_embeddings(input_ids)
            word_embeddings[input_ids < NUM_SPECIAL_WORDS] \
                = self.special_word_embeddings(input_ids[input_ids < NUM_SPECIAL_WORDS])
            return word_embeddings
        else:
            return self.word_embeddings(input_ids)

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_visual_embeddings,
                text_mask,
                object_linguistic_embeddings,
                object_reps,
                object_reps_su,
                object_mask,
                object_meta,
                adj_mat = None,
                mentioned_lbl = None,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False,
                output_attention_probs=False):

        # get seamless concatenate embeddings and mask
        embedding_output, attention_mask, text_mask_new, object_mask_new = self.embedding(text_input_ids,
                                                                                          text_token_type_ids,
                                                                                          text_visual_embeddings,
                                                                                          text_mask,
                                                                                          object_linguistic_embeddings,
                                                                                          object_meta,
                                                                                          object_reps,
                                                                                          object_reps_su,
                                                                                          object_mask,
                                                                                          adj_mat[0],
                                                                                          mentioned_lbl)


        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_attention_mask = 1.0 - extended_attention_mask
        # extended_attention_mask[extended_attention_mask != 0] = float('-inf')

        if output_attention_probs:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers=output_all_encoded_layers,
                                                           output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers,
                                          output_attention_probs=output_attention_probs)


        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if output_text_and_object_separately:
            if not output_all_encoded_layers:
                encoded_layers = [encoded_layers]
            encoded_layers_text = []
            encoded_layers_object = []
            for encoded_layer in encoded_layers:
                max_text_len = text_input_ids.shape[1]
                max_object_len = object_mask_new.shape[1]
                encoded_layer_text = encoded_layer[:, :max_text_len]
                # encoded_layer_object = encoded_layer.new_zeros(
                #     (encoded_layer.shape[0], max_object_len, encoded_layer.shape[2]))
                # print(encoded_layer_object.size(), object_mask.size(), encoded_layer.size(), object_mask_new.size())
                # print(object_mask_new)
                # encoded_layer_object[object_mask] = encoded_layer[object_mask_new]
                encoded_layer_object = encoded_layer[:, max_text_len:-1]
                encoded_layers_text.append(encoded_layer_text)
                encoded_layers_object.append(encoded_layer_object)
            if not output_all_encoded_layers:
                encoded_layers_text = encoded_layers_text[0]
                encoded_layers_object = encoded_layers_object[0]
            if output_attention_probs:
                return encoded_layers_text, encoded_layers_object, pooled_output, attention_probs
            else:
                return encoded_layers_text, encoded_layers_object, pooled_output
        else:
            if output_attention_probs:
                return encoded_layers, pooled_output, attention_probs
            else:
                return encoded_layers, pooled_output

    def embedding(self,
                  text_input_ids,
                  text_token_type_ids,
                  text_visual_embeddings,
                  text_mask,
                  object_linguistic_embeddings,
                  object_meta,
                  object_reps,
                  object_reps_su,
                  object_mask,
                  adj_mat=None,
                  mentioned_lbl=None):

        # print(text_input_ids.size())
        # print(text_token_type_ids.size())
        # print(text_visual_embeddings.size())
        # print(text_mask.size())
        # print(object_linguistic_embeddings.size())
        # print(object_mask.size())

        n_batch, n_obj, meta_len = object_meta.size()

        '''
        ###########################################
        get text vl embeddings
        ###########################################
        '''
        text_linguistic_embedding = self.word_embeddings_wrapper(text_input_ids.long())


        if self.visual_1x1_text is not None:
            text_visual_embeddings = self.visual_1x1_text(text_visual_embeddings)

        if self.config.visual_ln:
            text_visual_embeddings = self.visual_ln_text(text_visual_embeddings)
        else:
            text_visual_embeddings *= self.visual_scale_text

        text_vl_embeddings = text_linguistic_embedding + text_visual_embeddings
        # txt_vl_emb reshape [1, len, 768]
        # expand to [n_obj, len, 768]
        text_vl_embeddings = text_vl_embeddings.repeat(n_obj, 1, 1)


        '''
        ###########################################
        get object vl embeddings
        ###########################################
        '''

        # obj_meta reshape [1, n_obj, len] -> [n_obj, len]
        # embed shape [n_obj, len, 768] and exclude the cls token
        object_linguistic_embeddings = self.word_embeddings_wrapper(object_meta.view(n_batch*n_obj, -1).long())[:, 1:-1, :]

        '''
        ###########################################
        get mentioned_lbl embeddings
        ###########################################
        '''
        if mentioned_lbl is not None:
            mentioned_lbl = mentioned_lbl.view(n_obj, 1, 1).float()
            mentioned_lbl_embeddings = self.mentioned_lbl_embeddings(mentioned_lbl)
            object_linguistic_embeddings = torch.cat([object_linguistic_embeddings, mentioned_lbl_embeddings], dim=1)


        object_visual_embeddings = object_reps

        # if self.visual_1x1_object is not None:
        #     object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        # if self.config.visual_ln:
        #     object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)
        # else:
        #     object_visual_embeddings *= self.visual_scale_object

        # # obj_vl_emb reshape [1, n_obj, 768] -> [n_obj, 1, 768]
        # object_visual_embeddings = object_visual_embeddings.view(n_obj, 1, -1)
        # object_visual_embeddings = object_visual_embeddings.repeat(1, object_linguistic_embeddings.size(1), 1)

        # object_vl_embeddings = object_linguistic_embeddings + object_visual_embeddings


        '''
        ###########################################
        get gcn embeddings
        ###########################################
        '''
        if n_obj != adj_mat.size(0):
            raise ValueError('size not match')
        xs, ys = torch.where(adj_mat != 0)
        edge_type = adj_mat[xs, ys].view(-1)
        edge_index = torch.cat([xs.view(1,- 1), ys.view(1, -1)], dim=0)

        object_gcn_embddings = self.gcn(object_visual_embeddings.view(n_obj, -1), edge_index, edge_type)
        object_gcn_embddings = self.gcn_2(object_gcn_embddings, edge_index, edge_type)
        object_gcn_embddings = self.gcn_3(object_gcn_embddings, edge_index, edge_type).unsqueeze(1)
        # conv_obj_feats is of shape [n_obj, 1, n_feat]
        # _zero_id = torch.zeros((n_obj, ), dtype=torch.long, device=text_vl_embeddings.device)
        # gcn_token_embeddings = self.gcn_token_embeddings(_zero_id).unsqueeze(1)
        # gcn_vl_embeddings = gcn_token_embeddings + object_gcn_embddings
        object_gcn_embddings = self.visual_ln_object(object_gcn_embddings)
        object_gcn_embddings = object_gcn_embddings.repeat(1, object_linguistic_embeddings.size(1), 1)




        object_vl_embeddings = object_linguistic_embeddings + object_gcn_embddings

        '''
        ###########################################
        surrounding embeddings
        ###########################################
        '''
        if object_reps_su is not None:
            object_surrounding_visual_embeddings = self.visual_ln_object(object_reps_su.view(n_obj, 8, -1))
            object_surrounding_linguistic_embeddings = self.obj_su_embeddings(torch.zeros((n_obj, 8), dtype=torch.long, device=text_vl_embeddings.device))
            object_surrounding_vl_embeddings = object_surrounding_linguistic_embeddings + object_surrounding_visual_embeddings
            object_vl_embeddings = torch.cat([object_vl_embeddings, object_surrounding_vl_embeddings], dim=1)

        '''
        ###########################################
        end embeddings
        ###########################################
        '''
        _zero_id = torch.zeros((n_obj, ), dtype=torch.long, device=text_vl_embeddings.device)


        # vl_embeddings = torch.cat([text_vl_embeddings, object_vl_embeddings, gcn_vl_embeddings, self.end_embedding(_zero_id).unsqueeze(1)], dim=1)
        vl_embeddings = torch.cat([text_vl_embeddings, object_vl_embeddings, self.end_embedding(_zero_id).unsqueeze(1)], dim=1)


        max_length = vl_embeddings.size(1)
        vl_embed_size = text_vl_embeddings.size(-1)


        grid_ind, grid_pos = torch.meshgrid(torch.arange(n_obj, dtype=torch.long, device=text_vl_embeddings.device),
                                            torch.arange(max_length, dtype=torch.long, device=text_vl_embeddings.device))
        text_end = text_mask.new_zeros((n_obj, 1)).long().fill_(max_length-2)
        object_end = text_mask.new_zeros((n_obj, 1)).long().fill_(max_length-1)



        # token type embeddings/ segment embeddings
        obj_type_ids = text_token_type_ids.new_zeros((n_obj, object_vl_embeddings.size(1)+1)).fill_(2)

        text_type_ids = text_token_type_ids.repeat(n_obj, 1)
        # text_type_ids = text_token_type_ids.new_zeros((n_obj, text_vl_embeddings.size(1)))
        token_type_ids = torch.cat([text_type_ids, obj_type_ids], dim=1)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)


        # position embeddings

        txt_pos = grid_pos[:, :max_length-2] + self.position_padding_idx + 1
        if self.config.obj_pos_id_relative:
            obj_pos = txt_pos.new_zeros((n_obj, 1)) + (text_end + self.position_padding_idx + 1)
            end_pos = txt_pos.new_zeros((n_obj, 1)) + (text_end + self.position_padding_idx + 2)
            
        else:
            assert False, "Don't use position id 510/511 for objects and [END]!!!"
            obj_pos = txt_pos.new_zeros((n_obj, 1)) + (self.config.max_position_embeddings - 2)
            end_pos = txt_pos.new_zeros((n_obj, 1)) + (self.config.max_position_embeddings - 1)
        position_ids = torch.cat([txt_pos, obj_pos, end_pos], dim=-1)
        position_embeddings = self.position_embeddings(position_ids)
        # mask = text_mask.new_zeros((bs, max_length))
        # mask[torch.where(grid_pos <= object_end)] = 1
        mask = torch.cat([text_mask.new_ones((n_obj, max_length-1)), text_mask.new_zeros((n_obj, 1))], dim=-1)

        embeddings = vl_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)



        return embeddings, mask, grid_pos < text_end, (grid_pos >= text_end) & (grid_pos < object_end)

    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        unexpected_keys = []
        for k, v in pretrained_state_dict.items():
            if k.startswith('bert.'):
                k = k[len('bert.'):]
            elif k.startswith('roberta.'):
                k = k[len('roberta.'):]
            else:
                unexpected_keys.append(k)
                continue
            if 'gamma' in k:
                k = k.replace('gamma', 'weight')
            if 'beta' in k:
                k = k.replace('beta', 'bias')
            if k.startswith('encoder.'):
                k_ = k[len('encoder.'):]
                if k_ in self.encoder.state_dict():
                    encoder_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            elif k.startswith('embeddings.'):
                k_ = k[len('embeddings.'):]
                if k_ == 'word_embeddings.weight':
                    nn.init.normal_(self.word_embeddings.weight)
                    data1 = v.to(dtype=self.word_embeddings.weight.data.dtype, device=self.word_embeddings.weight.data.device)
                    v_len = data1.size(0)
                    self.word_embeddings.weight.data[:v_len, :] = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                            device=self.word_embeddings.weight.data.device)
                elif k_ == 'position_embeddings.weight':
                    self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                device=self.position_embeddings.weight.data.device)
                elif k_ == 'token_type_embeddings.weight':
                    self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                        dtype=self.token_type_embeddings.weight.data.dtype,
                        device=self.token_type_embeddings.weight.data.device)
                    if v.size(0) == 1:
                        # Todo: roberta token type embedding
                        self.token_type_embeddings.weight.data[1] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        self.token_type_embeddings.weight.data[2] = v[0].clone().to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)

                elif k_.startswith('LayerNorm.'):
                    k__ = k_[len('LayerNorm.'):]
                    if k__ in self.embedding_LayerNorm.state_dict():
                        embedding_ln_pretrained_state_dict[k__] = v
                    else:
                        unexpected_keys.append(k)
                else:
                    unexpected_keys.append(k)
            elif self.config.with_pooler and k.startswith('pooler.'):
                k_ = k[len('pooler.'):]
                if k_ in self.pooler.state_dict():
                    pooler_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(k)
            else:
                unexpected_keys.append(k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)


class VisualLinguisticBertForPretraining(VisualLinguisticBert):
    def __init__(self, config, language_pretrained_model_path=None,
                 with_rel_head=True, with_mlm_head=True, with_mvrc_head=True):

        super(VisualLinguisticBertForPretraining, self).__init__(config, language_pretrained_model_path=None)

        self.with_rel_head = with_rel_head
        self.with_mlm_head = with_mlm_head
        self.with_mvrc_head = with_mvrc_head
        if with_rel_head:
            self.relationsip_head = VisualLinguisticBertRelationshipPredictionHead(config)
        if with_mlm_head:
            self.mlm_head = BertOnlyMLMHead(config, self.word_embeddings.weight)
        if with_mvrc_head:
            self.mvrc_head = VisualLinguisticBertMVRCHead(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        # load language pretrained model
        if language_pretrained_model_path is not None:
            self.load_language_pretrained_model(language_pretrained_model_path)

        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.pos_embedding_frozen:
            for p in self.position_embeddings.parameters():
                p.requires_grad = False

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_visual_embeddings,
                text_mask,
                object_vl_embeddings,
                object_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False):

        text_out, object_out, pooled_rep = super(VisualLinguisticBertForPretraining, self).forward(
            text_input_ids,
            text_token_type_ids,
            text_visual_embeddings,
            text_mask,
            object_vl_embeddings,
            object_mask,
            output_all_encoded_layers=False,
            output_text_and_object_separately=True
        )

        if self.with_rel_head:
            relationship_logits = self.relationsip_head(pooled_rep)
        else:
            relationship_logits = None
        if self.with_mlm_head:
            mlm_logits = self.mlm_head(text_out)
        else:
            mlm_logits = None
        if self.with_mvrc_head:
            mvrc_logits = self.mvrc_head(object_out)
        else:
            mvrc_logits = None

        return relationship_logits, mlm_logits, mvrc_logits

    def load_language_pretrained_model(self, language_pretrained_model_path):
        pretrained_state_dict = torch.load(language_pretrained_model_path, map_location=lambda storage, loc: storage)
        encoder_pretrained_state_dict = {}
        pooler_pretrained_state_dict = {}
        embedding_ln_pretrained_state_dict = {}
        relationship_head_pretrained_state_dict = {}
        mlm_head_pretrained_state_dict = {}
        unexpected_keys = []
        for _k, v in pretrained_state_dict.items():
            if _k.startswith('bert.') or _k.startswith('roberta.'):
                k = _k[len('bert.'):] if _k.startswith('bert.') else _k[len('roberta.'):]
                if 'gamma' in k:
                    k = k.replace('gamma', 'weight')
                if 'beta' in k:
                    k = k.replace('beta', 'bias')
                if k.startswith('encoder.'):
                    k_ = k[len('encoder.'):]
                    if k_ in self.encoder.state_dict():
                        encoder_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
                elif k.startswith('embeddings.'):
                    k_ = k[len('embeddings.'):]
                    if k_ == 'word_embeddings.weight':
                        self.word_embeddings.weight.data = v.to(dtype=self.word_embeddings.weight.data.dtype,
                                                                device=self.word_embeddings.weight.data.device)
                    elif k_ == 'position_embeddings.weight':
                        self.position_embeddings.weight.data = v.to(dtype=self.position_embeddings.weight.data.dtype,
                                                                    device=self.position_embeddings.weight.data.device)
                    elif k_ == 'token_type_embeddings.weight':
                        self.token_type_embeddings.weight.data[:v.size(0)] = v.to(
                            dtype=self.token_type_embeddings.weight.data.dtype,
                            device=self.token_type_embeddings.weight.data.device)
                        if v.size(0) == 1:
                            # Todo: roberta token type embedding
                            self.token_type_embeddings.weight.data[1] = v[0].to(
                                dtype=self.token_type_embeddings.weight.data.dtype,
                                device=self.token_type_embeddings.weight.data.device)
                    elif k_.startswith('LayerNorm.'):
                        k__ = k_[len('LayerNorm.'):]
                        if k__ in self.embedding_LayerNorm.state_dict():
                            embedding_ln_pretrained_state_dict[k__] = v
                        else:
                            unexpected_keys.append(_k)
                    else:
                        unexpected_keys.append(_k)
                elif self.config.with_pooler and k.startswith('pooler.'):
                    k_ = k[len('pooler.'):]
                    if k_ in self.pooler.state_dict():
                        pooler_pretrained_state_dict[k_] = v
                    else:
                        unexpected_keys.append(_k)
            elif _k.startswith('cls.seq_relationship.') and self.with_rel_head:
                k_ = _k[len('cls.seq_relationship.'):]
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.relationsip_head.caption_image_relationship.state_dict():
                    relationship_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            elif (_k.startswith('cls.predictions.') or _k.startswith('lm_head.')) and self.with_mlm_head:
                k_ = _k[len('cls.predictions.'):] if _k.startswith('cls.predictions.') else _k[len('lm_head.'):]
                if _k.startswith('lm_head.'):
                    if 'dense' in k_ or 'layer_norm' in k_:
                        k_ = 'transform.' + k_
                    if 'layer_norm' in k_:
                        k_ = k_.replace('layer_norm', 'LayerNorm')
                if 'gamma' in k_:
                    k_ = k_.replace('gamma', 'weight')
                if 'beta' in k_:
                    k_ = k_.replace('beta', 'bias')
                if k_ in self.mlm_head.predictions.state_dict():
                    mlm_head_pretrained_state_dict[k_] = v
                else:
                    unexpected_keys.append(_k)
            else:
                unexpected_keys.append(_k)
        if len(unexpected_keys) > 0:
            print("Warnings: Unexpected keys: {}.".format(unexpected_keys))
        self.embedding_LayerNorm.load_state_dict(embedding_ln_pretrained_state_dict)
        self.encoder.load_state_dict(encoder_pretrained_state_dict)
        if self.config.with_pooler and len(pooler_pretrained_state_dict) > 0:
            self.pooler.load_state_dict(pooler_pretrained_state_dict)
        if self.with_rel_head and len(relationship_head_pretrained_state_dict) > 0:
            self.relationsip_head.caption_image_relationship.load_state_dict(relationship_head_pretrained_state_dict)
        if self.with_mlm_head:
            self.mlm_head.predictions.load_state_dict(mlm_head_pretrained_state_dict)


class VisualLinguisticBertMVRCHeadTransform(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHeadTransform, self).__init__(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

        self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


class VisualLinguisticBertMVRCHead(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertMVRCHead, self).__init__(config)

        self.transform = VisualLinguisticBertMVRCHeadTransform(config)
        self.region_cls_pred = nn.Linear(config.hidden_size, config.visual_region_classes)
        self.apply(self.init_weights)

    def forward(self, hidden_states):

        hidden_states = self.transform(hidden_states)
        logits = self.region_cls_pred(hidden_states)

        return logits


class VisualLinguisticBertRelationshipPredictionHead(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBertRelationshipPredictionHead, self).__init__(config)

        self.caption_image_relationship = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, pooled_rep):

        relationship_logits = self.caption_image_relationship(pooled_rep)

        return relationship_logits




