import os
import numpy as np
import time
import jsonlines
import json
import _pickle as cPickle
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy

import torch
from torch.utils.data import Dataset
# from external.pytorch_pretrained_bert import BertTokenizer, BasicTokenizer
from transformers import AutoTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.mask import generate_instance_mask
from common.nlp.misc import get_align_matrix
from common.utils.misc import block_digonal_matrix
from common.nlp.misc import random_word_with_token_ids
# from common.nlp.roberta import RobertaTokenizer

class VisualDialog_mlm(Dataset):
	def __init__(self, ann_file, root_path, data_path, img_path, adj_path=None, transform=None, test_mode=False,
				 tokenizer=None, pretrained_model_name=None,
				 add_image_as_a_box=False, mask_size=(14, 14),
				 seq_len=500,
				 **kwargs):
		super(SIMMC2Dataset_mlm, self).__init__()


		self.seq_len = seq_len

		categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
					'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
					'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
					'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
					'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
					'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
					'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
					'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
					'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
		self.category_to_idx = {c: i for i, c in enumerate(categories)}
		self.data_path = data_path
		self.root_path = root_path
		self.img_path = img_path
		self.ann_file = os.path.join(data_path, ann_file)
		self.transform = transform
		self.test_mode = test_mode

		self.add_image_as_a_box = add_image_as_a_box
		self.mask_size = mask_size

		if tokenizer is None:
			if pretrained_model_name is None:
				pretrained_model_name = 'bert-base-uncased'
			# if 'roberta' in pretrained_model_name:
			# 	tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
			# else:
			tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		self.tokenizer = tokenizer
		special_tokens = ['[USER]', '[SYS]', '[QRY]']
		special_tokens_dict = {'additional_special_tokens': special_tokens}
		num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

		if adj_path is not None:
			self.adj_mat = pickle.load(open(adj_path, 'rb'))
		else:
			self.adj_mat = None

		self.database = self.load_annotations(self.ann_file)

		self.person_name_id = 0
		if 'train' in ann_file:
			self.img_prefix = 'train2014/COCO_train2014_'
		elif 'val' in ann_file:
			self.img_prefix = 'val2014/COCO_val2014_'

	def load_annotations(self, ann_file):

		# ignore or not find cached database, reload it from annotation file
		print('loading database from {}...'.format(ann_file))
		database = json.load(open(ann_file, 'r'))['data']

		return database

	def convert_metadata(self, obj_meta):
		tokenized_meta = [self.tokenizer(i)['input_ids'] for i in obj_meta]
		max_len = max([len(i) for i in tokenized_meta])
		padded_tokenized_meta = [i + [0] * (max_len - len(i)) for i in tokenized_meta]
		if self.add_image_as_a_box:
			padded_tokenized_meta = [[0] * max_len] + padded_tokenized_meta
		return torch.Tensor(padded_tokenized_meta)


	def __getitem__(self, index):
		# self.person_name_id = 0
		idb = deepcopy(self.database[index])
		idb_cp = deepcopy(idb)

		non_obj_tag = 0 if self.add_image_as_a_box else -1


		idb['text'] = self.tokenizer(idb['text'])['input_ids']
		idb['text'] = torch.Tensor(idb['text'])

		if 'text_lbl' in idb.keys():
			idb['text_lbl'] = self.tokenizer(idb['text_lbl'])['input_ids']
			idb['text_lbl'] = torch.Tensor(idb['text_lbl'])
		else:
			# create masks
			idb['text_lbl'] = deepcopy(idb['text'])
			mask_arr = torch.rand(idb['text'].shape)
			mask_arr = (mask_arr < 0.15) * (idb['text'] != 101) * (idb['text'] != 102)
			idb['text'][mask_arr] = 103

		image = self._load_image(idb['img_fn'])
		w0, h0 = image.size
		boxes = torch.Tensor(idb['bboxes']).float()
		if self.add_image_as_a_box:
			image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1]])
			boxes = torch.cat((image_box, boxes), dim=0).float()


		if self.add_image_as_a_box:
			q_tag = torch.zeros(idb['text'].shape)
		else:
			q_tag = torch.zeros(idb['text'].shape).fill_(-1)

		q_token_type = torch.zeros(idb['text'].shape)

		idb['text'] = torch.cat([idb['text'].unsqueeze(1), q_tag.unsqueeze(1), q_token_type.unsqueeze(1)], dim=1)


		# check box

		# transform
		im_info = torch.tensor([w0, h0, 1.0, 1.0, index])

		if self.transform is not None:
			image, boxes, im_info = self.transform(image, boxes, im_info)

		# clamp boxes
		w = im_info[0].item()
		h = im_info[1].item()
		boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
		boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
		
		outputs = (image, boxes, idb['text'], idb['text_lbl'], im_info)

		return outputs

	def __len__(self):
		return len(self.database)

	def _load_image(self, path):
		path = '0'* (12-len(str(path))) + str(path)
		p = os.path.join(self.img_path, self.img_prefix + path + '.jpg')
		if os.path.isfile(p):
			return Image.open(p).convert('RGB')
		else:
			raise ValueError('Img not exist')


	def _load_json(self, path):
		with open(path, 'r') as f:
			return json.load(f)

	@property
	def data_names(self):
		data_names = ['image', 'boxes', 'mlm_text',
		'mlm_label', 'im_info']	
		return data_names


