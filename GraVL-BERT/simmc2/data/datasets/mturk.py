# Original Copyright (c) 2019 Weijie Su. Licensed under the MIT License.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

class mturkDataset(Dataset):
	def __init__(self, ann_file, root_path, data_path, img_path, adj_path=None, transform=None, test_mode=False,
				 tokenizer=None, pretrained_model_name=None,
				 add_image_as_a_box=False, mask_size=(14, 14),
				 seq_len=500, use_mentioned_lbl=False, use_3d_coords=False, use_turn_lbl=False,
				 **kwargs):
		super(mturkDataset, self).__init__()


		self.seq_len = seq_len

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

		
		self.use_mentioned_lbl = use_mentioned_lbl
		self.use_3d_coords = use_3d_coords
		self.use_turn_lbl = use_turn_lbl

	def load_annotations(self, ann_file):

		# ignore or not find cached database, reload it from annotation file
		print('loading database from {}...'.format(ann_file))
		database = json.load(open(ann_file, 'r'))

		return database

	def enlarge_bbox(self, bbox, w0, h0):
		w, h = bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]
		bbox[:,0] = (bbox[:,0] - 0.15*w).clamp(min=0)
		bbox[:,2] = (bbox[:,2] + 0.15*w).clamp(max=w0)
		bbox[:,1] = (bbox[:,1] - 0.15*h).clamp(min=0)
		bbox[:,3] = (bbox[:,3] + 0.15*h).clamp(max=h0)
		return bbox

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


		"""
		questions and input ids
		"""

		idb['question'] = self.tokenizer(idb['question'])['input_ids']
		# truncate text to seq_len
		q = idb['question'][0]
		if len(idb['question']) + 1 > self.seq_len:
			idb['question'] = idb['question'][0:1] + idb['question'][-300:]

		question = idb['question']
		if self.add_image_as_a_box:
			q_tag = [0 for i in question]
		else:
			q_tag = [-1 for i in question]
		q_token_type = [0 for i in question]
		q_with_tag = list(zip(question, q_tag, q_token_type))

		"""
		images
		"""
		images = []
		for img in idb['img_fn']:
			img = self._load_image(img)
			img = img.resize([256, 256])
			images.append(img)
		dst = Image.new('RGB', (256*3, 256))
		for i in range(3):
			dst.paste(images[i], (256*i, 0))
		boxes = torch.tensor([[0, 0, 256, 255], [256, 0, 512, 255], [512, 0, 767, 255]])


		"""
		labels
		"""

		label = np.array(idb['answer_choices'])

		"""
		bboxes and 3d coords
		"""

		# if len(objects) > 0:
		# 	boxes = torch.tensor(idb['boxes']).float()
		# 	# boxes = self.enlarge_bbox(boxes, w0, h0)
		# 	if self.use_3d_coords:
		# 		if len(idb['3d_pos']) == 0:
		# 			coords_3d = torch.zeros([boxes.shape[0], 3])
		# 		else:
		# 			coords_3d = torch.tensor(idb['3d_pos']).float()
		# 	else:
		# 		coords_3d = None


		"""
		mentioned label
		"""

		# if self.use_mentioned_lbl:
		# 	# mentioned_lbl = np.zeros([1, len(objects)])
		# 	# mentioend_inds = [idb['objects'].index(i) for i in idb['mentioned_lbl']]
		# 	# mentioned_lbl[0, mentioend_inds] = 1
		# 	# mentioned_lbl = mentioned_lbl.reshape([-1])
		# 	mentioned_lbl = [idb['question'].count(self.tokenizer('[USER]')['input_ids'][1]) - idb['mentioned_lbl'][i] if i in idb['mentioned_lbl'] else -1 for i in idb['objects']]
		# 	mentioned_lbl = np.array(mentioned_lbl).reshape([-1])
		# else:
		# 	mentioned_lbl = None


		image_box = torch.as_tensor([[0, 0, 767, 256]])
		boxes = torch.cat((image_box, boxes), dim=0).float()


		"""
		transform
		"""
		im_info = torch.tensor([768, 256, 1.0, 1.0, index])

		if self.transform is not None:
			image, boxes, im_info = self.transform(dst, boxes, im_info)

		"""
		clamp boxes
		"""

		w = im_info[0].item()
		h = im_info[1].item()
		boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
		boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

		

		obj_meta = self.convert_metadata(idb['obj_meta'])

		
		"""
		adjacency matrices
		"""

		# adj = self.adj_mat[idb['img_fn']]
		# if self.add_image_as_a_box:
		# 	base = np.zeros([len(objects), len(objects)])
		# 	base[1:, 1:] = adj
		# 	base[:, 0] = 5
		# 	base[0, :] = 5
		# adj = base
		# if adj.shape[0] != len(objects):
		# 	raise ValueError('adj size and n_obj do not match!')

		objects = [0, 1, 2]
		answer_choices, adj, coords_3d, mentioned_lbl, turn_lbl = [], np.zeros([3,3]), [], [], []

		outputs = (image, boxes, objects, q_with_tag, answer_choices, label, im_info, obj_meta, adj)

		return outputs

	def __len__(self):
		return len(self.database)

	def _load_image(self, path):
		p = os.path.join(self.img_path, path.replace('{asin_images_dir_path}/', ''))
		if os.path.isfile(p):
			return Image.open(p)
		else:
			raise ValueError()


	def _load_json(self, path):
		with open(path, 'r') as f:
			return json.load(f)

	@property
	def data_names(self):
		data_names = ['image', 'boxes', 'objects',
			'question', 'answer', 'label',
			'im_info', 'obj_meta', 'adj_mat']


		return data_names


