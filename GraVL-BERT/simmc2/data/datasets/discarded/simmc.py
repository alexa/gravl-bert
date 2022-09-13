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

class SIMMC2Dataset(Dataset):
	def __init__(self, ann_file, root_path, data_path, img_path, adj_path=None, transform=None, test_mode=False,
				 tokenizer=None, pretrained_model_name=None,
				 add_image_as_a_box=False, mask_size=(14, 14),
				 seq_len=500,
				 **kwargs):
		super(SIMMC2Dataset, self).__init__()


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

	# def retokenize_and_convert_to_ids_with_tag(self, tokens, objects_replace_name, non_obj_tag=-1):
	# 	parsed_tokens = []
	# 	tags = []
	# 	align_ids = []
	# 	raw = []
	# 	align_id = 0
	# 	for mixed_token in tokens:
	# 	    if isinstance(mixed_token, list):
	# 	        tokens = [objects_replace_name[o] for o in mixed_token]
	# 	        retokenized_tokens = self.tokenizer.tokenize(tokens[0])
	# 	        raw.append(tokens[0])
	# 	        tags.extend([mixed_token[0] + non_obj_tag + 1 for _ in retokenized_tokens])
	# 	        align_ids.extend([align_id for _ in retokenized_tokens])
	# 	        align_id += 1
	# 	        for token, o in zip(tokens[1:], mixed_token[1:]):
	# 	            retokenized_tokens.append('and')
	# 	            tags.append(non_obj_tag)
	# 	            align_ids.append(align_id)
	# 	            align_id += 1
	# 	            re_tokens = self.tokenizer.tokenize(token)
	# 	            retokenized_tokens.extend(re_tokens)
	# 	            tags.extend([o + non_obj_tag + 1 for _ in re_tokens])
	# 	            align_ids.extend([align_id for _ in re_tokens])
	# 	            align_id += 1
	# 	            raw.extend(['and', token])
	# 	        parsed_tokens.extend(retokenized_tokens)
	# 	    else:
	# 	        if self.basic_align:
	# 	            # basic align
	# 	            basic_tokens = self.basic_tokenizer.tokenize(mixed_token)
	# 	            raw.extend(basic_tokens)
	# 	            for t in basic_tokens:
	# 	                retokenized_tokens = self.tokenizer.tokenize(t)
	# 	                parsed_tokens.extend(retokenized_tokens)
	# 	                align_ids.extend([align_id for _ in retokenized_tokens])
	# 	                tags.extend([non_obj_tag for _ in retokenized_tokens])
	# 	                align_id += 1
	# 	        else:
	# 	            # fully align to original tokens
	# 	            raw.append(mixed_token)
	# 	            retokenized_tokens = self.tokenizer.tokenize(mixed_token)
	# 	            parsed_tokens.extend(retokenized_tokens)
	# 	            align_ids.extend([align_id for _ in retokenized_tokens])
	# 	            tags.extend([non_obj_tag for _ in retokenized_tokens])
	# 	            align_id += 1
	# 	ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
	# 	ids_with_tag = list(zip(ids, tags, align_ids))

	# 	return ids_with_tag, raw


	def __getitem__(self, index):
		# self.person_name_id = 0
		idb = deepcopy(self.database[index])
		idb_cp = deepcopy(idb)

		non_obj_tag = 0 if self.add_image_as_a_box else -1


		idb['question'] = self.tokenizer(idb['question'])['input_ids']

		answer_choices = []
		for a in idb['answer_choices']:
			answer_choices.append(idb['objects'].index(a))

		idb['rationale_choices'] = None
		# truncate text to seq_len
		q = idb['question'][0]
		if len(idb['question']) + 1 > self.seq_len:
			idb['question'] = idb['question'][0:1] + idb['question'][-300:]

		image = self._load_image(idb['img_fn'])
		w0, h0 = image.size

		
		objects = [i for i in range(len(idb['objects']))]
		# extract bounding boxes and instance masks in metadata
		boxes = torch.zeros((len(objects), 6))
		label = np.zeros([1, len(objects)])
		label[0, answer_choices] = 1
		label = label.reshape([-1])

		if len(objects) > 0:
			boxes = torch.tensor(idb['boxes']).float()
		if self.add_image_as_a_box:
			image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1]])
			boxes = torch.cat((image_box, boxes), dim=0).float()
			objects = [0] + objects
			label = np.concatenate([np.array([1]), label])


		question = idb['question']
		


		if self.add_image_as_a_box:
			q_tag = [0 for i in question]
		else:
			q_tag = [-1 for i in question]
		q_token_type = [0 for i in question]

		q_with_tag = list(zip(question, q_tag, q_token_type))


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

		

		if np.sum(label) == 0:
			print(label)
			print(answer_choices)
			print(idb['objects'])
			print(idb['answer_choices'])
			raise ValueError()

		obj_meta = self.convert_metadata(idb['obj_meta'])
		

		if self.adj_mat is None:
			if not self.test_mode:
				outputs = (image, boxes, objects, q_with_tag, answer_choices, label, im_info, obj_meta)
			else:
				outputs = (image, boxes, objects, q_with_tag, answer_choices, label, im_info, obj_meta)
		else:
			adj = self.adj_mat[idb['img_fn']]
			if self.add_image_as_a_box:
				base = np.zeros([len(objects), len(objects)])
				base[1:, 1:] = adj
			adj = base
			if adj.shape[0] != len(objects):
				raise ValueError('adj size and n_obj do not match!')
			if not self.test_mode:
				outputs = (image, boxes, objects, q_with_tag, answer_choices, label, im_info, obj_meta, adj)
			else:
				outputs = (image, boxes, objects, q_with_tag, answer_choices, label, im_info, obj_meta, adj)

		return outputs

	def __len__(self):
		return len(self.database)

	def _load_image(self, path):
		p = os.path.join(self.img_path, path + '.png')
		if os.path.isfile(p):
			return Image.open(p)
		elif os.path.isfile(os.path.join(self.img_path, path.replace('m_', '') + '.png')):
			return Image.open(os.path.join(self.img_path, path.replace('m_', '') + '.png'))


	def _load_json(self, path):
		with open(path, 'r') as f:
			return json.load(f)

	@property
	def data_names(self):
		if self.adj_mat is None:
			if not self.test_mode:
				data_names = ['image', 'boxes', 'objects',
				'question', 'answer', 'label', 
				'im_info', 'obj_meta']
			else:
				data_names = ['image', 'boxes', 'objects',
				'question', 'answer', 'label',
				'im_info', 'obj_meta']
		else:
			if not self.test_mode:
				data_names = ['image', 'boxes', 'objects',
				'question', 'answer', 'label', 
				'im_info', 'obj_meta', 'adj_mat']
			else:
				data_names = ['image', 'boxes', 'objects',
				'question', 'answer', 'label',
				'im_info', 'obj_meta', 'adj_mat']			

		return data_names


