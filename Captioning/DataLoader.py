import os
import json
import nltk
import time
import torch
from PIL import Image
import json
from transformers import AutoTokenizer
import random
import numpy as np
class DataLoader():
	def __init__(self, dir_path, mode='train'):
		self.images = None
		self.captions_dict = None
		# self.data = None
		# self.vocab = vocab
		self.data = json.load(open(dir_path, 'r'))
		if mode == 'train':
			self.data = self.data['train']
		elif mode == 'val':
			self.data = self.data['val']
		self.data_len = len(self.data)
		self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
		special_tokens = ['cubby', 'hoodie', 'joggers']
		special_tokens_dict = {'additional_special_tokens': special_tokens}
		num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
		self.image_root = '../SIMMC2_data/simmc2_scene_images_dstc10_public/'

	def gen_data(self):
		inds = [i for i in range(len(self.data))]
		random.shuffle(inds)
		for ind in inds:
			d = self.data[ind]
			try:
				image = Image.open(os.path.join(self.image_root, d['img_path'])).convert('RGB')
			except:
				image = Image.fromarray(np.zeros([224,224]).astype(np.uint8)).convert('RGB')
				print(ind, d['img_path'], 'fail to load')
			boxes = d['boxes']
			captions = [self.tokenizer(c)['input_ids'] for c in d['captions']]

			yield (image, boxes, captions)
