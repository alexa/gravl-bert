import os
import torch
import pickle
import argparse
from PIL import Image
import torch.nn as nn
from utils import get_cnn
from Decoder import RNN
from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
# from DataLoader import DataLoader, shuffle_data
import json
import glob
import copy
from transformers import AutoTokenizer
import cv2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('-i')
	parser.add_argument('-model')
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	args = parser.parse_args()

	# with open(os.path.join(args.model, 'vocab.pkl'), 'rb') as f:
	#     vocab = pickle.load(f)

	transform = transforms.Compose([transforms.Resize((224, 224)), 
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])

	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
	special_tokens = ['cubby', 'hoodie', 'joggers']
	special_tokens_dict = {'additional_special_tokens': special_tokens}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	embedding_dim = 512
	vocab_size = 28996 + 3
	hidden_dim = 512
	model_name = args.model
	cnn = get_cnn(architecture = model_name, embedding_dim = embedding_dim)
	lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
	           vocab_size = vocab_size)

	iteration = args.epoch
	cnn_file = 'iter_' + str(iteration) + '_cnn.pth.tar'
	lstm_file = 'iter_' + str(iteration) + '_lstm.pth.tar'
	cnn.load_state_dict(torch.load(os.path.join('../TrainedModels/IC_models', cnn_file)))
	lstm.load_state_dict(torch.load(os.path.join('../TrainedModels/IC_models', lstm_file)))

	cnn.eval()
	cnn.cuda()
	lstm.cuda()

	image_source = '../SIMMC2_data/simmc2_scene_images_dstc10_public/'
	image_lst = glob.glob(image_source + '*.png')
	print(len(image_lst))
	scene_source = '../SIMMC2_data/public/'
	rec = {}

	for idx, imp in enumerate(image_lst):

		img_save = cv2.imread(imp)
		scene_path = scene_source + imp.split('/')[-1].replace('.png', '_scene.json')
		if not os.path.isfile(scene_path):
			scene_path = scene_source + 'm_' + scene_path.split('/')[-1]
		objs = json.load(open(scene_path, 'r'))['scenes'][0]['objects']
		try:
			img = Image.open(imp).convert('RGB')
		except:
			rec[scene_path.split('/')[-1]] = []
			continue
		text_rec = []
		for n, obj in enumerate(objs):
			bbox = [obj['bbox'][0], obj['bbox'][1], obj['bbox'][0]+obj['bbox'][3], obj['bbox'][1]+obj['bbox'][2]]

			image = transform(copy.copy(img).crop(bbox))
			image = image.unsqueeze(0)
			image = Variable(image).cuda()
			cnn_out = cnn(image)
			ids_list = lstm.greedy(cnn_out)
			text = (' '.join(tokenizer.convert_ids_to_tokens(ids_list))).replace('[SEP]', '').replace('[CLS]', '')
			cv2.rectangle(img_save, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
			cv2.putText(img_save, text, (bbox[0], bbox[1]-12), 0, 0.5, (0,255,0), 2)
			text_rec.append(text)

		rec[scene_path.split('/')[-1]] = text_rec

	json.dump(rec, open('../SIMMC2_data/raw_captions.json', 'w'))



