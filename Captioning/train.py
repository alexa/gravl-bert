import os
import torch
import time
import pickle
import argparse
import torch.nn as nn
from Decoder import RNN
from utils import get_cnn
import matplotlib.pyplot as plt
from Vocabulary import Vocabulary
from torchvision import transforms
from torch.autograd import Variable
from Preprocess import load_captions
from DataLoader import DataLoader
import copy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model')
	parser.add_argument('-dir', type = str, default = 'data3.json')
	parser.add_argument('-save_iter', type = int, default = 6)
	parser.add_argument('-learning_rate', type=float, default = 1e-5)
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-gpu_device', type=int)
	parser.add_argument('-hidden_dim', type=int, default = 512)
	parser.add_argument('-embedding_dim', type=int, default = 512)

	args = parser.parse_args()
	train_dir = args.dir
	threshold = 5

	# captions_dict = load_captions(train_dir)
	# vocab = Vocabulary(captions_dict, threshold)
	# with open(os.path.join(args.model, 'vocab.pkl'), 'wb') as f:
	# 	pickle.dump(vocab, f)
	# 	print('dictionary dump')
	dir_path = 'data3.json'
        os.mkdir('../TrainedModels/IC_models')
	transform = transforms.Compose([transforms.Resize((224, 224)), 
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5),
														 (0.5, 0.5, 0.5))
									])

	train_loader = DataLoader(train_dir, 'train')
	val_loader = DataLoader(train_dir, 'val')
	
	print(train_dir + ' loaded')
	# embedding_dim = 512
	vocab_size = 28996 + 3
	hidden_dim = 512
	# learning_rate = 1e-3
	model_name = args.model
	cnn = get_cnn(architecture = model_name, embedding_dim = args.embedding_dim)
	lstm = RNN(embedding_dim = args.embedding_dim, hidden_dim = args.hidden_dim, 
			   vocab_size = vocab_size)
	# cnn.eval()
	# lstm.train()

	# cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
	# lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
	cnn.load_state_dict(torch.load('iter_200_cnn.pth.tar'))
	lstm.load_state_dict(torch.load('iter_200_lstm.pth.tar'))

	
	if torch.cuda.is_available():
		with torch.cuda.device(args.gpu_device):
			cnn.cuda()
			lstm.cuda()
			# iteration = args.epoch
			# cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
			# lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
			# cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file)))
			# lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file)))
	
	criterion = nn.CrossEntropyLoss()
	params = list(cnn.linear.parameters()) + list(lstm.parameters()) 
	optimizer = torch.optim.Adam(params, lr = args.learning_rate)
	num_epochs = 100
	best_loss = 1000

	for epoch in range(num_epochs):
		data = train_loader.gen_data()
		loss_list = []
		for ind, (image, boxes, captions) in enumerate(data):
			n_boxes = len(boxes)
			for n in range(n_boxes):
				optimizer.zero_grad()
				box = boxes[n]
				# box in x, y, w, h
				box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
				im = copy.copy(image).crop(box)
				im = transform(im).cuda().unsqueeze(0)
				caption = torch.Tensor(captions[n]).cuda().long()
				caption_train = caption[:-1] # remove <end>
				cnn_out = cnn(im)
				lstm_out = lstm(cnn_out, caption_train)
				loss = criterion(lstm_out, caption)
				loss.backward()
				optimizer.step()
				loss_list.append(loss.item())
		avg_loss = torch.mean(torch.Tensor(loss_list))	
		print('epoch %d avg_train_loss %f'%(epoch, avg_loss))		
		data = val_loader.gen_data()
		loss_list = []
		with torch.no_grad():
			for ind, (image, boxes, captions) in enumerate(data):
				n_boxes = len(boxes)
				for n in range(n_boxes):
					box = boxes[n]
					# box in x, y, w, h
					box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
					im = copy.copy(image).crop(box)
					im = transform(im).cuda().unsqueeze(0)
					caption = torch.Tensor(captions[n]).cuda().long()
					caption_train = caption[:-1] # remove <end>
					cnn_out = cnn(im)
					lstm_out = lstm(cnn_out, caption_train)
					loss = criterion(lstm_out, caption)
					loss_list.append(loss.item())
		avg_loss = torch.mean(torch.Tensor(loss_list))	
		print('epoch %d avg_val_loss %f'%(epoch, avg_loss))	
		if avg_loss < best_loss:
			print('Saving models')
			torch.save(cnn.state_dict(), os.path.join('../TrainedModels/IC_models', 'iter_%d_cnn.pth.tar'%(epoch)))
			torch.save(lstm.state_dict(), os.path.join('../TrainedModels/IC_models', 'iter_%d_lstm.pth.tar'%(epoch)))
			best_loss = avg_loss

