import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field, TabularDataset

import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import pandas as pd


__all__ = [

	'process_annotations', 'collate_fn', 'PhoenixDataset'
]



def process_annotations(annotations, csv_dir='/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'):
	'''
		this function pad and numericalize the annotation batch
	'''
	if not os.path.exists('.data/train.annotations.csv'):
		csv_file = pd.read_csv(os.path.join(csv_dir, 'train.corpus.csv'))
		data = [csv_file.iloc[i,0].split('|')[3] for i in range(len(csv_file))]
		f = open('.data/train.annotations.csv','a')
		for annotation in data:
			f.write(annotation+'\n')
		f.close()

	if not os.path.exists('.data/dev.annotations.csv'):
		csv_file = pd.read_csv(os.path.join(csv_dir,'dev.corpus.csv'))
		data = [csv_file.iloc[i,0].split('|')[3] for i in range(len(csv_file))]
		f = open('.data/dev.annotations.csv','a')
		for annotation in data:
			f.write(annotation+'\n')
		f.close()

	if not os.path.exists('.data/test.annotations.csv'):
		csv_file = pd.read_csv(os.path.join(csv_dir,'test.corpus.csv'))
		data = [csv_file.iloc[i,0].split('|')[3] for i in range(len(csv_file))]
		f = open('.data/test.annotations.csv','a')
		for annotation in data:
			f.write(annotation+'\n')
		f.close()



	TRG = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

	train_set, dev_set, test_set = TabularDataset.splits(path ='.data/', 
		train='train.annotations.csv', validation = 'dev.annotations.csv', 
		test = 'test.annotations.csv',format = 'csv', fields=[("TRG", TRG)])

	TRG.build_vocab(train_set, min_freq = 2)

	return TRG.process(annotations)




def collate_fn(batch):
	'''
		this function pad the variant sequence length to 
		fixed length for DataLoader
	'''
	clips = [item['clip'] for item in batch ]
	annotations = [item['annotation'].lower().split() for item in batch]
	clips_padded = pad_sequence([clip for clip in clips], batch_first = True)
	annotations = process_annotations(annotations)

	return {'clip': clips_padded, 'annotation': annotations}



class PhoenixDataset(object):
	"""
		custom dataset class for RWTH-Weather-Phoenix 2014
	"""
	def __init__(self, csv_dir, root_dir, transforms):
		super(PhoenixDataset, self).__init__()
		self.csv_file = pd.read_csv(csv_dir)
		self.root_dir = root_dir
		self.transforms = transforms


	def __len__(self):
		return len(self.csv_file)


	def __getitem__(self, idx):
		if(torch.is_tensor(idx)):
			idx = idx.tolist()

		clip_path = os.path.join(self.root_dir ,self.csv_file.iloc[idx, 0].split('|')[1])
		collection = io.imread_collection(clip_path)
		clip = torch.zeros(len(collection),3,224,224)
		for i, img in enumerate(collection):
			clip[i,:,:,:] = self.transforms(img)

		annotation = self.csv_file.iloc[idx, 0].split('|')[3]
		sample = {'clip': clip, 'annotation': annotation}

		return sample


	def statistic_analysis(self):
		'''
		analysis the dataset
		'''
		size = len(self.csv_file)
		annotations_len = [len(self.csv_file.iloc[i,0].split('|')[3].split()) for i in range(size)]
		print('corpus pairs: {}\n'.format(size))
		print('max_anotation_length: {}'.format(max(annotations_len)))
		print('min_anotation_length: {}'.format(min(annotations_len)))
		print('ave_anotation_length: {}\n'.format(sum(annotations_len)/size))
		clip_path = os.path.join(self.root_dir, self.csv_file.iloc[0,0].split('|')[1])
		collection = io.imread_collection(clip_path)
		frame_size = collection[0].shape
		print('frame_size: {}'.format(frame_size))
		clip_pathes = [os.path.join(self.root_dir, self.csv_file.iloc[i,0].split('|')[1]) for i in range(size)]
		clip_len = [len(glob.glob(i)) for i in clip_pathes]
		print('max_clip_length: {}'.format(max(clip_len)))
		print('min_clip_length: {}'.format(min(clip_len)))
		print('ave_clip_length: {}'.format(sum(clip_len)/size))

		


if __name__ == '__main__':

	csv_root =  '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'
	clip_root = '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px'

	# test set
	print('================ test set ==============')
	test_csv_dir = os.path.join(csv_root, 'test.corpus.csv')
	test_root_dir = os.path.join(clip_root, 'test')
	test_set = PhoenixDataset(test_csv_dir, test_root_dir, transforms=None)
	test_set.statistic_analysis()

	# dev set
	print('================ dev set ==============')
	dev_csv_dir = os.path.join(csv_root, 'dev.corpus.csv')
	dev_root_dir = os.path.join(clip_root, 'dev')
	dev_set = PhoenixDataset(dev_csv_dir, dev_root_dir, transforms=None)
	dev_set.statistic_analysis()

	# training set
	print('================ train set ==============')
	train_csv_dir = os.path.join(csv_root, 'train.corpus.csv')
	train_root_dir = os.path.join(clip_root, 'train')
	train_set = PhoenixDataset(train_csv_dir, train_root_dir, transforms=None)
	train_set.statistic_analysis()


	process_annotations()