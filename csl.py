# continuous sign language recognition

import os
import time
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import utils

# loading dataset
csv_root =  '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'
clip_root = '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px'

transform = transforms.Compose([transforms.ToPILImage(),
								transforms.RandomCrop(224, pad_if_needed=True),
								transforms.ToTensor()])


train_csv_dir = os.path.join(csv_root, 'train.corpus.csv')
train_root_dir = os.path.join(clip_root, 'train')
train_set = utils.PhoenixDataset(train_csv_dir, train_root_dir, transform)
train_loader = torch.utils.data.DataLoader(train_set, 
	batch_size = 1, shuffle = True, num_workers = 0, collate_fn= utils.collate_fn)


dev_csv_dir = os.path.join(csv_root, 'dev.corpus.csv')
dev_root_dir = os.path.join(clip_root, 'dev')
dev_set = utils.PhoenixDataset(dev_csv_dir, dev_root_dir, transform)
dev_loader = torch.utils.data.DataLoader(dev_set,
	batch_size = 2, shuffle = True, num_workers = 0, collate_fn= utils.collate_fn)


test_csv_dir = os.path.join(csv_root, 'test.corpus.csv')
test_root_dir = os.path.join(clip_root, 'test')
test_set = utils.PhoenixDataset(test_csv_dir, test_root_dir, transform)
test_loader = torch.utils.data.DataLoader(test_set,
	batch_size = 4, shuffle = True, num_workers = 0, collate_fn= utils.collate_fn)



# define model
for batch_idx, sample_batch in enumerate(train_loader):
	print(utils.itos(sample_batch['annotation']))

	if batch_idx == 0:
		break


# 整理nlp custom dataset方法， type dir help方法