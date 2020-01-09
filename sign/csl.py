# continuous sign language recognition

import os
import time

import torch
from torchvision import transforms

import utils

# loading dataset

csv_root =  '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'
clip_root = '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px'

transform = transforms.Compose([transforms.ToPILImage(),
								transforms.RandomCrop(224, pad_if_needed=True),
								transforms.ToTensor()])

print("loading training dataset...")
since = time.time()
train_csv_dir = os.path.join(csv_root, 'train.corpus.csv')
train_root_dir = os.path.join(clip_root, 'train')
train_set = utils.PhoenixDataset(train_csv_dir, train_root_dir, transform)
train_loader = torch.utils.data.DataLoader(train_set, 
	batch_size = 4, shuffle = True, num_workers = 4, collate_fn= utils.collate_fn)
print("training dataset loaded, time duration: {}\n".format(time.time()-since))

print("lodaing developing dataset...")
since = time.time()
dev_csv_dir = os.path.join(csv_root, 'dev.corpus.csv')
dev_root_dir = os.path.join(clip_root, 'dev')
dev_set = utils.PhoenixDataset(dev_csv_dir, dev_root_dir, transform)
dev_loader = torch.utils.data.DataLoader(dev_set,
	batch_size = 2, shuffle = True, num_workers = 4, collate_fn= utils.collate_fn)
print("developing dataset loaded, time duration: {}\n".format(time.time()-since))

print("loading test dataset...")
since = time.time()
test_csv_dir = os.path.join(csv_root, 'test.corpus.csv')
test_root_dir = os.path.join(clip_root, 'test')
test_set = utils.PhoenixDataset(test_csv_dir, test_root_dir, transform)
test_loader = torch.utils.data.DataLoader(test_set,
	batch_size = 2, shuffle = True, num_workers = 4, collate_fn= utils.collate_fn)
print("test dataset loaded, time duration: {}\n".format(time.time()-since))


# define model
print("loading first batch")
since = time.time()
for batch_idx, sample_batch in enumerate(train_loader):
	print(batch_idx)
	print("loaded, time duration: {}\n".format(time.time()-since))

	if batch_idx == 1:
		break



# 检查sample是否正确
# 整理nlp custom dataset方法， type help方法
# 优化代码（读取速度，代码可读性）