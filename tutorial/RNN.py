import torch
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# field is the object for tokenize, padding and numericalize. 
# The arguments define the tokenizer, language, and padding information
SRC = Field(tokenize = 'spacy',
			tokenizer_language = 'de',
			init_token = '<sos>',
			end_token = '<eos>',
			lower = True)

TRG = Field(tokenize = 'spacy',
			tokenizer_language = 'en',
			init_token = '<sos>',
			end_token = '<eos>',
			lower = True)


train_data, dev_data, test_data = Multi30k.splits(exts= ('.de', '.en'),
	fields = (SRC, TRG), root = '~/data/Multi30k', train = 'train', validation = 'val', test = 'test2016') 


SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)










SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

# load/download raw snetnence pairs,then tokenize using space and en/de
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))


# build vocabulary
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
# print(SRC.vocab.stoi)
print(SRC.vocab.itos)


# define batch iterators/dataloaders
train_ietrator, valid_iterator, test_iterator = BucketIterator.splits(
	(train_data, valid_data, test_data),
	batch_size = 2,
	device = device)


for i, batch in enumerate(train_ietrator):
	print(batch.src.shape)
	print(batch.trg.shape)
	print(batch.src)
	print(batch.trg)

	if i==0:
		break


# define encoder class
class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, 
				input_dim: int,
				emd_dim: int,
				enc_hid_dim: int,
				dec_hid_dim: int,
				dropout: float):
		super(Encoder, self).__init__()

		self.input_dim = input_dim		# the vocabulary len
		self.emd_dim = emd_dim		# the emdedding feature/vector dimension   
		self.enc_hid_dim = enc_hid_dim		# the encoder hidden feature/vector dimension
		self.dec_hid_dim = dec_hid_dim		# the concatenated bidirectional encoder hidden feature/vector dimension
		self.dropout = dropout

	def forward(self, src: Tensor) -> Tuple[Tensor]:
		
		pass


