import torch
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# load/download raw snetnence pairs,then tokenize
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

# build vocabulary
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print(len(SRC.vocab))

# define batch iterators/dataloaders
train_ietrator, valid_iterator, test_iterator = BucketIterator.splits(
	(train_data, valid_data, test_data),
	batch_size = 128,
	device = device)


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


