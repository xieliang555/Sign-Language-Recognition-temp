import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator

import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np

'''
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
'''


'''
from jiwer import wer

y = "hello world is"
y_ = "hello duck it is"

err = wer(y, y_)
print(err)
'''

'''
from nltk.translate.bleu_score import corpus_bleu

hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
         'ensures', 'that', 'the', 'military', 'always',
        'obeys', 'the', 'commands', 'of', 'the', 'party']
ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        'ensures', 'that', 'the', 'military', 'will', 'forever',
         'heed', 'Party', 'commands']
ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
         'guarantees', 'the', 'military', 'forces', 'always',
         'being', 'under', 'the', 'command', 'of', 'the', 'Party']
ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
         'army', 'always', 'to', 'heed', 'the', 'directions',
        'of', 'the', 'party']

hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
         'interested', 'in', 'world', 'history']
ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
         'because', 'he', 'read', 'the', 'book']

# list_of_references = [[ref1a],[ref2a]]
list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
hypotheses = [hyp1, hyp2]
bleu = corpus_bleu(list_of_references, hypotheses)
print(bleu)
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRG = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

csv_dir = '/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'

train_set, dev_set, test_set = TabularDataset.splits(path=csv_dir, 
        train= 'train.corpus.csv', validation='dev.corpus.csv', 
        test='test.corpus.csv', format='csv', fields=[("TRG", TRG)])



TRG.build_vocab(train_set, min_freq = 2)
# print(help(TRG))

a = '__ON__ LIEB ZUSCHAUER ABEND WINTER GESTERN loc-NORD SCHOTTLAND loc-REGION UEBERSCHWEMMUNG AMERIKA IX'.lower().split()
b= 'loc-WEST WARM loc-WEST BLEIBEN KUEHL'.lower().split()
c= TRG.process([a,b])
print(c.shape)

train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_set, dev_set, test_set),batch_size = 2, device = device)

# print(help(train_iter))

for i, batch in enumerate(train_iter):
        # print(help(batch))

        if i==0:
                break
