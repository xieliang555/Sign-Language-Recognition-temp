from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field, TabularDataset
import matplotlib.pyplot as plt
import numpy as np


def process_annotations(annotations, csv_dir='/media/xieliang555/新加卷/数据集/phoenix2014-release/phoenix-2014-multisigner/annotations/manual'):
    '''
            this function pad and numericalize the annotation batch
    '''
    if not os.path.exists('.data/train.annotations.csv'):
        csv_file = pd.read_csv(os.path.join(csv_dir, 'train.corpus.csv'))
        data = [csv_file.iloc[i, 0].split('|')[3]
                for i in range(len(csv_file))]
        f = open('.data/train.annotations.csv', 'a')
        for annotation in data:
            f.write(annotation+'\n')
        f.close()

    if not os.path.exists('.data/dev.annotations.csv'):
        csv_file = pd.read_csv(os.path.join(csv_dir, 'dev.corpus.csv'))
        data = [csv_file.iloc[i, 0].split('|')[3]
                for i in range(len(csv_file))]
        f = open('.data/dev.annotations.csv', 'a')
        for annotation in data:
            f.write(annotation+'\n')
        f.close()

    if not os.path.exists('.data/test.annotations.csv'):
        csv_file = pd.read_csv(os.path.join(csv_dir, 'test.corpus.csv'))
        data = [csv_file.iloc[i, 0].split('|')[3]
                for i in range(len(csv_file))]
        f = open('.data/test.annotations.csv', 'a')
        for annotation in data:
            f.write(annotation+'\n')
        f.close()

    TRG = Field(tokenize="spacy",
                tokenizer_language="de",
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_set, dev_set, test_set = TabularDataset.splits(path='.data/',
                                                         train='train.annotations.csv', validation='dev.annotations.csv',
                                                         test='test.annotations.csv', format='csv', fields=[("TRG", TRG)])

    TRG.build_vocab(train_set, min_freq=2)

    return TRG.process(annotations)


def collate_fn(batch):
    '''
            this function pad the variant video sequence length to the
            fixed length and preprocess the annotations for the DataLoader
    '''
    clips = [item['clip'] for item in batch]
    annotations = [item['annotation'].lower().split() for item in batch]
    clips_padded = pad_sequence([clip for clip in clips], batch_first=True)
    annotations = process_annotations(annotations)

    return {'clip': clips_padded, 'annotation': annotations}


def itos(annotations):
    """
            transform numerica to text
            this is the reverse function of process_annotations
    """
    TRG = Field(tokenize='spacy',
                tokenizer_language='de',
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_set, dev_set, test_set = TabularDataset.splits(path='.data/',
                                                         train='train.annotations.csv', validation='dev.annotations.csv',
                                                         test='test.annotations.csv', format='csv', fields=[('TRG', TRG)])

    TRG.build_vocab(train_set, min_freq=2)

    annotations = annotations.transpose(1, 0)
    text = [[TRG.vocab.itos[i] for i in annotation]
            for annotation in annotations]

    return text