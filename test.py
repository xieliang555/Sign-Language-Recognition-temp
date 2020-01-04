import torch

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