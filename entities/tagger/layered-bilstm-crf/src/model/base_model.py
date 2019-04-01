import os
os.environ['CHAINER_SEED'] = '0'
import numpy as np
import random
random.seed(0)
np.random.seed(0)
import chainer
import re


class ModelBase(chainer.Chain):
    def __init__(self):
        pass

    def load_pretrained(self, path):
        c_found = 0
        c_lower = 0
        c_zeros = 0
        emb_invalid=0
        n_words = len(self.id_to_word)
        pretrained = {}

        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                line = line.rstrip().split(' ')
                try:
                    pretrained[line[0]] = self.xp.array([float(x) for x in line[1:]]).astype(self.xp.float32)
                except:
                    emb_invalid += 1

            if emb_invalid > 0:
                print ('WARNING: {} invalid lines'.format(emb_invalid))

            for i in range(n_words):
                word = self.id_to_word[i]
                if word in pretrained:
                    self.word_embed.W.data[i] = pretrained[word]
                    c_found += 1
                elif word.lower() in pretrained:
                    self.word_embed.W.data[i] = pretrained[word.lower()]
                    c_lower += 1
                elif re.sub('\d', '0', word.lower()) in pretrained:
                    self.word_embed.W.data[i] = pretrained[re.sub('\d', '0', word.lower())]
                    c_zeros += 1

        print ('{} / {} {:.2f} words have been initialized with pretrained embeddings.'.format (
            (c_found + c_lower + c_zeros), n_words, ((c_found + c_lower + c_zeros) / n_words*100)))
        print ('{} found directly, {} after lowercasing, ''{} after lowercasing + zero.'.format(
            c_found, c_lower, c_zeros))
