import gensim.models
from gensim.models import word2vec
import sys

model = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
model.save_word2vec_format(sys.argv[2], binary=False)