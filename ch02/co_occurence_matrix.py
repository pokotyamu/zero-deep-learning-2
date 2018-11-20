import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess
from common.util import create_co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(create_co_matrix(corpus, len(word_to_id)))
