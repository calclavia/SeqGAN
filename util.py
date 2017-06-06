import numpy as np
import json

from constants import *
from tqdm import tqdm

def discount_rewards(r, gamma=0.99):
    """ Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def load_corpus():
    """
    Loads the dataset and returns a list of sequence.
    """
    fpath = 'data/obama.txt'
    with open(fpath, encoding='utf-8') as f:
        text = f.read()
    return text

def load_embedding(word_index):
    """
    Loads a pre-trained word embedding into an embedding matrix
    """
    print('Loading word embeddings...')
    embeddings_index = {}
    fpath = 'data/glove.6B.100d.txt'

    with open(fpath, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_json_dict(fpath):
    with open(fpath, encoding='utf-8') as f:
        return json.loads(f.read())
