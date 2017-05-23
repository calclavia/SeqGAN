from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import random
import numpy as np
import os
import json

from models import *
from constants import *

TEMP = .8

def sample(distribution, temp=1.0):
    distr = np.log(distribution) / temp
    distr = np.exp(distr) / np.sum(np.exp(distr))
    return np.random.choice(MAX_VOCAB, 1, p=distr)[0]

def prep_tokenizer(tokenizer):
    fpath = 'data/obama.txt'
    with open(fpath, encoding='utf-8') as f:
        text = f.read()
        texts = text.split('\n\n')
    tokenizer.fit_on_texts(texts)

def load_embedding(tokenizer):
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

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print (embedding_matrix.shape)
    return embedding_matrix

def main():
    """
    Main function executed to start training on dataset.
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    prep_tokenizer(tokenizer)
    # Load embedding matrix
    embedding_matrix = load_embedding(tokenizer)

    # Create models
    base_model = create_base_model(embedding_matrix)
    generator = create_generator(base_model)

    os.makedirs('out', exist_ok=True)

    # Load in model weights
    generator.load_weights('out/model.h5')

    # Load word index
    idx = tokenizer.word_index
    inv_idx = {v: k for k, v in idx.items()}

    # Generative sampling, store results in results, seed with current results
    results = np.zeros([1, SEQ_LEN])
    seed = np.zeros([1, SEQ_LEN])
    seed[0][0] = random.randint(0,MAX_VOCAB)
    for iter in range(1,SEQ_LEN):
        chosen = generator.predict(seed) 
        distr = chosen[0][iter]
        choice = sample(distr, temp=TEMP)
        results[0][iter-1] = choice
        seed = results

    final_sequence = results[0][:-1].astype(int)
    textual = [inv_idx[word] for word in final_sequence]

    # Print resulting sentence
    print ('\nResulting Sentence, temp = ' + str(TEMP) + ':')
    print (' '.join(textual))

if __name__ == '__main__':
    main()
