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
from util import *

def sample(distribution, temp=1.0):
    distr = np.log(distribution) / temp
    distr = np.exp(distr) / np.sum(np.exp(distr))
    return np.random.choice(MAX_VOCAB, 1, p=distr)[0]

def main():
    """
    Main function executed to start training on dataset.
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB)

    # TODO: Generator shouldn't need to load the dataset
    texts = load_corpus()
    tokenizer.fit_on_texts(texts)

    # Load embedding matrix
    embedding_matrix = load_embedding(tokenizer.word_index)

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
    seed[0][0] = random.randint(0, MAX_VOCAB)

    for i in range(1, SEQ_LEN):
        chosen = generator.predict(seed)
        distr = chosen[0][i]
        choice = sample(distr, temp=TEMP)
        results[0][i-1] = choice
        seed = results

    final_sequence = results[0][:-1].astype(int)
    textual = [inv_idx[word] for word in final_sequence]

    # Print resulting sentence
    print ('\nResulting Sentence, temp = ' + str(TEMP) + ':')
    print (' '.join(textual))

if __name__ == '__main__':
    main()
