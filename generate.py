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
    results = [0 for _ in range(SEQ_LEN - 1)] + [random.randint(0, MAX_VOCAB)]

    for i in range(SEQ_LEN):
        # Take the last SEQ_LEN results and feed it in.
        last_results = results[-SEQ_LEN:]
        # Create batch dimension
        feed = np.reshape(last_results, [1, -1])

        distr = generator.predict(feed)
        # Pick the last result
        distr = distr[0][-1]
        choice = sample(distr, temp=TEMP)
        results.append(choice)

    # Ignore null words
    textual = [inv_idx[word] for word in results if word != 0]

    # Print resulting sentence
    print ('\nResulting Sentence, temp = ' + str(TEMP) + ':')
    print (' '.join(textual))

if __name__ == '__main__':
    main()
