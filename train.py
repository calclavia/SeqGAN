from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

from models import *
from constants import *

def load_data():
    """
    Loads training data from file and processes it.
    """
    print('Loading data...')
    fpath = 'data/obama.txt'

    with open(fpath, encoding='utf-8') as f:
        # Split large text into paragraphs
        texts = f.read().split('\n\n')

    # Tokenize words
    tokenizer = Tokenizer(num_words=NUM_VOCAB)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    print('Found {} unique tokens.'.format(len(tokenizer.word_index)))

    # Create training data and target data.
    # Target data is training data shfited by one word
    train_data = pad_sequences(sequences[:][:-1], maxlen=SEQ_LEN)
    target_data = pad_sequences(sequences[:][1:], maxlen=SEQ_LEN)

    assert train_data.shape == target_data.shape

    print('Shape of train_data/target_data tensor:', train_data.shape)
    return train_data, target_data

def main():
    load_data()

    base_model = create_base_model()
    generator = create_generator(base_model)

if __name__ == '__main__':
    main()
