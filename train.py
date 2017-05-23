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
        text = f.read()
        print('Corpus char length:', len(text))
        texts = text.split('\n\n')

    # Tokenize words
    tokenizer = Tokenizer(num_words=NUM_VOCAB)
    tokenizer.fit_on_texts(texts)
    # A list of sequences. Each sequence has a different length.
    sequences = tokenizer.texts_to_sequences(texts)

    print('Found {} unique tokens.'.format(len(tokenizer.word_index)))

    # Create training data and target data.
    # Truncates and pads sequences so that they're the same length.
    train_data = pad_sequences(sequences[:][:-1], maxlen=SEQ_LEN)
    # Target data is training data shfited by one word
    target_data = pad_sequences(sequences[:][1:], maxlen=SEQ_LEN)
    # Convert to one-hot vector
    target_data = np.array([to_categorical(seq, NUM_VOCAB) for seq in target_data])

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of target_data tensor:', target_data.shape)
    return train_data, target_data

def main():
    train_data, target_data = load_data()

    base_model = create_base_model()
    generator = create_generator(base_model)

    # MLE Training
    generator.fit(train_data, target_data, validation_split=0.1, epochs=100)

if __name__ == '__main__':
    main()
