from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

from models import *
from constants import *

def load_data():
    fpath = 'data/obama.txt'
    f = open(fpath, encoding='latin-1')
    texts = f.read()

    # Tokenize words
    tokenizer = Tokenizer(num_words=NUM_VOCAB)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=SEQ_LEN)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

def main():
    load_data()

    base_model = create_base_model()
    generator = create_generator(base_model)

if __name__ == '__main__':
    main()
