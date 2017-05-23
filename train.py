from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import json

from models import *
from constants import *

def load_data(tokenizer):
    """
    Loads training data from file and processes it.
    """
    print('Loading data...')
    fpath = 'data/obama.txt'

    with open(fpath, encoding='utf-8') as f:
        text = f.read()
        # Split large text into paragraphs
        texts = text.split('\n\n')

    # Tokenize words
    tokenizer.fit_on_texts(texts)
    # A list of sequences. Each sequence has a different length.
    sequences = tokenizer.texts_to_sequences(texts)

    print('Average sequence length:', np.mean([len(seq) for seq in sequences]))
    print('Found {} unique tokens.'.format(len(tokenizer.word_index)))

    # Create training data and target data.
    # Truncates and pads sequences so that they're the same length.
    train_data = pad_sequences([x[:-1] for x in sequences], maxlen=SEQ_LEN)
    # Target data is training data shfited by one word
    target_data = pad_sequences([x[1:] for x in sequences], maxlen=SEQ_LEN)

    # Convert to one-hot vector
    target_data = np.array([to_categorical(seq, MAX_VOCAB) for seq in target_data])

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of target_data tensor:', target_data.shape)
    return train_data, target_data

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

    return embedding_matrix

def main():
    """
    Main function executed to start training on dataset.
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    # Load data
    train_data, target_data = load_data(tokenizer)
    # Load embedding matrix
    embedding_matrix = load_embedding(tokenizer)

    # Create models
    base_model = create_base_model(embedding_matrix)
    generator = create_generator(base_model)

    os.makedirs('out', exist_ok=True)

    # Write word index to file for generation
    with open('out/word_index.json', 'w') as f:
        json.dump(tokenizer.word_index, f)

    # MLE Training
    generator.fit(
        train_data,
        target_data,
        validation_split=0.1,
        epochs=1000,
        batch_size=128,
        callbacks=[ModelCheckpoint('out/model.h5', save_best_only=True)]
    )

if __name__ == '__main__':
    main()
