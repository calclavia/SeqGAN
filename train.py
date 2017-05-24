from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import numpy as np
import os
import json

from models import *
from constants import *
from util import *
from generate import generate

def load_data(tokenizer):
    """
    Loads training data from file and processes it.
    """
    print('Loading data...')
    # Prepare the tokenizer
    text = load_corpus()

    # Split based on sentences
    texts = sent_tokenize(text)

    tokenizer.fit_on_texts(texts)

    # A list of sequences. Each sequence has a different length.
    sequences = tokenizer.texts_to_sequences(texts)

    # Add null character to beginning of each sequence
    sequences = [[0] + x for x in sequences]

    print('Average sequence length:', np.mean([len(seq) for seq in sequences]))
    print('Max sequence length:', max([len(seq) for seq in sequences]))
    print('Found {} unique tokens.'.format(len(tokenizer.word_index)))

    # Create training data and target data.
    # Truncates and pads sequences so that they're the same length.
    train_data = pad_sequences([x[:-1] for x in sequences], maxlen=SEQ_LEN)
    # Target data is training data shfited by one word
    target_data = pad_sequences([x[1:] for x in sequences], maxlen=SEQ_LEN)
    # TODO: Vocab size seems to be a limitation in what outputs it can make...

    # Convert to one-hot vector
    target_data = np.array([to_categorical(seq, MAX_VOCAB) for seq in target_data])

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of target_data tensor:', target_data.shape)
    return train_data, target_data

def main():
    """
    Main function executed to start training on dataset.
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    # Load data
    train_data, target_data = load_data(tokenizer)
    # Load embedding matrix
    embedding_matrix = load_embedding(tokenizer.word_index)

    # Create models
    base_model = create_base_model(embedding_matrix)
    generator = create_generator(base_model)
    discriminator = create_discriminator(base_model)

    os.makedirs('out', exist_ok=True)

    # Write word index to file for generation
    with open('out/word_index.json', 'w') as f:
        json.dump(tokenizer.word_index, f)

    generator.summary()

    # MLE Pre-training
    print('Pre-training generator...')
    generator.fit(
        train_data,
        target_data,
        validation_split=0.1,
        epochs=1000,
        batch_size=128,
        callbacks=[
            ModelCheckpoint('out/generator.h5', save_best_only=True),
            EarlyStopping(patience=5)
        ]
    )
    
    # TODO: The discriminator may catestrophically interfere with shared model
    # TODO: Consider freezing weights of perform parallel training.

    # Generate fake samples
    num_real = train_data.shape[0]
    print('Generating {} fake samples...'.format(NUM_FAKE))
    fake_samples = [generate(generator, SEQ_LEN) for i in tqdm(range(NUM_FAKE))]

    # Generate discriminator train and targets
    d_train = np.concatenate([np.array(fake_samples), train_data], axis=0)
    d_targets = np.concatenate([np.zeros((NUM_FAKE,)), np.ones((num_real,))])

    print('Pre-training discriminator...')
    discriminator.fit(
        d_train,
        d_targets,
        validation_split=0.1,
        epochs=1000,
        batch_size=128,
        callbacks=[
            ModelCheckpoint('out/discriminator.h5', save_best_only=True),
            EarlyStopping(patience=5)
        ]
    )

if __name__ == '__main__':
    main()
