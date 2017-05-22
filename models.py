from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Activation

from constants import *

def create_base_model():
    """
    Base model shared by generator and discriminator.
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = seq_input

    x = Embedding(NUM_VOCAB, NUM_UNITS)(x)
    x = LSTM(NUM_UNITS)(x)

    return Model(seq_input, x)

def create_generator(base_model):
    """
    Generator model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = base_model(seq_input)

    model = Model(seq_input, x)
    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model
