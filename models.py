from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation

from constants import *

def create_base_model():
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = seq_input
    x = LSTM(NUM_UNITS)(x)

    return Model(seq_input, x)
