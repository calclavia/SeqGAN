from keras.models import Model
from keras.layers import *

from constants import *

def create_base_model(embedding_matrix):
    """
    Base model shared by generator and discriminator.
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    # Conver to embedding space
    x = Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False
    )(seq_input)

    # LSTM with dropout
    x = LSTM(NUM_UNITS, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(NUM_UNITS, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    return Model(seq_input, x)

def create_generator(base_model):
    """
    Generator model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = base_model(seq_input)

    # Prediction
    x = Dense(MAX_VOCAB)(x)
    x = Activation('softmax')(x)

    model = Model(seq_input, x)
    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model
