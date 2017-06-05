from keras.models import Model
from keras.layers import *
from keras import backend as K

from constants import *

def pg_loss(advantage):
    def f(y_true, y_pred):
        """
        Policy gradient loss
        """
        # L = \sum{A * log(p)}
        responsible_outputs = K.sum(y_true * y_pred, axis=1)
        return -K.sum(advantage * K.log(responsible_outputs))
    return f

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

    # x = Dropout(0.2)(x)

    # LSTM with dropout
    x = LSTM(NUM_UNITS, return_sequences=True)(x)
    # x = Dropout(0.5)(x)

    return Model(seq_input, x)

def create_generator(base_model):
    """
    Generator model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = base_model(seq_input)

    x = LSTM(NUM_UNITS)(x)#, return_sequences=True)(x)
    # x = Dropout(0.5)(x)

    # Prediction (probability)
    x = Dense(MAX_VOCAB)(x)
    x = Activation('softmax')(x)

    model = Model(seq_input, x)
    return model

def mle(generator):
    """
    Wraps the generator in an MLE model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = generator(seq_input)

    model = Model(seq_input, x)
    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model

def pg(generator):
    """
    Wraps the generator in a policy gradient model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')
    # Advantages for loss function
    adv_input = Input(shape=(1,))

    x = generator(seq_input)

    model = Model([seq_input, adv_input], x)
    model.compile(
        optimizer='nadam',
        loss=pg_loss(adv_input)
    )

    return model

def create_discriminator(base_model):
    """
    Discriminator model
    """
    seq_input = Input(shape=(SEQ_LEN,), dtype='int32')

    x = base_model(seq_input)

    x = LSTM(NUM_UNITS)(x)
    # x = Dropout(0.5)(x)

    # Prediction (1 = real, 0 = fake)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    model = Model(seq_input, x)
    model.compile(
        optimizer='nadam',
        loss='binary_crossentropy',
        metrics=['acc']
    )

    return model
