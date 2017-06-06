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
        responsible_outputs = K.sum(y_true * y_pred, axis=2)
        return -K.sum(advantage * K.log(responsible_outputs))
    return f

def create_base_model(num_chars):
    """
    Base model shared by generator and discriminator.
    """
    seq_input = Input(shape=(SEQ_LEN,num_chars,), dtype='float32')

    # Conver to embedding space
    x = seq_input

    # LSTM with dropout
    x = LSTM(NUM_UNITS, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    return Model(seq_input, x)

def create_generator(base_model, num_chars):
    """
    Generator model
    """
    seq_input = Input(shape=(SEQ_LEN,num_chars,), dtype='float32')

    x = base_model(seq_input)

    x = LSTM(NUM_UNITS, return_sequences=True)(x)
    x = Dropout(0.5)(x)

    # Prediction (probability)
    x = Dense(num_chars)(x)
    x = Activation('softmax')(x)

    model = Model(seq_input, x)
    return model

def mle(generator, num_chars):
    """
    Wraps the generator in an MLE model
    """
    seq_input = Input(shape=(SEQ_LEN,num_chars,), dtype='float32')

    x = generator(seq_input)

    model = Model(seq_input, x)
    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy'
    )

    return model

def pg(generator, num_chars):
    """
    Wraps the generator in a policy gradient model
    """
    seq_input = Input(shape=(SEQ_LEN,num_chars,), dtype='float32')
    # Advantages for loss function
    adv_input = Input(shape=(1,))

    x = generator(seq_input)

    model = Model([seq_input, adv_input], x)
    model.compile(
        optimizer='nadam',
        loss=pg_loss(adv_input)
    )

    return model

def create_discriminator(base_model, num_chars):
    """
    Discriminator model
    """
    seq_input = Input(shape=(SEQ_LEN,num_chars,), dtype='float32')

    x = base_model(seq_input)

    x = LSTM(NUM_UNITS)(x)
    x = Dropout(0.5)(x)

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
