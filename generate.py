from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import random
import numpy as np
import os

from models import *
from constants import *
from util import *
import argparse

def sample(distr, temp=1):
    if temp != 1:
        distr = np.log(distr) / temp
        distr = np.exp(distr) / np.sum(np.exp(distr), axis=1)[:, None]
    return [np.random.choice(MAX_VOCAB, 1, p=distr[b])[0] for b in range(distr.shape[0])]

def generate(generator, length=GEN_LEN, batch=1):
    # Generative sampling
    outputs = np.zeros((batch, SEQ_LEN))

    for i in range(length):
        # Take the last SEQ_LEN outputs and feed it in.
        feed = outputs[:, -SEQ_LEN:]

        distr = generator.predict(feed)
        distr = np.array(distr)
        # Pick the last result for each batch
        distr = distr[:, -1]
        choices = np.reshape(sample(distr, temp=TEMP), [-1, 1])
        outputs = np.hstack([outputs, choices])

    # Slice out the last words (Ignore the buffer)
    outputs = np.array(outputs)
    return outputs[:, -length:]

def write_outputs(inv_idx, results, prefix=''):
    for i, result in enumerate(results):
        # Ignore null words
        textual = [inv_idx[word] for word in result if word != 0]
        joined_text = ' '.join(textual)

        # Write result to file
        with open('out/outputs/output_{}_{}.txt'.format(prefix, i), 'w') as f:
            f.write(joined_text)

def main():
    """
    Main function executed to start training on dataset.
    """
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument('--len', dest='gen_len', type=int, default=GEN_LEN)
    parser.add_argument('--count', dest='gen_count', type=int, default=1)

    args = parser.parse_args()

    # Load word index
    word_index = load_json_dict('out/word_index.json')

    # Load embedding matrix
    embedding_matrix = load_embedding(word_index)

    # Create models
    base_model = create_base_model(embedding_matrix)
    generator = create_generator(base_model)

    os.makedirs('out', exist_ok=True)

    # Load in model weights
    generator.load_weights(G_MODEL_PATH)

    # Load word index
    inv_idx = {v: k for k, v in word_index.items()}

    results = generate(generator, args.gen_len, args.gen_count)

    write_outputs(inv_idx, results)
    print('Output written to file.')

if __name__ == '__main__':
    main()
