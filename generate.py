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

def sample(distr, temp=1.0):
    distr = np.log(distr) / temp
    distr = np.exp(distr) / np.sum(np.exp(distr))
    return [np.random.choice(MAX_VOCAB, 1, p=distr[b])[0] for b in range(distr.shape[0])]

def generate(generator, length=GEN_LEN, batch=1):
    # Generative sampling, store results in results, seed with current results
    results = np.zeros((batch, SEQ_LEN))

    for i in range(length):
        # Take the last SEQ_LEN results and feed it in.
        feed = results[:, -SEQ_LEN:]

        distr = generator.predict(feed)
        distr = np.array(distr)
        # Pick the last result for each batch
        distr = distr[:, -1]
        choices = np.array(sample(distr, temp=TEMP))
        results = np.hstack([results, choices])

    # Slice out the last words (Ignore the buffer)
    results = np.array(results)
    return results[:, -length:]

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

    results = generate(generator, parser.gen_len, parser.gen_count)[0]

    # Ignore null words
    textual = [inv_idx[word] for word in results if word != 0]

    # Print resulting sentences
    print('\nResulting Sentence, temp = ' + str(TEMP) + ':')
    joined_text = ' '.join(textual)
    print(joined_text)

    # Write result to file
    with open('out/output.txt', 'w') as f:
        f.write(joined_text)

if __name__ == '__main__':
    main()
