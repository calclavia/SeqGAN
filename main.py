from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import nltk
import random
import argparse
import numpy as np
import nltk
import argparse
import os
import json

from models import *
from constants import *
from util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        help='What to do?',
                        choices=['train', 'generate', 'bleu',],
                        required=True)
    parser.add_argument('--temperature',
                        help='Temperature for softmax sampling. [0,1]',
                        type=int, default=.8, required=False)
    parser.add_argument('--len', dest='gen_len', type=int, required=False, default=GEN_LEN)
    parser.add_argument('--count', dest='gen_count', type=int, required=False, default=1)
    parser.add_argument('--skip-gen-pretrain', dest='pretrain_gen',
                        action='store_false', required=False)
    parser.set_defaults(pretrain_gen=True)
    args = parser.parse_args()

    if (args.task == 'bleu'):
        test_bleu(args)
    elif (args.task == 'generate'):
        generate(args)
    else:
        train(args)


def load_data(tokenizer):
    """
    Loads training data from file and processes it.
    """
    print('Loading data...')
    # Prepare the tokenizer
    text = load_corpus()

    # Split based on sentences
    sentences = sent_tokenize(text)

    tokenizer.fit_on_texts(sentences)

    # A list of sequences. Each sequence has a different length.
    sentences = tokenizer.texts_to_sequences(sentences)

    sequences = []

    # Slice long sentences into subsequences of SEQ_LEN
    for sent in sentences:
        for i in range(0, len(sent) - SEQ_LEN + 1, TRAIN_WINDOW):
            sequences.append(sent[i: i + SEQ_LEN])

    print('Number of sequences:', len(sequences))
    print('Average sequence length:', np.mean([len(seq) for seq in sequences]))
    print('Max sequence length:', max([len(seq) for seq in sequences]))
    print('Found {} unique tokens.'.format(len(tokenizer.word_index)))

    # Create training data and target data.
    # Truncates and pads sequences so that they're the same length.
    train_data = pad_sequences([x[:-1] for x in sequences], maxlen=SEQ_LEN)
    # Target data is training data shfited by one word
    target_data = pad_sequences([x[1:] for x in sequences], maxlen=SEQ_LEN)
    # TODO: Vocab size seems to be a limits outputs

    # Convert to one-hot vector
    target_data = np.array([to_categorical(seq, MAX_VOCAB) for seq in target_data])

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of target_data tensor:', target_data.shape)
    return train_data, target_data

def train(args):
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
    if args.pretrain_gen:
        print('Pre-training generator...')
        generator.fit(
            train_data,
            target_data,
            validation_split=0.1,
            epochs=1000,
            batch_size=BATCH_SIZE,
            callbacks=[
                ModelCheckpoint(G_MODEL_PATH, save_best_only=True),
                EarlyStopping(patience=5)
            ]
        )
    else:
        generator.load_weights(G_MODEL_PATH)

    # TODO: The discriminator may catestrophically interfere with shared model
    # TODO: Consider freezing weights or perform parallel training.

    # Generate fake samples
    num_real = train_data.shape[0]
    print('Generating {} fake samples...'.format(NUM_FAKE))
    fake_batches = [gen_from_model(generator, SEQ_LEN, batch=FAKE_GEN_BATCH_SIZE) for i in tqdm(range(NUM_FAKE // FAKE_GEN_BATCH_SIZE))]
    fake_samples = np.concatenate(fake_batches, axis=0)

    # Generate discriminator train and targets
    d_train = np.concatenate([np.array(fake_samples), train_data], axis=0)
    d_targets = np.concatenate([np.zeros((NUM_FAKE,)), np.ones((num_real,))])

    print('Pre-training discriminator...')
    discriminator.fit(
        d_train,
        d_targets,
        validation_split=0.1,
        epochs=1000,
        batch_size=BATCH_SIZE,
        callbacks=[
            ModelCheckpoint(D_MODEL_PATH, save_best_only=True),
            EarlyStopping(patience=5)
        ]
    )


def sample(distr, temp=1.0):
    distr = np.log(distr) / temp
    distr = np.exp(distr) / np.sum(np.exp(distr), axis=1)[:, None]
    return [np.random.choice(MAX_VOCAB, 1, p=distr[b])[0] for b in range(distr.shape[0])]

def gen_from_model(generator, length=GEN_LEN, batch=1):
    # Generative sampling, store results in results, seed with current results
    results = np.zeros((batch, SEQ_LEN))

    for i in range(length):
        # Take the last SEQ_LEN results and feed it in.
        feed = results[:, -SEQ_LEN:]

        distr = generator.predict(feed)
        distr = np.array(distr)
        # Pick the last result for each batch
        distr = distr[:, -1]
        choices = np.reshape(sample(distr, temp=TEMP), [-1, 1])
        results = np.hstack([results, choices])

    # Slice out the last words (Ignore the buffer)
    results = np.array(results)
    return results[:, -length:]

def generate(args):
    """
    Main function executed to generate samples from a model.
    """
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

    results = gen_from_model(generator, args.gen_len, args.gen_count)

    # Ignore null words
    for i, result in enumerate(results):
        textual = [inv_idx[word] for word in result if word != 0]

        # Print resulting sentences
        print('\nResulting Sentence, temp = ' + str(TEMP) + ':')
        joined_text = ' '.join(textual)
        print(joined_text)

        # Write result to file
        with open('out/output_{}.txt'.format(i), 'w') as f:
            f.write(joined_text)


def test_bleu(args): #model,data_string,n=2):
    """
    Evaluates model's max activation outputs for a real sequence, by BLEU score.
    model is generator, pass in trained
    data_string is list of embedded tokens from real data... [23,244,12,70] etc
    """
    # Load data
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    train_data, target_data = load_data(tokenizer)
    embedding_matrix = load_embedding(tokenizer.word_index)

    # Load model
    base_model = create_base_model(embedding_matrix)
    model = create_generator(base_model)
    model.load_weights(G_MODEL_PATH)

    # Get word mapping for outputs -> text
    word_index = load_json_dict('out/word_index.json')
    inv_idx = {v: k for k, v in word_index.items()}

    def toText (list_outputs):
        return [inv_idx[word] for word in list_outputs if word != 0]

    data_string = train_data[np.random.randint(len(train_data))]
    reals = np.array([data_string])

    result = model.predict(reals)
    distrs = np.array(result)
    choices = [sample(distr, temp=args.temperature) for distr in distrs]
    fake_text = ' '.join(toText(choices[0][1:-1]))
    real_text = ' '.join(toText(data_string[1:-1]))

    score = nltk.translate.bleu_score.sentence_bleu(real_text, fake_text)
    print ("Score is:" + str(score))
    

if __name__ == '__main__':
    main()
