from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback

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
    parser.add_argument('--skip-gen-pretrain', dest='pretrain_gen', action='store_false', required=False)
    parser.add_argument('--skip-dis-pretrain', dest='pretrain_dis', action='store_false', required=False)
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
    print('Min sequence length:', min([len(seq) for seq in sequences]))
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

    # Inverse word index
    inv_idx = {v: k for k, v in tokenizer.word_index.items()}

    # Create models
    # TODO: Do model sharing later
    # base_model = create_base_model(embedding_matrix)
    generator = create_generator(create_base_model(embedding_matrix))
    discriminator = create_discriminator(create_base_model(embedding_matrix))

    os.makedirs('out', exist_ok=True)
    os.makedirs('out/outputs', exist_ok=True)

    # Write word index to file for generation
    with open('out/word_index.json', 'w') as f:
        json.dump(tokenizer.word_index, f)

    # MLE Pre-training
    if args.pretrain_gen:
        print('Pre-training generator...')
        # Wrap the generator with MLE loss
        mle_generator = mle(generator)

        mle_generator.fit(
            train_data,
            target_data,
            epochs=1000,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(monitor='loss', patience=3),
                LambdaCallback(on_epoch_end=lambda a, b: generator.save_weights(G_MODEL_PATH))
            ]
        )
    else:
        generator.load_weights(G_MODEL_PATH)

    if args.pretrain_dis:
        # TODO: This is sort of redundant, since the GAN part should already be training discriminator. Modulate this.
        # TODO: The discriminator may catestrophically interfere with shared model
        # TODO: Consider freezing weights or perform parallel training?
        # Generate fake samples
        num_real = train_data.shape[0]
        print('Generating {} fake samples...'.format(NUM_FAKE))
        fake_batches = [generate_seq(generator, SEQ_LEN, batch=FAKE_GEN_BATCH_SIZE) for i in tqdm(range(NUM_FAKE // FAKE_GEN_BATCH_SIZE))]
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
    else:
        discriminator.load_weights(D_MODEL_PATH)

    # GAN Training
    print('GAN training...')
    pg_generator = pg(generator)

    running_rewards = deque(maxlen=100)
    t = tqdm(range(10000))

    # Targets for discriminator
    d_targets = np.concatenate([np.zeros((ROLLOUT_BATCH,)), np.ones((ROLLOUT_BATCH,))])

    for e in t:
        ## Train generator
        for g in range(0,G):
            # Perform rollouts
            outputs = generate_seq(generator, SEQ_LEN, ROLLOUT_BATCH)

            # Compute advantages/rewards per rollout using D
            rewards = discriminator.predict(outputs)

            # Advantages has shape (batch, 1)
            # Normalize advantages
            avg_rewards = np.mean(rewards)
            # TODO: Should we discount the rewards?
            # advantages = rewards * 2 - 1
            # Normalize rewards
            std_rewards = np.std(rewards)
            advantages = (rewards - avg_rewards) / (std_rewards if std_rewards != 0 else 1)

            # Recreate inputs by shifting output to the right and left pad by zero
            inputs = np.pad(outputs[:, :-1], ((0, 0), (1, 0)), 'constant')

            # Convert outputs into one-hot version to use as target labels
            chosen = np.array([to_categorical(o, MAX_VOCAB) for o in outputs])
            print (outputs)

            # Perform gradient updates
            pg_generator.train_on_batch([inputs, advantages], chosen)

        ## Train discriminator
        # for k in range(10):
        # Create data samples. Fake, Real
        # Randomly pick real data from training set
        rand_ids = np.random.randint(train_data.shape[0], size=ROLLOUT_BATCH)
        d_train = np.concatenate([outputs, train_data[rand_ids, :]], axis=0)

        # Train to classify fake and real data
        d_metric = discriminator.train_on_batch(d_train, d_targets)

        # Update progress bar
        running_rewards.append(avg_rewards)
        t.set_postfix(
            reward=np.mean(running_rewards),
            d_loss=d_metric[0],
            d_acc=d_metric[1]
        )

        if e % 32 == 0:
            # TODO: Should we save in the same path?
            generator.save_weights(RL_G_MODEL_PATH)
            discriminator.save_weights(RL_D_MODEL_PATH)
            write_outputs(inv_idx, outputs, str(e))

def sample(distr, temp=1):
    if temp != 1:
        distr = np.log(distr) / temp
        distr = np.exp(distr) / np.sum(np.exp(distr), axis=1)[:, None]
    return [np.random.choice(MAX_VOCAB, 1, p=distr[b])[0] for b in range(distr.shape[0])]

def generate_seq(generator, length=GEN_LEN, batch=1):
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

def generate(args):
    """
    Main function executed to start training on dataset.
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

    results = generate_seq(generator, args.gen_len, args.gen_count)

    write_outputs(inv_idx, results)
    print('Output written to file.')



def test_bleu(args):
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
    #model.load_weights(G_MODEL_PATH)

    # Get word mapping for outputs -> text
    word_index = load_json_dict('out/word_index.json')
    inv_idx = {v: k for k, v in word_index.items()}

    def toText (list_outputs):
        return [inv_idx[word] for word in list_outputs if word != 0]

    # Get body of fakes (hypotheses) and reals.. the actual test corpus
    # SeqGan paper checks against (whole test set) - test against 20%
    fifth = int(train_data.shape[0]/5)
    reals = train_data[:fifth,:]
    fakes = generate_seq(model, length=GEN_LEN, batch=BLEU_SAMPLES)

    # Convert to text for BLEU scoring
    fakes_text = [' '.join(toText(fake)) for fake in fakes]
    reals_text = [' '.join(toText(real)) for real in reals]

    # 4-gram BLEU score
    scores = []
    for fake_sample in fakes_text:
        score = nltk.translate.bleu_score.sentence_bleu(reals_text, fake_sample, weights=[.25,.25,.25,.25])
        scores.append(score)

    avg_score = sum(scores) / float(len(scores))
    print ("Score is:" + str(avg_score))
    

if __name__ == '__main__':
    main()