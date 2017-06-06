from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import numpy as np
import argparse
import os
import json
from collections import deque

from models import *
from constants import *
from util import *
from generate import *

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
    target_data = [x[-1] for x in sequences]
    # TODO: Vocab size seems to be a limits outputs

    # Convert to one-hot vector
    target_data = np.array([to_categorical(seq, MAX_VOCAB) for seq in target_data])
    target_data = np.squeeze(target_data)

    print('Shape of train_data tensor:', train_data.shape)
    print('Shape of target_data tensor:', target_data.shape)
    return train_data, target_data

def main():
    """
    Main function executed to start training on dataset.
    """
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument('--skip-gen-pretrain', dest='pretrain_gen', action='store_false')
    parser.add_argument('--skip-dis-pretrain', dest='pretrain_dis', action='store_false')
    parser.set_defaults(pretrain_gen=True)
    parser.set_defaults(pretrain_dis=True)

    args = parser.parse_args()

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
            validation_split=0.1,
            epochs=100,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(patience=3),
                LambdaCallback(on_epoch_end=lambda a, b: generator.save_weights(G_MODEL_PATH))
            ]
        )
    else:
        pass
        # generator.load_weights(G_MODEL_PATH)

    if args.pretrain_dis:
        # TODO: This is sort of redundant, since the GAN part should already be training discriminator. Modulate this.
        # TODO: The discriminator may catestrophically interfere with shared model
        # TODO: Consider freezing weights or perform parallel training?
        # Generate fake samples
        num_real = train_data.shape[0]
        print('Generating {} fake samples...'.format(NUM_FAKE))
        fake_batches = [generate(generator, SEQ_LEN, batch=FAKE_GEN_BATCH_SIZE) for i in tqdm(range(NUM_FAKE // FAKE_GEN_BATCH_SIZE))]
        fake_samples = np.concatenate(fake_batches, axis=0)

        # Generate discriminator train and targets
        d_train = np.concatenate([np.array(fake_samples), train_data], axis=0)
        d_targets = np.concatenate([np.zeros((NUM_FAKE,)), np.ones((num_real,))])

        print('Pre-training discriminator...')
        discriminator.fit(
            d_train,
            d_targets,
            validation_split=0.1,
            epochs=100,
            batch_size=BATCH_SIZE,
            callbacks=[
                ModelCheckpoint(D_MODEL_PATH, save_best_only=True),
                EarlyStopping(patience=3)
            ]
        )
    else:
        discriminator.load_weights(D_MODEL_PATH)

    # GAN Training
    print('GAN training...')
    pg_generator = pg(generator)

    running_rewards = deque(maxlen=100)
    t = tqdm(range(10000))

    # Targets for discriminator (fake, real)
    d_targets = np.concatenate([np.zeros((ROLLOUT_BATCH,)), np.ones((ROLLOUT_BATCH,))])
    real_samp = train_data[0, :]
    print(to_text(real_samp, inv_idx))
    for e in t:
        ## Train generator
        # Perform rollouts
        outputs, inputs = generate(generator, SEQ_LEN, ROLLOUT_BATCH)

        # Compute advantage/rewards per rollout using D
        # rewards = discriminator.predict(outputs)
        rewards = [sum([1 if oo == real_samp[i] else 0 for i, oo in enumerate(o)]) for o in outputs]
        # rewards = np.array([list(o).count(1) for o in outputs]).astype(float)

        # Advantages has shape (batch, 1)
        # Normalize rewards for calculating advantage
        # Rescale rewards
        adv = rewards
        adv = [[0] * (SEQ_LEN - 1) + [r] for r in adv]
        adv = [discount_rewards(a) for a in adv]
        # adv -= np.mean(adv)
        # adv /= np.std(adv)

        g_inputs = np.reshape(inputs, [-1, SEQ_LEN])
        g_chosen = np.reshape(outputs, [-1, 1])
        g_chosen = to_categorical(g_chosen, MAX_VOCAB)
        g_adv = np.reshape(adv, [-1, 1])

        # Perform gradient updates
        pg_generator.train_on_batch([g_inputs, g_adv], g_chosen)

        ## Train discriminator
        # Create data samples. Fake, Real
        # Randomly pick real data from training set
        """
        # rand_ids = np.random.randint(train_data.shape[0], size=ROLLOUT_BATCH)
        rand_ids = [0 for i in range(ROLLOUT_BATCH)]
        real = train_data[rand_ids, :]
        d_train = np.concatenate([outputs, real], axis=0)

        # Train to classify fake and real data
        d_metric = discriminator.train_on_batch(d_train, d_targets)
        """
        d_metric = [0, 0]

        # Update progress bar
        running_rewards.append(np.mean(rewards))

        t.set_postfix(
            reward=np.mean(running_rewards),
            d_loss=d_metric[0],
            d_acc=d_metric[1]
        )

        if e % 50 == 0:
            # TODO: Should we save in the same path?
            generator.save_weights(RL_G_MODEL_PATH)
            discriminator.save_weights(RL_D_MODEL_PATH)
            write_outputs(inv_idx, outputs, str(e))

            # real_re = discriminator.predict(real)
            # print(rewards)
            # print(real_re)
            # print([[1 if oo == real_samp[i] else 0 for i, oo in enumerate(o)] for o in outputs])

if __name__ == '__main__':
    main()
