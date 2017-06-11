"""
Trains the model on corpus
"""
import torch
from torch.autograd import Variable

import random
import numpy as np
import argparse

from util import *
from model import *
from constants import *
from data_loader import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cudnn', dest='no_cudnn', action='store_true', required=False)
    parser.add_argument('--epoch', help='The epoch to start at', type=int, default=0)
    args = parser.parse_args()

    if args.no_cudnn:
        # Disables CudNN when insufficient memory
        print('Disabling CudNN')
        torch.backends.cudnn.enabled = False

    print('Loading data...')
    text = load_corpus()
    text = unicodeToAscii(text)

    print('Building models...')
    common = CommonModule(N_CHARS, G_UNITS)
    generator = Generator(N_CHARS, G_UNITS, common).cuda()
    discriminator = Discriminator(N_CHARS, D_UNITS, common).cuda()

    try:
        generator.load_state_dict(torch.load(G_MODEL_PATH))
        print('Loaded generator model')
    except Exception as e:
        print('Unable to load generator', e)

    try:
        discriminator.load_state_dict(torch.load(D_MODEL_PATH))
        print('Loaded discriminator model')
    except Exception as e:
        print('Unable to load discriminator', e)

    print('Training...')
    # Keep track of statistics
    run_g_loss = None
    run_g_reward = None
    run_d_loss = None
    run_d_acc = None

    t = tqdm(range(args.epoch, MAX_ITERATIONS))

    # Start training iterations
    for i in t:
        # Probability of training using GAN vs MLE
        gan_prob = min(i / GAN_ITERATIONS, 1)

        # Flip a coin and randomly decide to either train using GAN or MLE
        if np.random.random() < gan_prob:
            # Train generator (REINFORCE)
            fake_text, outputs = generator.sample(batch=BATCH_SIZE, length=SEQ_LEN, eval_mode=False)
            input_seqs = Variable(input_tensors(fake_text)).cuda()
            rewards, _ = discriminator(input_seqs, None)
            generator.reinforce(outputs, rewards)
            g_reward = torch.mean(rewards).data[0]
            run_g_reward = g_reward if run_g_reward is None else run_g_reward * 0.99 + g_reward * 0.01
        else:
            # Train generator (MLE)
            input_seqs, target_seqs = make_g_batch(text)
            g_loss = generator.train_step(input_seqs, target_seqs)
            run_g_loss = g_loss if run_g_loss is None else run_g_loss * 0.99 + g_loss * 0.01

        for g in range(D_STEPS):
            # Train discriminator
            fake_text, outputs = generator.sample(batch=BATCH_SIZE // 2, length=SEQ_LEN)
            input_seqs, target_seqs = make_d_batch(fake_text, text)
            d_loss, d_acc = discriminator.train_step(input_seqs, target_seqs)

            run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
            run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
            run_d_acc = d_acc if run_d_acc is None else run_d_acc * 0.99 + d_acc * 0.01

        # Track loss
        t.set_postfix(gan_prob=gan_prob, g_loss=run_g_loss, g_reward=run_g_reward, d_loss=run_d_loss, d_acc=run_d_acc)

        if i % 100 == 0 and i > 0:
            print('=== Generating Sample ===')
            print(generator.sample()[0][0])
            print('=== End Sample ===')
            # Save models
            torch.save(generator.state_dict(), G_MODEL_PATH)
            torch.save(discriminator.state_dict(), D_MODEL_PATH)

if __name__ == '__main__':
    main()
