import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np

from util import *
from p_model import *
from constants import *
from data_loader import *

def main():
    print('Loading data...')
    text = load_corpus()
    text = unicodeToAscii(text)

    print('Building models...')
    common = CommonModule(n_chars, g_units)
    generator = Generator(n_chars, g_units, common).cuda()
    discriminator = Discriminator(n_chars, d_units, common).cuda()

    print('Training...')
    run_g_loss = None
    run_d_loss = None
    run_d_acc = None

    max_iterations = 100000
    gan_iteration = 20000
    t = tqdm(range(max_iterations))

    for i in t:
        if np.random.random() < i / gan_iteration:
            # Train generator (REINFORCE)
            fake_text, outputs = generator.sample(batch=BATCH_SIZE, length=SEQ_LEN, eval_mode=False)
            input_seqs = Variable(input_tensors(fake_text)).cuda()
            rewards, _ = discriminator(input_seqs, None)
            generator.reinforce(outputs, rewards)
        else:
            # Train generator (MLE)
            input_seqs, target_seqs = make_g_batch(text)
            g_loss = generator.train_step(input_seqs, target_seqs)
            run_g_loss = g_loss if run_g_loss is None else run_g_loss * 0.99 + g_loss * 0.01

        # Train discriminator
        fake_text, outputs = generator.sample(batch=BATCH_SIZE // 2, length=SEQ_LEN)
        input_seqs, target_seqs = make_d_batch(fake_text, text)
        d_loss, d_acc = discriminator.train_step(input_seqs, target_seqs)

        run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
        run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
        run_d_acc = d_acc if run_d_acc is None else run_d_acc * 0.99 + d_acc * 0.01

        # Track loss
        t.set_postfix(g_loss=run_g_loss, d_loss=run_d_loss, d_acc=run_d_acc)

        if i % 1000 == 0:
            print('=== Generating Sample ===')
            print(generator.sample()[0][0])
            print('=== End Sample ===')
            torch.save(generator.state_dict(), 'out/generator.torch')
            torch.save(discriminator.state_dict(), 'out/discriminator.torch')

if __name__ == '__main__':
    main()
