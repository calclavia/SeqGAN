import torch
import torch.nn as nn
from torch.autograd import Variable

import unicodedata
import random
import numpy as np

from util import *
from p_model import *
from constants import *

def unicodeToAscii(s):
    """
    Turns unicode to ASCII
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_chars
    )

def train_seq(text, seq_len=128):
    """
    Extracts a random training pair from the corpus
    """
    index = random.randint(0, len(text) - seq_len - 1)
    input_seq = Variable(input_tensor(text[index:index + seq_len])).cuda()
    target_seq = Variable(target_tensor(text[index + 1:index + seq_len + 1])).cuda()
    return input_seq, target_seq

def input_tensor(line):
    """
    Converts a string to a one-hot tensor
    """
    tensor = torch.zeros(len(line), 1, n_chars)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_chars.find(letter)] = 1
    return tensor

def target_tensor(line):
    """
    Converts a string to a character ID tensor
    """
    letter_indexes = [all_chars.find(c) for c in line]
    return torch.LongTensor(letter_indexes)

def make_g_batch(text):
    """
    Samples from a text and generates a batch of inputs
    """
    input_seqs = []
    target_seqs = []
    # Create data batch
    for b in range(batch_size):
        input_seq, target_seq = train_seq(text)
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    input_seqs = torch.cat(input_seqs, dim=1)
    target_seqs = torch.stack(target_seqs, dim=1)
    return input_seqs, target_seqs

def make_d_batch(fake_text, real_text):
    """
    Samples from a text and generates a batch of inputs
    """
    input_seqs = []
    num = batch_size // 2

    # Create fake batch
    for b in range(num):
        input_seq, target_seq = train_seq(fake_text)
        input_seqs.append(input_seq)

    # Create real batch
    for b in range(num):
        input_seq, target_seq = train_seq(real_text)
        input_seqs.append(input_seq)

    input_seqs = torch.cat(input_seqs, dim=1)
    target_seqs = Variable(torch.Tensor([0] * num + [1] * num)).cuda()
    return input_seqs, target_seqs

def sample(model, length=256):
    """
    Samples from the model and generates text.
    """
    model.eval()
    hidden = None
    output_str = all_chars[random.randint(0, n_chars)]
    current_char = Variable(input_tensor(output_str), volatile=True).cuda()

    for i in range(length):
        output, hidden = model(current_char, hidden)
        dist = np.exp(output.data[0][0].cpu().numpy())
        choice = np.random.choice(n_chars, 1, p=dist)[0]
        letter = all_chars[choice]
        current_char = Variable(input_tensor(letter)).cuda()
        output_str += letter

    return output_str

def main():
    print('Loading data...')
    text = load_corpus()
    text = unicodeToAscii(text)

    print('Building models...')
    generator = Generator(n_chars, g_units).cuda()
    discriminator = Discriminator(n_chars, d_units).cuda()

    print('Training...')
    run_g_loss = None
    run_d_loss = None

    t = tqdm(range(10000))

    for i in t:
        # Train generator
        input_seqs, target_seqs = make_g_batch(text)
        g_loss = generator.train_step(input_seqs, target_seqs)

        # Train discriminator
        fake_text = sample(generator)
        input_seqs, target_seqs = make_d_batch(fake_text, text)
        d_loss = discriminator.train_step(input_seqs, target_seqs)

        # Track loss
        run_g_loss = g_loss if run_g_loss is None else run_g_loss * 0.999 + g_loss * 0.001
        run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.999 + d_loss * 0.001
        t.set_postfix(g_loss=run_g_loss, d_loss=run_d_loss)

        if i % 1000 == 0:
            print('=== Generating Sample ===')
            print(fake_text)
            print('=== End Sample ===')
            torch.save(generator.state_dict(), 'out/generator.torch')

if __name__ == '__main__':
    main()
