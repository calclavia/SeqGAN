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

def random_subseq(seq, seq_len=SEQ_LEN):
    index = random.randint(0, len(seq) - seq_len - 1)
    return seq[index:index + seq_len]

def input_tensor(line):
    """
    Converts a string to a one-hot tensor
    """
    tensor = torch.zeros(len(line), 1, n_chars)
    for i in range(len(line)):
        letter = line[i]
        tensor[i][0][all_chars.find(letter)] = 1
    return tensor

def input_tensors(lines):
    """
    Converts a string to a one-hot tensor
    """
    tensor = torch.zeros(len(lines[0]), len(lines), n_chars)
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            tensor[j][i][all_chars.find(lines[i][j])] = 1
    return tensor

def target_tensors(lines):
    """
    Converts a string to a character ID tensor
    """
    letter_indexes = [[all_chars.find(c) for c in line] for line in lines]
    return torch.LongTensor(letter_indexes)

def make_g_batch(text):
    """
    Samples from a text and generates a batch of inputs
    """
    # Create data batch
    input_seqs = []
    target_seqs = []

    for b in range(batch_size):
        seq = random_subseq(text, SEQ_LEN + 1)
        input_seqs.append(seq[:-1])
        target_seqs.append(seq[1:])

    input_seqs = Variable(input_tensors(input_seqs)).cuda()
    target_seqs = Variable(target_tensors(target_seqs)).cuda()
    return input_seqs, target_seqs

def make_d_batch(fake_text, real_text):
    """
    Samples from a text and generates a batch of inputs
    """
    num = batch_size // 2

    # Create fake batch
    input_seqs = fake_text

    # Create real batch
    input_seqs += [random_subseq(real_text) for b in range(num)]

    input_seqs = Variable(input_tensors(input_seqs)).cuda()
    target_seqs = Variable(torch.Tensor([0] * num + [1] * num)).cuda()
    return input_seqs, target_seqs

def sample(model, batch=1, length=512):
    """
    Samples from the model and generates text.
    """
    model.eval()
    hidden = None
    output_str = [all_chars[random.randint(0, n_chars - 1)] for i in range(batch)]
    current_char = Variable(input_tensors(output_str), volatile=True).cuda()

    for i in range(length):
        output, hidden = model(current_char, hidden)
        dists = torch.exp(output).data[0].cpu().numpy()
        choices = [np.random.choice(n_chars, 1, p=dist)[0] for dist in dists]
        letters = [all_chars[choice] for choice in choices]
        current_char = Variable(input_tensors(letters)).cuda()

        for i, letter in enumerate(letters):
            output_str[i] += letter

    return output_str

def main():
    print('Loading data...')
    text = load_corpus()
    text = unicodeToAscii(text)

    print('Building models...')
    common = CommonModule(n_chars, g_units).cuda()
    generator = Generator(n_chars, g_units, common).cuda()
    discriminator = Discriminator(n_chars, g_units, common).cuda()

    print('Training...')
    run_g_loss = None
    run_d_loss = None
    run_d_acc = None

    t = tqdm(range(100000))

    for i in t:
        # Train generator
        input_seqs, target_seqs = make_g_batch(text)
        g_loss = generator.train_step(input_seqs, target_seqs)
        run_g_loss = g_loss if run_g_loss is None else run_g_loss * 0.99 + g_loss * 0.01

        if i % 4 == 0:
            # Train discriminator
            fake_text = sample(generator, batch=16, length=SEQ_LEN)
            input_seqs, target_seqs = make_d_batch(fake_text, text)
            d_loss, d_acc = discriminator.train_step(input_seqs, target_seqs)

            run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
            run_d_loss = d_loss if run_d_loss is None else run_d_loss * 0.99 + d_loss * 0.01
            run_d_acc = d_acc if run_d_acc is None else run_d_acc * 0.99 + d_acc * 0.01

        # Track loss
        t.set_postfix(g_loss=run_g_loss, d_loss=run_d_loss, d_acc=run_d_acc)

        if i % 1000 == 0:
            print('=== Generating Sample ===')
            print(sample(generator)[0])
            print('=== End Sample ===')
            torch.save(generator.state_dict(), 'out/generator.torch')
            torch.save(discriminator.state_dict(), 'out/discriminator.torch')

if __name__ == '__main__':
    main()
