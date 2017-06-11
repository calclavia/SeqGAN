import torch
import torch.nn as nn
from torch.autograd import Variable

import unicodedata
import random

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

def input_tensors(lines):
    """
    Converts a string to a one-hot tensor
    """
    tensor = torch.zeros(len(lines[0]), len(lines), n_chars)
    for i in range(tensor.size()[1]):
        for t in range(tensor.size()[0]):
            tensor[t][i][all_chars.find(lines[i][t])] = 1
    return tensor

def target_tensors(lines):
    """
    Converts a string to a character ID tensor
    """
    letter_indexes = [[all_chars.find(c) for c in line] for line in lines]
    # Change tensor back into [seq, batch]
    return torch.LongTensor(letter_indexes).permute(1, 0)

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
