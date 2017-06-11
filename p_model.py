import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random
import numpy as np

from constants import *
from data_loader import *

class CommonModule(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_chars, hidden_size, num_layers=1, dropout=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        out, hidden = self.lstm(inputs, hidden)
        out = self.dropout(out)
        return out, hidden

class Generator(nn.Module):
    def __init__(self, num_chars, hidden_size, common):
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.common = common
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, dropout=0.5)
        self.prediction = nn.Linear(hidden_size, num_chars)
        self.log_softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        seq_len = inputs.size()[0]

        out, hidden1 = self.common(inputs, hidden[0] if hidden else None)
        out, hidden2 = self.lstm(out, hidden[1] if hidden else None)

        out = out.view(-1, self.hidden_size)
        out = self.dropout(out)
        out = self.prediction(out)
        out = self.log_softmax(out)
        out = out.view(seq_len, -1, self.num_chars)
        return out, (hidden1, hidden2)

    def train_step(self, input_seqs, target_seqs):
        """
        Perform a single training step using MLE
        """
        self.train()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters())

        hidden = None

        # Zero out the gradient
        optimizer.zero_grad()

        output, hidden = self.forward(input_seqs, hidden)
        loss = criterion(output.view(-1, n_chars), target_seqs.view(-1))

        loss.backward()
        optimizer.step()

        return loss.data[0] / input_seqs.size()[0]

    def sample(self, batch=1, length=512, eval_mode=True):
        """
        Samples from the model and generates text.
        Returns the output strings and log probabilities of all outputs
        """
        # TODO: Don't train with Dropout when reinforcing?
        self.eval()

        if not eval_mode:
            self.zero_grad()

        hidden = None

        chosen_prob = []
        output_str = ['' for i in range(batch)]
        starting_str = [all_chars[random.randint(0, n_chars - 1)] for i in range(batch)]
        current_char = Variable(input_tensors(starting_str), volatile=eval_mode).cuda()

        for i in range(length):
            output, hidden = self.forward(current_char, hidden)
            dists = torch.exp(output).data[0].cpu().numpy()

            # Sample from distribution
            choices = [np.random.choice(n_chars, 1, p=dist)[0] for dist in dists]
            letters = [all_chars[choice] for choice in choices]
            chosen_prob.append(torch.cat([output[0, b, choice] for b, choice in enumerate(choices)], dim=0))

            current_char = Variable(input_tensors(letters)).cuda()

            for i, letter in enumerate(letters):
                output_str[i] += letter
        return output_str, torch.stack(chosen_prob, dim=0)

    def reinforce(self, outputs, rewards):
        optimizer = optim.Adam(self.parameters())
        advantages = rewards
        loss = -torch.sum(torch.mm(outputs, advantages))
        loss.backward()
        optimizer.step()
        return loss

class Discriminator(nn.Module):
    def __init__(self, num_chars, hidden_size, common):
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.common = common
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, dropout=0.5)
        self.prediction = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        out, hidden1 = self.common(inputs, hidden[0] if hidden else None)
        out, hidden2 = self.lstm(out, hidden[1] if hidden else None)

        # We only care about the last output
        out = out[-1, :, :]
        out = self.dropout(out)
        out = self.prediction(out)
        out = self.sigmoid(out)
        return out, (hidden1, hidden2)

    def train_step(self, input_seqs, target_seqs):
        """
        Perform a single training step
        """
        self.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters())

        hidden = None

        # Zero out the gradient
        optimizer.zero_grad()

        output, hidden = self.forward(input_seqs, hidden)
        loss = criterion(output, target_seqs)

        # Compute accuracy
        b_outputs = torch.round(output)
        accuracy = torch.mean(target_seqs * b_outputs + (1 - target_seqs) * (1 - b_outputs))

        loss.backward()
        optimizer.step()

        return output, loss.data[0] / input_seqs.size()[0], accuracy.data[0]
