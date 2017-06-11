import torch
import torch.nn as nn
import torch.optim as optim

from constants import *

class Generator(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_chars, hidden_size, num_layers=3, dropout=0.5)
        self.prediction = nn.Linear(hidden_size, num_chars)
        self.log_softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        seq_len = inputs.size()[0]

        out, hidden = self.lstm(inputs, hidden)
        out = out.view(-1, self.hidden_size)
        out = self.dropout(out)
        out = self.prediction(out)
        out = self.log_softmax(out)
        out = out.view(seq_len, -1, self.num_chars)
        return out, hidden

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

class Discriminator(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super().__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_chars, hidden_size, num_layers=2, dropout=0.5)
        self.prediction = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, hidden):
        seq_len = inputs.size()[0]

        out, hidden = self.lstm(inputs, hidden)

        # We only care about the last output
        out = out[-1, :, :]
        out = self.dropout(out)
        out = self.prediction(out)
        out = self.sigmoid(out)
        return out, hidden

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

        loss.backward()
        optimizer.step()

        return loss.data[0] / input_seqs.size()[0]
