from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', dropout_prob=0.):
        super(TextGenerationModel, self).__init__()
        embedding_dim = 200
        self.embed = nn.Embedding(vocabulary_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers, dropout=dropout_prob)
        self.projection = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.to(device)

    def forward(self, x, h_and_c=None):
        embedding = self.embed(x)
        hidden_states, (h, c) = self.lstm(embedding, h_and_c)
        return self.projection(hidden_states), (h, c)
