# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 

import geopy.distance

class FlashBack(nn.Module):
    def __init__(self, parameters):
        super(FlashBack, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

        input_size = self.loc_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

        self.lambda_t = 0.1
        self.lambda_s = 50

    def f_t(self, delta_t):
        return ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2) * torch.exp(-(delta_t/86400*self.lambda_t))

    def f_s(self, delta_s):
        return torch.exp(-(delta_s*self.lambda_s)) 

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, coord):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        x = self.dropout(loc_emb)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)

        seq_len, hidden_size = out.size()
        out_w = torch.zeros(seq_len, hidden_size, device=out.device)

        for i in range(seq_len):
            weight = 0
            start = max(i-10, 0)
            for j in range(start, i+1):
                dist_t = tim[i] - tim[j]
                dist_s = torch.norm(coord[i] - coord[j])

                a_j = self.f_t(dist_t)
                b_j = self.f_s(dist_s)
                w_j = a_j*b_j + 1e-10 # small epsilon to avoid 0 division
                
                weight += w_j
                out_w[i] += w_j * out[j]
            out_w[i] /= weight
        
        out = F.selu(out_w)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score

# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score