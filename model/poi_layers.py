import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()

        self.h_size = h_size
        self.key_encoder = nn.Linear(h_size, h_size)
        self.query_encoder = nn.Linear(h_size, h_size)
        self.value_encoder = nn.Linear(h_size, h_size)

    def forward(self, target, neighs, kernels):
        batch, K, e_size = neighs.size()

        query = self.query_encoder(target)
        query = query.unsqueeze(1)

        keys = self.key_encoder(neighs)
        values = self.value_encoder(neighs)

        attention_value = torch.bmm(query, keys.transpose(1, 2))
        attention_value /= np.sqrt(self.h_size)
        attention_value = attention_value.squeeze(1)

        attention_value = attention_value * kernels
        attention_value = F.softmax(attention_value, dim=-1)

        attention_value = attention_value.unsqueeze(1)
        agg_vector = torch.bmm(attention_value, values)
        agg_vector = agg_vector.squeeze(1)

        return agg_vector 

class DownConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        stride = input_size // output_size
        kernel_size = input_size - (output_size - 1) * stride
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        stride = output_size // input_size
        kernel_size = output_size - (input_size - 1) * stride
        self.conv = nn.ConvTranspose1d(1, 1, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)
