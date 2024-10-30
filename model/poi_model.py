import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
from json import encoder

from poi_utils import *
from poi_layers import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import geopy.distance
import math 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
import random
import pickle
import time
import scipy.sparse as sp

import itertools
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser
import numpy as np
import math
from torch.nn.parameter import Parameter
import argparse

import torch.optim as optim

class AE_GE(torch.nn.Module):
    def __init__(self, tfidf_dim=273, out_dim=500, emb_dim=64, device='cpu'):
        super(AE_GE, self).__init__()

        self.geo_SA = SelfAttention(out_dim)
        self.sem_SA = SelfAttention(out_dim)

        self.down1 = DownConv(out_dim, 256)
        self.down2 = DownConv(256, 64)

        self.feat_encoder = nn.Linear(tfidf_dim+2, emb_dim)
        self.time_encoder = nn.Linear(emb_dim, emb_dim)
        self.projection = nn.Linear(emb_dim*2, emb_dim)

        self.up1 = UpConv(64, 256)
        self.up2 = UpConv(256, out_dim)

        self.loss_fn = nn.MSELoss()

    def forward(self, batch_data):
        o_embed, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = batch_data

        geo_self_embed = torch.mean(geo_K_embed, dim=1)
        geo_emb = self.geo_SA(geo_self_embed, geo_K_embed, geo_K_dist)

        sem_self_embed = torch.mean(sem_K_embed, dim=1)
        sem_emb = self.sem_SA(sem_self_embed, sem_K_embed, sem_K_dist)

        _input_emb = geo_emb + sem_emb
        _input_emb = _input_emb.unsqueeze(1)

        x = self.down1(_input_emb)
        x = self.down2(x)

        feat_emb = self.feat_encoder(torch.cat([cate, coord], dim=1))
        feat_emb = feat_emb.unsqueeze(1)

        x = x + feat_emb
        x = self.up1(x)
        x = self.up2(x)
        x = x.squeeze(1)

        out_emb = x + geo_K_embed[:, 0, :]
        batch_loss = torch.sqrt(self.loss_fn(out_emb, o_embed))
        return batch_loss

    def forward_test(self, test_batch_data):
        geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = test_batch_data

        geo_K_dist = geo_K_dist.unsqueeze(0)
        geo_K_embed = geo_K_embed.unsqueeze(0)
        sem_K_dist = sem_K_dist.unsqueeze(0)
        sem_K_embed = sem_K_embed.unsqueeze(0)
        cate = cate.unsqueeze(0)
        coord = coord.unsqueeze(0)

        geo_self_embed = torch.mean(geo_K_embed, dim=1)
        geo_emb = self.geo_SA(geo_self_embed, geo_K_embed, geo_K_dist)

        sem_self_embed = torch.mean(sem_K_embed, dim=1)
        sem_emb = self.sem_SA(sem_self_embed, sem_K_embed, sem_K_dist)

        _input_emb = geo_emb + sem_emb
        _input_emb = _input_emb.unsqueeze(1)

        x = self.down1(_input_emb)
        x = self.down2(x)

        feat_emb = self.feat_encoder(torch.cat([cate, coord], dim=1))
        feat_emb = feat_emb.unsqueeze(1)

        x = x + feat_emb
        x = self.up1(x)
        x = self.up2(x)
        x = x.squeeze(1)

        out_emb = x.squeeze(0) + geo_K_embed.squeeze(0)[0]
        return out_emb

class DI_GE_IO(torch.nn.Module):
    def __init__(self, t_range=50, h_size=64, tfidf_dim=273, out_dim=500, device='cpu'):
        super(DI_GE_IO, self).__init__()

        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.out_dim = out_dim
        self.h_size = h_size
        self.device = device

        self.geo_SA = SelfAttention(out_dim*2)
        self.sem_SA = SelfAttention(out_dim*2)

        self.down1 = DownConv(out_dim*2, 256)
        self.down2 = DownConv(256, 64)

        self.feat_encoder = nn.Linear(tfidf_dim+2, h_size)
        # self.feat_encoder = nn.Linear(2, h_size)

        self.projection = nn.Linear(h_size*2, h_size)

        self.up1 = UpConv(64, 256)
        self.up2 = UpConv(256, out_dim*2)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def estimate_noise(self, noise_batch_data, t):
        noise_embs, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = noise_batch_data
        noise_embs = noise_embs.squeeze(1)

        geo_emb = self.geo_SA(noise_embs, geo_K_embed, geo_K_dist)
        sem_emb = self.sem_SA(noise_embs, sem_K_embed, sem_K_dist)

        _input_emb = geo_emb + sem_emb
        _input_emb = _input_emb.unsqueeze(1)

        x = self.down1(_input_emb)
        x = self.down2(x)

        feat_emb = self.feat_encoder(torch.cat([cate, coord], dim=1))
        # feat_emb = self.feat_encoder(coord)

        time_emb = self.pos_encoding(t, self.h_size)
        extra_emb = torch.cat([feat_emb, time_emb], dim=1)
        feat_emb = self.projection(extra_emb)
        feat_emb = feat_emb.unsqueeze(1)

        x = x + feat_emb
        x = self.up1(x)
        x = self.up2(x)
        # x = x.squeeze(1)
        return x

    def forward(self, batch_data):
        item_embed, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = batch_data
        target_embed = item_embed

        ts = torch.randint(0, self.t_range, [target_embed.shape[0]], device=self.device)
        noise_embs = []
        epsilons = torch.randn(target_embed.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_embs.append(
                (math.sqrt(a_hat) * target_embed[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_embs = torch.stack(noise_embs, dim=0)

        noise_batch_data = [noise_embs, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord]
        e_hat = self.estimate_noise(noise_batch_data, ts.unsqueeze(-1).type(torch.float))

        batch_loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.out_dim), epsilons.reshape(-1, self.out_dim)
        )
        return batch_loss

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def forward_test(self, test_batch_data):
        geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = test_batch_data

        geo_K_dist = geo_K_dist.unsqueeze(0)
        geo_K_embed = geo_K_embed.unsqueeze(0)
        sem_K_dist = sem_K_dist.unsqueeze(0)
        sem_K_embed = sem_K_embed.unsqueeze(0)
        cate = cate.unsqueeze(0)
        coord = coord.unsqueeze(0)

        noise_embs = torch.randn((1, 1, self.out_dim*2)).to(self.device)
        sample_steps = torch.arange(self.t_range-1, 0, -1)
        for t in sample_steps:
            noise_batch_data = [noise_embs, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord]
            t = t.to(self.device)
            noise_embs = self.denoise_sample(noise_batch_data, t)
        return noise_embs.squeeze()

    def denoise_sample(self, noise_batch_data, t):
        noise_embs, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord = noise_batch_data
        with torch.no_grad():
            if t > 1:
                z = torch.randn(noise_embs.shape).to(self.device)
            else:
                z = 0
            e_hat = self.estimate_noise(noise_batch_data, t.view(1, 1).repeat(noise_embs.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            noise_embs = pre_scale * (noise_embs - e_scale * e_hat) + post_sigma
            return noise_embs