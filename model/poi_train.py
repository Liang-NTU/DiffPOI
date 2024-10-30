import sys
import random
import pickle
import time
import scipy.sparse as sp
import torch
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
from poi_data import *
from poi_utils import *
from poi_model import *

def test(model, data, device, poi_args):
    warm_data = data.warm_data
    cold_data = data.cold_data

    input_emb = torch.zeros(len(warm_data)+len(cold_data), 500)
    output_emb = torch.zeros(len(warm_data)+len(cold_data), 500)
    for poi in warm_data:
        i_emb = warm_data[poi]["i_embed"]
        o_emb = warm_data[poi]["o_embed"]
        input_emb[poi] = i_emb
        output_emb[poi] = o_emb

    keys = data.cold_data.keys()
    for poi in keys:
        sample_data = data.__get_test_item__(poi)

        geo_K_dist = sample_data["geo_K_dist"].to(device)
        geo_K_embed = sample_data["geo_K_embed"].to(device)
        sem_K_dist = sample_data["sem_K_dist"].to(device)
        sem_K_embed = sample_data["sem_K_embed"].to(device)
        cate = sample_data["cate"].to(device)
        coord = sample_data["coord"].to(device)

        batch_data = [geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord]
        pre_emb = model.forward_test(batch_data)

        input_emb[poi] = sample_data["geo_K_embed"][0, :500]
        input_emb[poi] += pre_emb.detach().cpu().clamp(-0.025, 0.025)[:500]

        output_emb[poi] = sample_data["geo_K_embed"][0, 500:]
        output_emb[poi] += pre_emb.detach().cpu().clamp(-0.025, 0.025)[500:]

    refine_data = [input_emb, output_emb]
    avg_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case = adapt_test(refine_data, poi_args)
    return avg_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case

def train(poi_args, diff_data):
    os.makedirs(f"./logs/{poi_args.data_name}", exist_ok=True)
    f_log = open(f"./logs/{poi_args.data_name}/{poi_args.data_name}.txt", "a")
    f_log.write(f"poi_args: {poi_args}\n")

    device = torch.device(poi_args.device if torch.cuda.is_available() else 'cpu')

    data = DiffSet(dataset=diff_data)
    data_loader = DataLoader(data, batch_size=poi_args.bs, shuffle=True)

    # set model, paras in args
    model = DI_GE_IO(device=device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=poi_args.lr, weight_decay=1e-5) 

    for epoch in range(poi_args.epoch):
        if epoch > 0 and epoch % 5 == 0:
            model.eval()
            avg_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case = test(model, data, device, poi_args)

            f_log.write(f"Adapted model performance (Ours):\n")
            f_log.write(f"avg_user_acc: {avg_acc} avg_warm_user_acc: {avg_warm_acc} avg_cold_user_acc: {avg_cold_acc} \n")
            f_log.write(f"avg_case_acc: {avg_acc_case} avg_warm_case_acc: {avg_warm_acc_case} avg_cold_case_acc: {avg_cold_acc_case} \n")
            f_log.write(f"----------------------------------------------------------------- \n")
            f_log.flush()

        model.train()
        loss_data = 0.0
        start = time.time()
        for i_batch, sample_batched in enumerate(data_loader):
            item_embed = sample_batched["item_embed"].to(device)
            geo_K_dist = sample_batched["geo_K_dist"].to(device)
            geo_K_embed = sample_batched["geo_K_embed"].to(device)
            sem_K_dist = sample_batched["sem_K_dist"].to(device)
            sem_K_embed = sample_batched["sem_K_embed"].to(device)
            cate = sample_batched["cate"].to(device)
            coord = sample_batched["coord"].to(device)

            batch_data = [item_embed, geo_K_dist, geo_K_embed, sem_K_dist, sem_K_embed, cate, coord]
            batch_loss = model(batch_data)

            opt.zero_grad()
            batch_loss.backward()
            loss_data += batch_loss.data.cpu().numpy()
            opt.step() # add some schedules
    f_log.close()

def get_poi_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='NYC')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--path', type=str, default="../base_model/model/simple/model.pt")
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=True, const=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    poi_args = get_poi_args()
    diff_data = fast_entry_fn(poi_args.path, poi_args)
    print(poi_args)

    if poi_args.fix_seed:
        # np.random.seed(42)
        # torch.manual_seed(42)

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print("seed fixed for reproducibility")
        print("cuda device:", poi_args.device)

    train(poi_args, diff_data)
