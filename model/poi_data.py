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

from poi_data import *
from poi_utils import *
from poi_model import *

def get_k_largest(array, k):
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(array)[::-1]
    
    # Get the K largest values and their corresponding indices
    k_largest_values = array[sorted_indices[:k]]
    k_largest_indices = sorted_indices[:k]
    
    return k_largest_values, k_largest_indices

def search_neigh(simi_array, poi, K):
    simi_vector = simi_array[poi]
    # simi_vector = simi_vector / np.sum(simi_vector)
    largest_values, largest_indices = get_k_largest(simi_vector, K)

    K_dist = largest_values
    K_dist = K_dist / np.sum(K_dist)
    
    K_embed = largest_indices 
    return K_dist, K_embed

class DiffSet(Dataset):
    def __init__(self, dataset, K=10):
        c2p, warm_data, cold_data = dataset
        self.warm_data = warm_data
        self.cold_data = cold_data

        # norm weight, bigger better for kernels
        path = "../data/meta_data/geo_simi.pt"
        if not os.path.exists(path):
            geo_simi = self.ini_geo(warm_data, cold_data)
            torch.save(geo_simi, path)
        else:
            geo_simi = torch.load(path)

        path = "../data/meta_data/sem_simi.pt"
        if not os.path.exists(path):
            self.IC(c2p)
            sem_simi = self.ini_sem(warm_data, cold_data, c2p)
            torch.save(sem_simi, path)
        else:
            sem_simi = torch.load(path)

        # feat transformation
        tfidf_model = self.ini_tfidf(warm_data, cold_data)

        # for each poi, define K neighs and distances
        train_data = {}
        for poi in warm_data:
            if poi not in train_data:
                train_data[poi] = {}
                train_data[poi]["i_embed"] = warm_data[poi]["i_embed"]
                train_data[poi]["o_embed"] = warm_data[poi]["o_embed"]
            
            geo_K_dist, geo_K_embed = search_neigh(geo_simi, poi, K)
            train_data[poi]["geo_K_dist"] = geo_K_dist
            # maybe embedding format, latter
            train_data[poi]["geo_K_embed"] = geo_K_embed
            
            sem_K_dist, sem_K_embed = search_neigh(sem_simi, poi, K)
            train_data[poi]["sem_K_dist"] = sem_K_dist
            train_data[poi]["sem_K_embed"] = sem_K_embed

            text_vector = tfidf_model.transform([warm_data[poi]["feat"][0]])
            text_vector = text_vector.toarray()[0]
            train_data[poi]["cate"] = text_vector

            coord = [float(warm_data[poi]["feat"][2]), float(warm_data[poi]["feat"][1])]
            train_data[poi]["coord"] = coord

        test_data = {}
        for poi in cold_data:
            if poi not in test_data:
                test_data[poi] = {}
                test_data[poi]["i_embed"] = None
                test_data[poi]["o_embed"] = None

            geo_K_dist, geo_K_embed = search_neigh(geo_simi, poi, K)
            test_data[poi]["geo_K_dist"] = geo_K_dist
            test_data[poi]["geo_K_embed"] = geo_K_embed
            
            sem_K_dist, sem_K_embed = search_neigh(sem_simi, poi, K)
            test_data[poi]["sem_K_dist"] = sem_K_dist
            test_data[poi]["sem_K_embed"] = sem_K_embed

            text_vector = tfidf_model.transform([cold_data[poi]["feat"][0]])
            text_vector = text_vector.toarray()[0]
            test_data[poi]["cate"] = text_vector

            coord = [float(cold_data[poi]["feat"][2]), float(cold_data[poi]["feat"][1])]
            test_data[poi]["coord"] = coord

        self.train_data = train_data
        self.test_data = test_data

    def ini_tfidf(self, warm_data, cold_data):
        warm_pois = list(warm_data.keys())
        cold_pois = list(cold_data.keys())
        all_pois = warm_pois + cold_pois

        text_data = []
        for poi in all_pois:
            if poi in warm_data:
                feat = warm_data[poi]["feat"][0]
            else:
                feat = cold_data[poi]["feat"][0]
            text_data.append(feat)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_data)
        return vectorizer

    def ini_geo(self, warm_data, cold_data):
        warm_pois = list(warm_data.keys())
        cold_pois = list(cold_data.keys())
        all_pois = warm_pois + cold_pois

        geo_simi = np.zeros((len(all_pois), len(warm_pois)))

        for s_poi in all_pois:
            for t_poi in warm_pois:
                if s_poi == t_poi:
                    kernel_val = 0.0
                    geo_simi[s_poi][t_poi] = kernel_val
                    continue

                if s_poi in warm_data:
                    s_feat = warm_data[s_poi]["feat"]
                else:
                    s_feat = cold_data[s_poi]["feat"]
                t_feat = warm_data[t_poi]["feat"]

                s_coord = [float(s_feat[2]), float(s_feat[1])]
                t_coord = [float(t_feat[2]), float(t_feat[1])]
                kernel_val = self.RBF_kernel(s_coord, t_coord)
                geo_simi[s_poi][t_poi] = kernel_val

        return geo_simi

    def ini_sem(self, warm_data, cold_data, c2p):
        warm_pois = list(warm_data.keys())
        cold_pois = list(cold_data.keys())
        all_pois = warm_pois + cold_pois
        
        sem_simi = np.zeros((len(all_pois), len(warm_pois)))

        for s_poi in all_pois:
            for t_poi in warm_pois:
                if s_poi == t_poi:
                    kernel_val = 0.0
                    sem_simi[s_poi][t_poi] = kernel_val
                    continue
                if s_poi in warm_data:
                    s_feat = warm_data[s_poi]["feat"]
                else:
                    s_feat = cold_data[s_poi]["feat"]
                t_feat = warm_data[t_poi]["feat"]

                s_cate = s_feat[0]
                t_cate = t_feat[0]
                kernel_val = self.Tree_kernel(s_cate, t_cate, c2p)
                sem_simi[s_poi][t_poi] = kernel_val

        return sem_simi

    # [lat, lon]
    def RBF_kernel(self, p1, p2, gamma=0.05): 
        dist = geopy.distance.geodesic(p1, p2).km
        kernel = np.exp(-gamma * dist ** 2) # bigger better
        return kernel

    def IC(self, c2p):
        def count(tree, key):
            num = 1
            for c in tree[key]:
                num += count(tree, c)
            return num

        p2c = {}
        for c in c2p:
            if c not in p2c:
                p2c[c] = []
            parent = c2p[c]
            if parent not in p2c:
                p2c[parent] = [] 
            p2c[parent].append(c)
        
        # num
        num_dict = {}
        for node in p2c:
            num_dict[node] = count(p2c, node)
        IC_dict = {}
        max_IC = 0
        for node in num_dict:
            num = num_dict[node]
            IC = -1 * math.log10(num/num_dict["Root"])
            IC_dict[node] = IC
            max_IC = max(max_IC, IC)
        self.IC_dict = IC_dict
        self.max_IC = max_IC

    def Tree_kernel(self, c1, c2, c2p, theta=1.0):
        def find_parent(c, c2p):
            parents = [c]
            while c != "Root":
                p = c2p[c]
                parents.append(p)
                c = p
            return parents

        def sub(c1, c2, c2p):
            c1_parents = find_parent(c1, c2p)
            c2_parents = find_parent(c2, c2p)
            
            for i in range(len(c1_parents)):
                node_i = c1_parents[i]
                for j in range(len(c2_parents)):
                    node_j = c2_parents[j]
                    if node_i == node_j:
                        return node_i

        def hop(c1, c2, c2p, node):
            c1_parents = find_parent(c1, c2p)
            c2_parents = find_parent(c2, c2p)
            path = 0 
            for c in c1_parents:
                if c != node:
                    path += 1 
                else:
                    break 
            for c in c2_parents:
                if c != node:
                    path += 1 
                else:
                    break 
            return path

        node = sub(c1, c2, c2p)
        IC = self.IC_dict[node] # bigger better

        path = hop(c1, c2, c2p, node)
        mh = 5
        SC = math.pow(1+theta, mh-path) 
        return IC * SC 

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample_data = self.train_data[idx]

        i_embed = sample_data["i_embed"].unsqueeze(0)
        o_embed = sample_data["o_embed"].unsqueeze(0)
        item_embed = torch.cat([i_embed, o_embed], dim=1)

        geo_K_dist = sample_data["geo_K_dist"]
        geo_K_ids = sample_data["geo_K_embed"]
        geo_K_embed = []
        for key in geo_K_ids:
            i_embed = self.warm_data[key]["i_embed"]
            o_embed = self.warm_data[key]["o_embed"]

            geo_item_embed = torch.cat([i_embed, o_embed], dim=0)
            geo_K_embed.append(geo_item_embed)
        geo_K_embed = torch.stack(geo_K_embed)

        sem_K_dist = sample_data["sem_K_dist"]
        sem_K_ids = sample_data["sem_K_embed"]
        sem_K_embed = []
        for key in sem_K_ids:
            i_embed = self.warm_data[key]["i_embed"]
            o_embed = self.warm_data[key]["o_embed"]

            sem_item_embed = torch.cat([i_embed, o_embed], dim=0)
            sem_K_embed.append(sem_item_embed)
        sem_K_embed = torch.stack(sem_K_embed)

        cate = sample_data["cate"]
        coord = sample_data["coord"]

        sample = {
            "item_embed": torch.FloatTensor(item_embed),
            "geo_K_dist": torch.FloatTensor(geo_K_dist),
            "geo_K_embed": torch.FloatTensor(geo_K_embed),
            "sem_K_dist": torch.FloatTensor(sem_K_dist),
            "sem_K_embed": torch.FloatTensor(sem_K_embed),
            "cate": torch.FloatTensor(cate),
            "coord": torch.FloatTensor(coord)
        }
        return sample

    def __get_test_item__(self, idx):
        sample_data = self.test_data[idx] 

        mean = 0.0
        stddev = 1.0
        impute_size = 500

        i_embed = torch.randn(1, impute_size) * stddev + mean
        o_embed = torch.randn(1, impute_size) * stddev + mean
        item_embed = torch.cat([i_embed, o_embed], dim=1)

        geo_K_dist = sample_data["geo_K_dist"]
        geo_K_ids = sample_data["geo_K_embed"]
        geo_K_embed = []
        for key in geo_K_ids:
            i_embed = self.warm_data[key]["i_embed"]
            o_embed = self.warm_data[key]["o_embed"]

            geo_item_embed = torch.cat([i_embed, o_embed], dim=0)
            geo_K_embed.append(geo_item_embed)
        geo_K_embed = torch.stack(geo_K_embed)

        sem_K_dist = sample_data["sem_K_dist"]
        sem_K_ids = sample_data["sem_K_embed"]
        sem_K_embed = []
        for key in sem_K_ids:
            i_embed = self.warm_data[key]["i_embed"]
            o_embed = self.warm_data[key]["o_embed"]

            sem_item_embed = torch.cat([i_embed, o_embed], dim=0)
            sem_K_embed.append(sem_item_embed)
        sem_K_embed = torch.stack(sem_K_embed)

        cate = sample_data["cate"]
        coord = sample_data["coord"]

        sample = {
            "item_embed": torch.FloatTensor(item_embed),
            "geo_K_dist": torch.FloatTensor(geo_K_dist),
            "geo_K_embed": torch.FloatTensor(geo_K_embed),
            "sem_K_dist": torch.FloatTensor(sem_K_dist),
            "sem_K_embed": torch.FloatTensor(sem_K_embed),
            "cate": torch.FloatTensor(cate),
            "coord": torch.FloatTensor(coord)
        }
        return sample

if __name__ == '__main__':
    base_args, diff_data = fast_entry_fn()
    data = DiffSet(diff_data)

    sample = data.__getitem__(0)
    print(sample)