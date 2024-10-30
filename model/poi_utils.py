from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import argparse
import numpy as np
from json import encoder

import sys 
sys.path.append("../")
sys.path.insert(0, '../base_model/')

from base_model.train import run_simple

def run(PATH, poi_args):
    test_setting = torch.load(PATH)
    model, data_test, test_idx, optimizer, criterion, parameters = test_setting
    print("Have loaded")

    output_emb = model.fc.weight.data
    input_emb = model.emb_loc.weight.data

    cold_pois = parameters.cold_pois
    warm_pois = parameters.warm_pois

    avg_loss, avg_acc, users_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case = run_simple(data_test, test_idx, 'test', 0.002, parameters.clip, model,
                                                      optimizer, criterion, parameters.model_mode, parameters)
    print('==>Test User Acc:{:.4f} Warm User Acc:{:.4f} Cold User Acc:{:.4f}'.format(avg_acc, avg_warm_acc, avg_cold_acc))
    print('==>Test Case Acc:{:.4f} Warm Case Acc:{:.4f} Cold Case Acc:{:.4f}'.format(avg_acc_case, avg_warm_acc_case, avg_cold_acc_case))
    print('----------------------------------------')

    os.makedirs(f"./logs/{poi_args.data_name}", exist_ok=True)
    f_log = open(f"./logs/{poi_args.data_name}/{poi_args.data_name}.txt", "w")

    f_log.write(f"Base model performance:\n")
    f_log.write(f"avg_user_acc: {avg_acc} avg_warm_user_acc: {avg_warm_acc} avg_cold_user_acc: {avg_cold_acc} \n")
    f_log.write(
        f"avg_case_acc: {avg_acc_case} avg_warm_case_acc: {avg_warm_acc_case} avg_cold_case_acc: {avg_cold_acc_case} \n")
    f_log.write(f"================================================================ \n")
    f_log.flush()

    return output_emb, input_emb, parameters

def parse_meta_data():
    path = "../data/meta_data/4sq_category_tree.txt"

    level_1, level_2, level_3 = "", "", ""
    c2p = {}
    for line in open(path, "r"):
        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            level_1 = line.strip()
            c2p[level_1] = "Root"
        if lspace == 4:
            level_2 = line.strip()
            c2p[level_2] = level_1
        if lspace == 8:
            level_3 = line.strip()
            c2p[level_3] = level_2
    return c2p

def anomaly_leaf(leaf_node):
    # anomaly category
    if leaf_node == "Athletic & Sport":
        leaf_node = "Athletics & Sports"
    if leaf_node == "Light Rail":
        leaf_node = "Light Rail Station"
    if leaf_node == "Spa / Massage":
        leaf_node = "Spa"
    if leaf_node == "Ferry":
        leaf_node = "Boat or Ferry"
    if leaf_node == "Sushi Restaurant":
        leaf_node = "Japanese Restaurant"
    if leaf_node == "Ramen /  Noodle House":
        leaf_node = "Noodle House"
    if leaf_node == "Café":
        leaf_node = "Cafeteria"
    if leaf_node == "Subway":
        leaf_node = "Bus Station"
    return leaf_node


def adapt_test(refine_data, poi_args):
    input_emb, output_emb = refine_data
    input_emb = input_emb.cuda(0)
    output_emb = output_emb.cuda(0)


    test_setting = torch.load(poi_args.path)
    model, data_test, test_idx, optimizer, criterion, parameters = test_setting

    print("*************** adapted *****************")
    model.fc.weight = nn.Parameter(output_emb)
    model.emb_loc.weight = nn.Parameter(input_emb)

    cold_pois = parameters.cold_pois
    warm_pois = parameters.warm_pois


    avg_loss, avg_acc, users_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case = run_simple(data_test, test_idx, 'test', 0.002, parameters.clip, model,
                                                      optimizer, criterion, parameters.model_mode, parameters)
    print('==>Test User Acc:{:.4f} Warm User Acc:{:.4f} Cold User Acc:{:.4f}'.format(avg_acc, avg_warm_acc, avg_cold_acc))
    print('==>Test Case Acc:{:.4f} Warm Case Acc:{:.4f} Cold Case Acc:{:.4f}'.format(avg_acc_case, avg_warm_acc_case, avg_cold_acc_case))

    return avg_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case

def collect_training(output_emb, input_emb, parameters):
    cold_pois = parameters.cold_pois
    warm_pois = parameters.warm_pois

    cold_pois = list(cold_pois)
    cold_pois.sort()

    warm_pois = list(warm_pois)
    warm_pois.sort()

    poi2feat = parameters.pid2feat
    new_poi2feat = {}
    for poi in poi2feat:
        feat = poi2feat[poi]
        new_feat = [anomaly_leaf(feat[0]), feat[1], feat[2]]
        new_poi2feat[poi] = new_feat
    poi2feat = new_poi2feat
    
    warm_data = {}
    cold_data = {}
    for poi in warm_pois:
        feat = poi2feat[poi]
        i_embed = input_emb[poi]
        o_embed = output_emb[poi]
        warm_data[poi] = {}
        warm_data[poi]["feat"] = feat
        warm_data[poi]["i_embed"] = i_embed.cpu()
        warm_data[poi]["o_embed"] = o_embed.cpu()
    for poi in cold_pois:
        feat = poi2feat[poi]
        cold_data[poi] = {}
        cold_data[poi]["feat"] = feat 

    meta_data = parse_meta_data()

    diff_data = [meta_data, warm_data, cold_data]
    return diff_data

def fast_entry_fn(path, poi_args):
    output_emb, input_emb, parameters = run(path, poi_args)
    diff_data = collect_training(output_emb, input_emb, parameters)
    return diff_data
