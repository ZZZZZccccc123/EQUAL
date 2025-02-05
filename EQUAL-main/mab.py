import math
import json
from tqdm import tqdm
import multiprocessing
import random
import pickle
import csv
import torch
import os
import pdb
from otscore import get_optimal_score
from typing import Any
import copy


import pandas as pd

import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import torch
import subprocess

import time
alpha = 0.001
cluster_size = 900
batchsize = 10  # Number of samples drawn from each cluster to estimate the overall optimal transport of the cluster
threshold = 0.2
batch = 50  # Number of clusters to sample from
num_gpu = 0  # Number of GPUs used for parallel computation
real_batchsize = 10  # Actual number of samples drawn from a specific cluster

cluster_score = {i: 0 for i in range(0, cluster_size)}  # Reward score for each cluster
cluster_chose = {i: 0 for i in range(0, cluster_size)}  # Number of times each cluster was selected (including when its neighbors were selected)
cluster_ucb = {i: 0 for i in range(0, cluster_size)}  # UCB score for each cluster
cluster_sample = {i: 0 for i in range(0, cluster_size)}  # Number of times each cluster was actually sampled
final_chose = {i: 0 for i in range(0, cluster_size)}  # Clusters that were ultimately selected
forbidden_cluster = []  # Clusters whose data have been fully taken
sum_chose = 0


def get_cluster_neighbour():
    print("getting cluster neighbour")
    cluster_neighbour = {}
    df = pd.read_csv('./data/cluster/AutoMathText-matrix.csv')
    cluster_distance = df.values

    for i in tqdm(range(cluster_size)):
        max_distance = 0
        cluster_neighbour[i] = []
        for j in range(cluster_size):
            max_distance = max(max_distance, cluster_distance[i][j])
            if cluster_distance[i][j] < threshold:
                cluster_neighbour[i].append(j)
    return cluster_neighbour, cluster_distance

def sample_minibatch(cluster_number):
    all_datasets = []
    with open("./data/cluster/AutoMathText_filter-q1q2_emb/dist-to-centroid-" + str(cluster_number) + ".jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            all_datasets.append(data)
    if((cluster_sample[cluster_number]+1)*real_batchsize<=len(all_datasets)):
        strlist = all_datasets[cluster_sample[cluster_number]*real_batchsize:(cluster_sample[cluster_number]+1)*real_batchsize]
        return True, strlist
    else:
        cluster_score[cluster_number] = -9999999
        cluster_ucb[cluster_number] = -9999999
        return False, None





def run_command(cmd):
    os.system(cmd)



def offline(merged_minibatch, batchsize):
    result = []
    for i in range(0, len(merged_minibatch), batchsize):
        tmp = []
        part = merged_minibatch[i:i+batchsize]
        for i in range(len(part)):
            tmp = tmp + part[i]["QA_pair"]
        result.append(get_optimal_score(tmp))
    print(sum(result)/len(result))
    return result

def get_cluster_reward(merged_minibatch, batchsize):
    final_result = offline(merged_minibatch, batchsize)
    return final_result

def sorted_highest_score(k=batch):
    sorted_items = sorted(cluster_ucb.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_items]
    return sorted_keys

selected_data = []

final_sum_chose = 0
final_number = 0
iteration = 146
sum_average = 0
pretrain_data = []
cluster_neighbour, cluster_distance = get_cluster_neighbour()
for k in tqdm(range(iteration)):
    merged_minibatch = []
    topk = []
    sorted_keys = sorted_highest_score(k=batch)
    loopindex = 0
    for i in range(batch):
        flag = False
        while flag == False:
            if k == 0:
                sample_index = loopindex
                loopindex = loopindex + 1
            else:
                sample_index = sorted_keys[loopindex]
                loopindex = loopindex + 1
            flag, minibatch = sample_minibatch(sample_index)
        merged_minibatch = merged_minibatch + minibatch
        topk.append({sample_index:cluster_sample[sample_index]})
    reward = get_cluster_reward(merged_minibatch, batchsize)
    averages = reward
    iteration_average = 0
    idict_index = 0
    for i_dict in topk:
        for key, value in i_dict.items():
            i = int(key)
            cluster_sample[i] += 1
        for cluster in cluster_neighbour[i]:
            sum_chose += 1
            cluster_score[cluster] += averages[idict_index]
            cluster_chose[cluster] += 1
        idict_index = idict_index + 1
    for key in cluster_ucb.keys():
        if cluster_chose[key] != 0:
            cluster_average_reward = cluster_score[key]/cluster_chose[key]
        else:
            cluster_average_reward = 0
        ucb = alpha*math.sqrt(2*(math.log(float(sum_chose)))/float(cluster_chose[key]+1))
        cluster_ucb[key] = cluster_average_reward + ucb
    for i in range(len(merged_minibatch)):
        for sample in merged_minibatch[i]["QA_pair"]:
            sample["id"] = merged_minibatch[i]["doc_id"]
            selected_data.append(sample)

    print(len(selected_data))

random.shuffle(selected_data)
with open('target.jsonl', 'w') as file:
    for item in selected_data:
        json.dump(item, file)
        file.write('\n')
print(len(selected_data))
