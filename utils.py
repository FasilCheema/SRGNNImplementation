'''
Author: Fasil Cheema
Purpose: This module contains utility functions for data 
          processing and preparation for this specific project
          This code is based/inspired off the paper and repo SRGNN:
          (Zhu, Qi, et al. "Shift-robust gnns: Overcoming the ...
          ... limitations of localized graph training data." ...
          ...  Advances in Neural Information Processing
          ...   Systems 34 (2021): 27965-27977.)
'''

import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from itertools import combinations 
from collections import defaultdict


def data_loader(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for i in range(len(names)):

        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f: 
            
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    
    if dataset_str == 'pubmed':
        idx_train = range(10000)
    elif dataset_str == 'cora':
        idx_train = range(1500)
    else:
        idx_train = range(1000)

    idx_val = range(len(y), len(y)+500)
    
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def createTraining(labels, max_train=200, balance=True, new_classes=[]):
    dist = defaultdict(list)
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)

    for idx,l in enumerate(labels.numpy().tolist()[:max_train]):
        dist[l].append(idx)
    # print(dist)
    cat = []
    _sum = 0
    if balance:
        for k in dist:
            if k in new_classes:
                continue
            _sum += len(dist[k])
            # cat += random.sample(dist[k], k=15)
            train_mask[random.sample(dist[k], k=3)] = 1
    for k in new_classes:
       train_mask[random.sample(dist[k], k=3)] = 1 
    # print(_sum, sum(train_mask))
    return train_mask
    # print(len(set(cat)))

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def createDBLPTraining(labels, idx_train, idx_val, idx_test, max_train=20, balance=True, new_classes=[], unknown=False):
    
    labels = [np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in labels]
    
    new_mapping = {}
    dist = defaultdict(list)
    new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test = [], [], [], [], []
    
    for idx in idx_train:
        dist[labels[idx]].append(idx)

    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)

    for k in dist:
        if max_train < len(dist[k]):
            new_idx_train += np.random.choice(dist[k], max_train, replace=False).tolist()
        else:
            new_idx_train += dist[k]


    for idx in idx_val:
        if labels[idx] in new_mapping:
            #unknown label id
            new_idx_val.append(idx)
        else:
            new_idx_val.append(idx)
    
    for idx in idx_test:
        if labels[idx] in new_mapping:
            #unknown label id
            new_idx_test.append(idx)
            in_idx_test.append(idx)
        else:
            #unknown class
            if unknown:
                new_idx_test.append(idx)
                out_idx_test.append(idx)
    
    for idx,label in enumerate(labels):
        if label < 0:
            continue
        if label in new_mapping:
            labels[idx] = new_mapping[label]
        else:
            labels[idx] = len(new_mapping)
    
    return new_idx_train, new_idx_val, in_idx_test, new_idx_test, out_idx_test, labels

#Not used to be used for other experiments
#generateUnseen(num_class,num_unseen)
def generateUnknown(num_class, num_unknown):
    
    unknown_samples = combinations(range(num_class), num_unknown)

    return unknown_samples

