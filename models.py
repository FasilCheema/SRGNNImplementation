
'''
Author: Fasil Cheema
Purpose: This module contains the GNN models used in this project
          This code is based/inspired off the paper and repo SRGNN:
          (Zhu, Qi, et al. "Shift-robust gnns: Overcoming the ...
          ... limitations of localized graph training data." ...
          ...  Advances in Neural Information Processing
          ...   Systems 34 (2021): 27965-27977.)
'''


import math
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from matrixtools import calc_A_hat
from torch.autograd import Function
from dgl.nn.pytorch.conv import SAGEConv,GraphConv,GATConv, SGConv

#Returns the normalized mean of the samples
def norm2(samples):
    
    L2Norm = F.normalize(samples, p=2, dim=1)
    
    return L2Norm

#Function to Convert a sparsematrix to torch (type=scipy.sparse)
def spmat2torch(X):
    
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    
    spmat_torch = torch.sparse.FloatTensor(torch.LongTensor(indices),torch.FloatTensor(coo.data),coo.shape)

    return spmat_torch

#Function to Convert a matrix to torch (type=scipy.sparse or matrix from numpy)
def matrix_to_torch(X):

    if sp.issparse(X):
    
        torch_mat = spmat2torch(X) 
    
    else:
    
        torch_mat = torch.FloatTensor(X)

    return torch_mat

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)

#Personalized Page Rank
class PPRPowerIteration(nn.Module):
    
    def __init__(self, in_feats, n_hidden, n_classes, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', spmat2torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.fcs = nn.ModuleList([nn.Linear(in_feats, n_hidden, bias=False), nn.Linear(n_hidden, n_classes, bias=False)])
        self.disc = nn.Linear(n_hidden, 2)
        self.bns = nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
    
    def forward(self, local_preds: torch.FloatTensor, bns = False):
        
        for l_id, layer in enumerate(self.fcs):
            local_preds = self.dropout(local_preds)
            if l_id != len(self.fcs) - 1:
                #print('here')
                local_preds = layer(local_preds)
                if bns:
                    local_preds = self.bns[l_id](local_preds)
                local_preds = F.tanh(local_preds)
            else:
                self.h = local_preds
                local_preds = layer(local_preds)
        
        preds = local_preds

        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds

        return preds
    
#Class to generate GraphSAGE model 
class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.activation = activation
        
        #Generate input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, norm=norm2, feat_drop=dropout, activation=activation))
        
        #Generate hidden layers
        for i in range(n_layers-1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, norm=norm2, feat_drop=dropout, activation=activation))
        
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None
    
    def forward(self, features):
        
        h = features
        
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        
        self.h = h
        
        feed_forward = self.layers[-1](self.g,h)
        
        return feed_forward
        
    
    def output(self, features):
        h = features

        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        
        return h

#Class to generate standard GCN
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        #Generate input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        
        #Generate hidden layers
        self.activation = activation
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))

        for i in range(n_layers-1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        
        #Generate output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None)) # activation None
        
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=True), nn.Linear(n_hidden, 2, bias=True)])
        self.disc = GraphConv(n_hidden, 2, activation=None)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, features, bns=False):
        
        h = features
        
        for idx, layer in enumerate(self.layers[:-1]):
            
            h = layer(self.g, h)
            
            if bns:
                h = self.bns[idx](h)
            
            h = self.activation(h)
            h = self.dropout(h)
        
        self.h = h
        feed_forward =  self.layers[-1](self.g, h)

        return feed_forward 

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

#Class of Graph Attention Transformer; inspired by Pytorch implementation
class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 num_heads=8):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        
        #Generate all hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        
        #Generate the output layer
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features, bins=False):
        
        h = features
        
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](self.g, h).flatten(1)
        
        self.h = h
        
        feed_forward = self.layers[-1](self.g, h).mean(1)

        return feed_forward

    def output(self, g, features):

        h = features

        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)

        out = self.layers[-1](g, h).mean(1)

        return out

#Class of simplifying graph convolutional networks: note this does not have hidden layers
# this just has the form Y = softmax(\Tilde{X} \Theta) where both \Tilde{X} \Theta are matrices  
class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 train_mask):
        super(SGC, self).__init__()
        
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SGConv(in_feats, n_hidden, k=2, cached=True))
        
        self.linear = nn.Linear(n_hidden, n_classes)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.train_mask = train_mask

    def forward(self, features, bns=False):
        
        h = features
        
        for layer in self.layers:
            h = layer(self.g, h)
            
        h = self.activation(h)
        self.h = h
        feed_forward = self.linear(h)
        
        return feed_forward

#This should be removed I have fixed the GCN module above:
#Class to generate graph convolutional networks
'''
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

'''