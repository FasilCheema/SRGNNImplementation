'''
Author: Fasil Cheema
Purpose: This module contains computationally demanding functions run on matrices
          This code is based/inspired off the paper and repo SRGNN:
          (Zhu, Qi, et al. "Shift-robust gnns: Overcoming the ...
          ... limitations of localized graph training data." ...
          ...  Advances in Neural Information Processing
          ...   Systems 34 (2021): 27965-27977.)
'''

import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from cvxopt import matrix, solvers



#Function to return the L2 norm of 2 vectors
def l2diff(vec1, vec2):
    
    norm = (vec1-vec2).norm(p=2)

    return norm  


#Function to return the difference of moments (specifically the mean)
def moment_diff(sx1, sx2, k):
 
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    m_diff = l2diff(ss1,ss2)

    return m_diff

#Function to compute the accuracy given predictions, labels, and an evaluator
def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]

#Function to compute the central moment disrepancy (Zellinger, Werner, et al. ICLR 2017) 
def cmd(X, X_test, K=5):
    
    x1 = X
    x2 = X_test
    
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    
    std_x1 = x1 - mean_x1
    std_x2 = x2 - mean_x2
    
    dm = l2diff(mean_x1,mean_x2)
    scms = [dm]

    for i in range(K-1):
        scms.append(moment_diff(std_x1,std_x2,i+2))
    
    cmd_result = sum(scms)

    return cmd_result


#Function to return the cross entropy given the input and the labels
def cross_entropy(x, labels):
    
    y = F.cross_entropy(x, labels.view(-1), reduction="none")
    ce = torch.mean(y)
    
    return ce

#Function to compute pairwise distances; if y is not given distance is computed for x with itself.
def pairwise_distances(x, y=None):

    x_norm = (x**2).sum(1).view(-1, 1)
    
    #checks if there is a y arg; if not computes dist matrix for x with itself
    if y is not None:
        y_tpose = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_tpose = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_tpose)
    
    dist_mat = torch.clamp(dist, 0.0, np.inf)

    return dist_mat 

# Function to compute the Maximum Mean Disrepancy (Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, Alexander Smola, JMLR 2012) 
def MMD(X,Xtest):

    CONST_1 = 1e0
    CONST_2 = 1e-1
    CONST_3 = 1e-3

    X_dist = torch.exp(- CONST_1 * pairwise_distances(X)) + torch.exp(- CONST_2 * pairwise_distances(X)) + torch.exp(- CONST_3 * pairwise_distances(X))
    
    Y_dist = torch.exp(- CONST_1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- CONST_2 * pairwise_distances(Xtest, Xtest)) + torch.exp(- CONST_3 * pairwise_distances(Xtest, Xtest))
    
    Cross_dist = torch.exp(- CONST_1 * pairwise_distances(X, Xtest)) + torch.exp(- CONST_2 * pairwise_distances(X, Xtest)) + torch.exp(- CONST_3 * pairwise_distances(X, Xtest))
    
    MMD_dist = X_dist.mean() + Y_dist.mean() - 2 * Cross_dist.mean() 
    
    return MMD_dist

#Function to compute Kernel Mean Matching
def KMM(X,Xtest, A_mat=None, _sigma=1e1):
    
    num = X.shape[0]
    
    b = np.ones([A_mat.shape[0],1]) * 20
    h = - 0.2 * np.ones((num,1))
    G = - np.eye(num)
    
    CONST_1 = 1e0
    CONST_2 = 1e-1
    CONST_3 = 1e-3

    X_dist = (torch.exp(- CONST_1 * pairwise_distances(X)) + torch.exp(- CONST_2 * pairwise_distances(X)) + torch.exp(- CONST_3 * pairwise_distances(X)))/3
    
    Cross_dist = (torch.exp(- CONST_1 * pairwise_distances(X, Xtest)) + torch.exp(- CONST_2 * pairwise_distances(X, Xtest)) + torch.exp(- CONST_3 * pairwise_distances(X, Xtest)))/3
    
    Y_dist = torch.exp(- CONST_1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- CONST_2 * pairwise_distances(Xtest, Xtest)) + torch.exp(- CONST_3 * pairwise_distances(Xtest, Xtest))
    
    
    MMD_dist = X_dist.mean() + Y_dist.mean() - 2 * Cross_dist.mean()
    
    Cross_dist = - X.shape[0] / Xtest.shape[0] * Cross_dist.matmul(torch.ones((Xtest.shape[0],1)))

    solvers.options['show_progress'] = False
    
    solution = solvers.qp(matrix(X_dist.numpy().astype(np.double)), matrix(Cross_dist.numpy().astype(np.double)), matrix(G), matrix(h), matrix(A_mat), matrix(b))

    KMM_sol = np.array(solution['x'])
    MMD_sol = MMD_dist.item()

    return KMM_sol, MMD_sol
    
#Function that takes the adjacency matrix and node features and smooths the features via diagonals
def calc_feat_smooth(adj, features):

    A = sp.diags(adj.sum(1).flatten().tolist())
    D = (A - adj)

    smooth_feat = (D * features)

    return smooth_feat
    

#Function that takes the adjacency matrix and node features and smooths the embeddings via diagonals
def calc_emb_smooth(adj, features):

    A = sp.diags(adj.sum(1).flatten().tolist())
    D = (A - adj)
    
    smooth_emb = ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])

    return smooth_emb

#Function to compute the approximate Matrix which is a type sp.spmatrix, and it returns a sp.spmatrix
def calc_A_hat(adj_matrix ):

    num_nodes = adj_matrix.shape[0]
    I_mat = sp.eye(num_nodes)

    A = adj_matrix + I_mat
    
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)

    #matrix multiplication (D A) (D)
    A_hat = D_invsqrt_corr @ A @ D_invsqrt_corr

    return A_hat