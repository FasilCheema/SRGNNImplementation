import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

from IPython import embed
import dgl
from models import GCN, GraphSAGE, PPRPowerIteration, SGC, GAT

from sklearn import preprocessing
from sklearn.metrics import f1_score
import networkx as nx

import utils
import argparse, pickle

import scipy.sparse as sp

from cvxopt import matrix, solvers


import warnings

warnings.simplefilter("ignore")

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


def main(args, new_classes):
    verbose = args.verbose
    device = torch.device("cpu")
    unk = False

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = utils.data_loader(args.dataset)
        labels = [np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]
        #idx_train, idx_val, in_idx_test, idx_test, out_idx_test, labels = utils.createTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, new_classes=new_classes, unknown=unk)
        features = torch.FloatTensor(utils.preprocess_features(features))
    
    device = torch.device("cpu")

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        # important to add self-loop
        min_max_scaler = preprocessing.MinMaxScaler()
        features = F.normalize(features, p=1,dim=1)
        feat_smooth_matrix = calc_feat_smooth(adj, feat)
        
        nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
        
        g = dgl.from_networkx(nx_g)
    else:
        raise ValueError("wrong dataset name")
    
    max_train = 20
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    labels = torch.LongTensor(labels)
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)
    
    #if len(new_classes) > 0:
    #    nb_classes = max(labels[idx_val]).item()
    #else:
    nb_classes = max(labels).item() + 1

    #
    xent = nn.CrossEntropyLoss(reduction='none')
    #xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    STAGE = 'pretrain'
    print('number of classes {}'.format(nb_classes))
    #output_edgelist(g, open('{}_edgelist.txt'.format(args.dataset), 'w'))
    #pos_emb = read_posit_emb(open('{}_dw.emb'.format(args.dataset), 'r'))
    
    #EDITED CODE BY FTC:
    #Disable the below

    best_val_acc = 0
    cnt_wait = 0
    finetune = False
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    
    
    num_seeds = []
    all_runs_data = defaultdict(list)
    feature_smoothness = []
    embedding_smoothness = []
    avg_dist, max_dist = [], []
    
    train_dump = pickle.load(open('intermediate/{}_dump.p'.format(args.dataset), 'rb'))
    ppr_vector = train_dump['ppr_vector']
    ppr_dist = train_dump['ppr_dist']
    #
    avg_mmd_dist = []
    training_seeds_run = pickle.load(open('data/localized_seeds_{}.p'.format(args.dataset), 'rb'))
    
    for curr_iter in range(args.n_repeats):

        #Generate biased training data
        if args.biased_sample:
            
            idx_train = training_seeds_run[curr_iter]
            label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
            
            for i, idx in enumerate(idx_train):
                label_balance_constraints[labels[idx], i] = 1
            

            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            
            if args.dataset == 'cora':
                idx_test = list(all_idx)
            
            iid_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, max_train = max_train)
            
            if args.arch >= 3:
                kmm_weight, MMD_dist = KMM(ppr_vector[idx_train, :], ppr_vector[iid_train, :], label_balance_constraints)

        else:
            idx_seed = np.random.randint(0,features.shape[0])
            idx_train, _, _, _, _, _ = utils.createDBLPTraining(one_hot_labels, ori_idx_train, idx_val, idx_test, max_train = max_train, new_classes=new_classes, unknown=unk)
            
            all_idx = set(range(g.number_of_nodes())) - set(idx_train)
            label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
            for i, idx in enumerate(idx_train):
                label_balance_constraints[labels[idx], i] = 1
            
            test_lbls = labels[idx_test]

                
        train_lbls = labels[idx_train]
        reg_lbls = torch.cat([torch.ones(len(idx_train), dtype=torch.long), torch.zeros(len(idx_train), dtype=torch.long)])

        #Initializes the model based off the selected GNN architecture 
        if args.gnn_arch == 'graphsage':
            model = GraphSAGE(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    #F.relu,
                    F.relu,
                    args.dropout,
                    args.aggregator_type
                    )
        elif args.gnn_arch == 'gat':
            num_heads = 4
            model = GAT(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    F.tanh,
                    args.dropout,
                    num_heads
                    )
        elif args.gnn_arch == 'ppnp':
            model = PPRPowerIteration(ft_size, args.n_hidden, nb_classes, adj, alpha=0.1, niter=10, drop_prob=args.dropout)
        elif args.gnn_arch == 'sgc':
            train_mask = 0.5
            model = SGC(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    F.tanh,
                    args.dropout,
                    train_mask
                    )
        else:
            
            model = GCN(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    #F.relu,
                    F.tanh,
                    args.dropout,
                    args.aggregator_type
                    )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model
        best_acc, best_epoch = 0.0, 0.0
        plot_x, plot_y, plot_z = [], [], []
        for epoch in range(args.n_epochs):
            if args.arch == 4 and epoch % 20 == 1:
                kmm_weight, MMD_dist = KMM(model.h[idx_train, :].detach().cpu(), model.h[idx_test, :].detach().cpu(), label_balance_constraints)

            model.train()
            optimizer.zero_grad()
            
            if args.biased_sample and False:
                reg_logits = model.reg_output(features)
                loss_1, loss_reg = xent(logits[idx_train], train_lbls), 0.2 * xent(reg_logits[idx_train+reg_samples], reg_lbls)
                loss =  loss_1 # + loss_reg
                #loss =  loss_1
                # print(loss_1.item(), loss_reg.item()) 
            else:
                if args.dataset != 'ogbn-arxiv':
                    logits = model(features)
                    loss = xent(logits[idx_train], labels[idx_train])
                else:
                    logits = model(features)
                    loss = cross_entropy(logits[idx_train], labels[idx_train])
                
                if args.arch == 0:
                    loss = loss.mean()
                    total_loss = loss
                elif args.arch == 1:
                    loss = loss.mean()
                    #total_loss = loss
                    #total_loss = loss + 1 * cmd(model.h[idx_train, :], model.h[idx_test, :])
                    #total_loss = loss + 1 * MMD(logits[idx_train, :], logits[idx_test, :])
                    total_loss = loss + 1 * MMD(model.h[idx_train, :], model.h[idx_test, :])
                elif args.arch == 2:
                    loss = loss.mean()
                    total_loss = loss + 1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
                elif args.arch in [3,4]:
                    loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean()
                    #total_loss = loss
                    total_loss = loss +  1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
                elif args.arch == 5:
                    loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean()
                    total_loss = loss
                    #total_loss = loss + 1 * MMD(logits[idx_train, :], logits[idx_test, :])
                #
            # preds = torch.argmax(logits[idx_train], dim=1).detach()
            if verbose and epoch % 1 == 0:
                
                plot_x.append(epoch)
                plot_y.append(loss.item())
                plot_z.append(cmd(model.h[idx_train, :], model.h[idx_test, :]).item())
            
            if verbose and epoch % 50 == 0:
            
                print("current MMD is {}".format(MMD(logits[idx_train, :], logits[idx_test, :]).detach().cpu().item()))
                print("current CMD is {}".format(cmd(model.h[idx_train, :], model.h[idx_test, :]).detach().cpu().item()))

            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                #EFTC: optional
                if verbose and epoch % 50 == 0:
                    model.eval()
                    logits = model(features)
                    preds_all = torch.argmax(logits, dim=1)
                    acc_val = f1_score(labels[idx_val].cpu(), preds_all[idx_val].cpu(), average='micro')
                    print(epoch, total_loss.item(), loss.item(), acc_val)
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_epoch = epoch
                        torch.save(model.state_dict(), 'best_model_{}.pt'.format(args.dataset))
             
        model.eval()
        embeds = model(features).detach()
        logits = embeds[idx_test]
        preds_all = torch.argmax(embeds, dim=1)
        embeds = embeds.cpu()
        
        micro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro'))
        macro_f1.append(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='macro'))
        
        if verbose:
            print('iteration:')
            print(curr_iter)

    return micro_f1, macro_f1, avg_mmd_dist

if __name__ == '__main__':

    rand_seed = 7
    verbose   = True 

    parser = argparse.ArgumentParser(description='SR-GNN')

    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="verbose")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--gnn-arch", type=str, default='gcn',
                        help="gnn arch of gcn/gat/graphsage")
    parser.add_argument("--SR", type=bool, default=False,
                        help="use shift-robust or not")
    parser.add_argument("--arch", type=int, default=0,
                        help="use which variant of the model")
    parser.add_argument("--biased-sample", type=bool, default=False,
                        help="use biased (non IID) training data")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-out", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 loss")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="print verbose step-wise information")
    parser.add_argument("--n-repeats", type=int, default=20,
                        help=".")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--num-unseen',type=int, default=1)
    parser.add_argument('--metapaths', type=list, default=['PAP'])
    parser.add_argument('--new-classes', type=list, default=[])
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    args = parser.parse_args()
    
    
    #Random seed initialization
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)


    #The number of classes necessary to define the model architecture in the softmax layer
    if args.dataset == 'cora':
        num_class = 7
    elif args.dataset == 'citeseer':
        num_class = 6
    elif args.dataset == 'ppi':
        num_class = 9
    elif args.dataset == 'dblp':
        num_class = 5

    if args.SR and args.gnn_arch == 'ppnp':
        args.arch = 3
    elif args.SR:
        args.arch = 2
    else:
        args.arch = 0
    
    
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    micro_f1, macro_f1, out_acc = main(args, [])

    torch.cuda.empty_cache()
    print(np.mean(in_acc), np.std(in_acc), np.mean(out_acc), np.std(out_acc))
    print("arch {}:".format(args.gnn_arch), np.mean(micro_f1), np.std(micro_f1), np.mean(macro_f1), np.std(macro_f1))
