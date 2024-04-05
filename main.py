'''
Author: Fasil Cheema
Purpose: This module is the main point of execution for this
          repository.
          This code is based/inspired off the paper and repo SRGNN:
          (Zhu, Qi, et al. "Shift-robust gnns: Overcoming the ...
          ... limitations of localized graph training data." ...
          ...  Advances in Neural Information Processing
          ...   Systems 34 (2021): 27965-27977.)
'''

import dgl
import utils
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import argparse, pickle
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
from cvxopt import matrix, solvers
from collections import defaultdict
from sklearn.metrics import f1_score
from models import GCN, GraphSAGE, PPRPowerIteration, SGC, GAT
from matrixtools import compute_acc, cmd, l2diff, moment_diff, cross_entropy, pairwise_distances, MMD, KMM, calc_feat_smooth, calc_emb_smooth, calc_A_hat


def main(args, new_classes):

    max_train = 20
    verbose = args.verbose
    device = torch.device("cpu")
    unk = False
    cnt_wait = 0
    best = 1e9
    best_t = 0

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = utils.data_loader(args.dataset)
        labels = [np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]
        features = torch.FloatTensor(utils.preprocess_features(features))
    
        min_max_scaler = preprocessing.MinMaxScaler()
        features = F.normalize(features, p=1,dim=1)
        feat_smooth_matrix = calc_feat_smooth(adj, feat)
        
        nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
        
        g = dgl.from_networkx(nx_g)
    else:
        raise ValueError("wrong dataset name")
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    
    labels = torch.LongTensor(labels)
    
    nb_classes = max(labels).item() + 1

    cross_ent_x = nn.CrossEntropyLoss(reduction='none')

    print('number of classes {}'.format(nb_classes))
    
    best_val_acc = 0
    cnt_wait = 0
    finetune = False
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []
    
    
    num_seeds = []
    all_runs_data = defaultdict(list)
    feature_smoothness = []
    embedding_smoothness = []
    avg_dist, max_dist = [], []

    if not(args.pretrain == 'none'):
        train_dump = pickle.load(open(args.pretrain))
    else:
        train_dump = pickle.load(open('intermediate/{}_dump.p'.format(args.dataset), 'rb'))
    
    ppr_vector = train_dump['ppr_vector']
    ppr_dist = train_dump['ppr_dist']
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

            #uses personalized page rank to generate the biased data 
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
        # can alternatively also implement F.relu
        if args.gnn == 'graphsage':
            model = GraphSAGE(g,
                    ft_size,
                    args.n_hidden,
                    nb_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout,
                    args.aggregator_type
                    )
        elif args.gnn == 'gat':
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
        elif args.gnn == 'ppnp':
            model = PPRPowerIteration(ft_size, args.n_hidden, nb_classes, adj, alpha=0.1, niter=10, drop_prob=args.dropout)
        elif args.gnn == 'sgc':
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
            
    
            logits = model(features)
            loss = cross_ent_x(logits[idx_train], labels[idx_train])
                
            if args.arch == 0:
                loss = loss.mean()
                total_loss = loss
            elif args.arch == 1:
                loss = loss.mean()
                total_loss = loss + 1 * MMD(model.h[idx_train, :], model.h[idx_test, :])
            elif args.arch == 2:
                loss = loss.mean()
                total_loss = loss + 1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
            elif args.arch in [3,4]:
                loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean()
                total_loss = loss +  1 * cmd(model.h[idx_train, :], model.h[iid_train, :])
            elif args.arch == 5:
                loss = (torch.Tensor(kmm_weight).reshape(-1) * (loss)).mean()
                total_loss = loss
            
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

    parser = argparse.ArgumentParser(description='SR-GNN')

    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="verbose")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--gnn", type=str, default='gcn',
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
    parser.add_argument("--n-repeats", type=int, default=20,
                        help=".")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--dataset',type=str, default='cora')
    parser.add_argument('--new-classes', type=list, default=[])
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    parser.add_argument('--pretrain',type=str, default='none')
    
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

    if args.SR and args.gnn == 'ppnp':
        args.arch = 3
    elif args.SR:
        args.arch = 2
    else:
        args.arch = 0
    
    #Note in_acc is ignored
    in_acc, out_acc, micro_f1, macro_f1 = [], [], [], []

    micro_f1, macro_f1, out_acc = main(args, [])

    torch.cuda.empty_cache()

    print(np.mean(in_acc), np.std(in_acc), np.mean(out_acc), np.std(out_acc))
    print("arch {}:".format(args.gnn), np.mean(micro_f1), np.std(micro_f1), np.mean(macro_f1), np.std(macro_f1))


    #This plot is meaningless
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(X, Y, Z, c='red')
    plt.show()
    '''