from urllib.request import CacheFTPHandler
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0

def read_triplets(file_name, file_name_valid,file_name_test): # done
    global n_entities, n_relations, n_nodes, n_users, n_items

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    test_triplets = np.loadtxt(file_name_test, dtype=np.int32)
    test_triplets = np.unique(test_triplets,axis=0)
    valid_triplets = np.loadtxt(file_name_valid, dtype=np.int32)
    valid_triplets = np.unique(valid_triplets,axis=0)

    n_users = max(max(can_triplets_np[:,0]), max(test_triplets[:,0]), max(valid_triplets[:,0])) + 1
    n_items = max(max(can_triplets_np[:,2]), max(test_triplets[:,2]), max(valid_triplets[:,2])) + 1
    can_triplets_np[:,2] = can_triplets_np[:,2] + n_users

    if args.inverse_r:
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:,1] = can_triplets_np[:,1]
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        triplets = can_triplets_np.copy()


    n_entities = n_users + n_items
    n_nodes = n_entities
    n_relations = max(max(triplets[:, 1]),max(test_triplets[:, 1]), max(valid_triplets[:,1])) + 1

    
    return triplets

def build_graph(triplets): # done
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("\nBegin to load user-text-item triples ...")
    for h_id, r_id, t_id, label in tqdm(triplets, ascii=True):
        if label == 1:
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            rd[r_id].append([h_id, t_id])

    return ckg_graph, rd



def build_sparse_relational_graph(relation_dict): # done
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        cf = np_mat.copy()
        cf[:,1] = cf[:,1] + n_users # [0, n_items) -> [n_users, n_users+n_items)
        vals = [1.] * len(cf)
        adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]

    return adj_mat_list, norm_mat_list, mean_mat_list

def load_data(model_args):
    global args
    args = model_args

    print('read user-text-item tripltes...')
    if args.dataset == 'data_a':
        triplets = read_triplets('./data_a/triplets_train_new.txt', './data_a/triplets_valid_new.txt','./data_a/triplets_test_new.txt')
    elif args.dataset == 'data_b':
        triplets = read_triplets('./data_b/triplets_train_new.txt', './data_b/triplets_valid_new.txt', './data_b/triplets_test_new.txt')
    else :
        raise NotImplementedError
    print('building the graph ...')
    graph, relation_dict = build_graph(triplets)


    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }

    return n_params, graph