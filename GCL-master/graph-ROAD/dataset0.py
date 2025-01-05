import os
import re
import numpy as np
import networkx as nx
from collections import Counter
from utils0 import compute_ppr, compute_ppr_IVN ,normalize_adj, edge_drop
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import csv
from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import glob
import torch
import torch.nn as nn
import pandas as pd
import time
from sys import getsizeof

def download(dataset):
    basedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(basedir, 'data', dataset)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        url = 'https:/www.chrsmrrs.com/graphkerneldatasets/{}.zip'.format(dataset)
        # url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{0}.zip'.format(dataset)
        zipfile = os.path.basename(url)
        os.system('wget {0}; unzip {1}'.format(url, zipfile))
        os.system('mv {0}/* {1}'.format(dataset, datadir))
        os.system('rm -r {0}'.format(dataset))
        os.system('rm {0}'.format(zipfile))


def process(dataset):
    src = os.path.join(os.path.dirname(__file__), 'data')
    prefix = os.path.join(src, dataset, dataset)

    graph_node_dict = {}
    with open('{0}_graph_indicator.txt'.format(prefix), 'r') as f:
        for idx, line in enumerate(f):
            graph_node_dict[idx + 1] = int(line.strip('\n'))
    max_nodes = Counter(graph_node_dict.values()).most_common(1)[0][1]

    node_labels = []
    if os.path.exists('{0}_node_labels.txt'.format(prefix)):
        with open('{0}_node_labels.txt'.format(prefix), 'r') as f:
            for line in f:
                node_labels += [int(line.strip('\n')) - 1]
            num_unique_node_labels = max(node_labels) + 1
    else:
        print('No node labels')

    node_attrs = []
    if os.path.exists('{0}_node_attributes.txt'.format(prefix)):
        with open('{0}_node_attributes.txt'.format(prefix), 'r') as f:
            for line in f:
                node_attrs.append(
                    np.array([float(attr) for attr in re.split("[,\s]+", line.strip("\s\n")) if attr], dtype=np.float)
                )
    else:
        print('No node attributes')

    graph_labels = []
    unique_labels = set()
    with open('{0}_graph_labels.txt'.format(prefix), 'r') as f:
        for line in f:
            val = int(line.strip('\n'))
            if val not in unique_labels:
                unique_labels.add(val)
            graph_labels.append(val)
    label_idx_dict = {val: idx for idx, val in enumerate(unique_labels)}
    graph_labels = np.array([label_idx_dict[l] for l in graph_labels])

    adj_list = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    index_graph = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    with open('{0}_A.txt'.format(prefix), 'r') as f:
        for line in f:
            u, v = tuple(map(int, line.strip('\n').split(',')))
            adj_list[graph_node_dict[u]].append((u, v))
            index_graph[graph_node_dict[u]] += [u, v]

    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs, pprs = [], []
    for idx in range(1, 1 + len(adj_list)):
        graph = nx.from_edgelist(adj_list[idx])
        if max_nodes is not None and graph.number_of_nodes() > max_nodes:
            continue

        graph.graph['label'] = graph_labels[idx - 1]
        for u in graph.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                graph.nodes[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                graph.nodes[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            graph.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        for node_idx, node in enumerate(graph.nodes()):
            mapping[node] = node_idx

        graphs.append(nx.relabel_nodes(graph, mapping))
        a = compute_ppr(graph, alpha=0.2)
        pprs.append(a) #compute_ppr(graph, alpha=0.2)

    if 'feat_dim' in graphs[0].graph:
        pass
    else:
        max_deg = max([max(dict(graph.degree).values()) for graph in graphs])
        for graph in graphs:
            for u in graph.nodes(data=True):
                f = np.zeros(max_deg + 1)
                f[graph.degree[u[0]]] = 1.0
                if 'label' in u[1]:
                    # b = u[1]['label']
                     # dtype=np.float
                    f = np.concatenate((np.array(u[1]['label']).astype(float) , f))
                graph.nodes[u[0]]['feat'] = f
    return graphs, pprs


def load(dataset):
    basedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(basedir, 'data', dataset)

    if not os.path.exists(datadir):
        # download(dataset)
        print('here0')
        graphs, diff = process(dataset)
        feat, adj, labels = [], [], []
        print('here1')
        for idx, graph in enumerate(graphs):
            # a = nx.to_numpy_array(graph)
            # print(a)
            adj.append(nx.to_numpy_array(graph))
            labels.append(graph.graph['label'])
            feat.append(np.array(list(nx.get_node_attributes(graph, 'feat').values())))

        adj, diff, feat, labels = np.array(adj, dtype=object), np.array(diff, dtype=object), np.array(feat, dtype=object), np.array(labels, dtype=object)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)

    else:
        adj = np.load(f'{datadir}/adj.npy', allow_pickle=True)
        diff = np.load(f'{datadir}/diff.npy', allow_pickle=True)
        feat = np.load(f'{datadir}/feat.npy', allow_pickle=True)
        labels = np.load(f'{datadir}/labels.npy', allow_pickle=True)

    max_nodes = max([a.shape[0] for a in adj])
    feat_dim = feat[0].shape[-1]

    num_nodes = [] #num_nodes按照下面计算得来，diff也会经过一些处理

    for idx in range(adj.shape[0]):

        num_nodes.append(adj[idx].shape[-1])

        adj[idx] = normalize_adj(adj[idx]).todense()

        diff[idx] = np.hstack(
            (np.vstack((diff[idx], np.zeros((max_nodes - diff[idx].shape[0], diff[idx].shape[0])))),
             np.zeros((max_nodes, max_nodes - diff[idx].shape[1]))))

        adj[idx] = np.hstack(
            (np.vstack((adj[idx], np.zeros((max_nodes - adj[idx].shape[0], adj[idx].shape[0])))),
             np.zeros((max_nodes, max_nodes - adj[idx].shape[1]))))

        feat[idx] = np.vstack((feat[idx], np.zeros((max_nodes - feat[idx].shape[0], feat_dim))))

    adj = np.array(adj.tolist()).reshape(-1, max_nodes, max_nodes)
    diff = np.array(diff.tolist()).reshape(-1, max_nodes, max_nodes)
    feat = np.array(feat.tolist()).reshape(-1, max_nodes, feat_dim)

    return adj, diff, feat, labels, num_nodes
def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # r_inv = np.power (rowsum, -1).flatten ()
    r_inv = np.power(rowsum, -0.5).flatten()  # 拉普拉斯对称归一化
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)  # 拉普拉斯对称归一化
    return mx

method = 2
# batch_size = 40
nnodes = 400 # 50 #
edge_drop_percent = 0.1

def load_normal_data(dataset_str,tag = 'train'): # 0,1,2,3,4
    print('Start Loading data')
    # tag = 'test' #'train'
    feat=0;labels=0#初始化
    time_start = time.time()

    dataset_str = 0
    node_path = '../../dataset-IVN/CHD/CHD_Split/Dataset50_40/' + tag + '/nodes/vectors' + str(dataset_str) + '.csv'

    idx_features_labels = np.genfromtxt(node_path, delimiter=',', dtype=np.dtype(np.float32))
    features = torch.FloatTensor(idx_features_labels[:, :-1])
    label = torch.tensor(idx_features_labels[:, -1])  # ,dtype = torch.int32
    features = torch.reshape(features, (-1, nnodes, features.shape[1]))
    label = torch.reshape(label, (-1, nnodes))

    feat = features
    labels = label




    time_end = time.time()
    print('Load over nodes of {} data-IVN, with {} graphs and time consuming {}'.format(tag,feat.shape[0],time_end - time_start))
    adjes = [];
    diffes = []

    edge_path = '../../dataset-IVN/CHD/CHD_Split/Dataset50_40/' + tag + '/edges/' + str(dataset_str) + '/'

    ''' train/test_count: the number of files in each folder '''
    files = os.listdir(edge_path)  # Read the folder
    num_png = len(files)  # Count the number of files in a folder

    for i in range(1, num_png + 1):
        # if i == 1 + 32 * 500:
        #     break
        # print('i',i)
        if tag == 'test':
            edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i - 1)), delimiter=',', dtype=np.int32)
        else:
            edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i)), delimiter=',', dtype=np.int32)

        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=(nnodes, nnodes), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # build symmetric adjacency matrix

        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # # adj = sparse_mx_to_torch_sparse_tensor(adj)
        # diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)
        #
        # adjes.append(adj.toarray())
        # diffes.append(diff)

        if method == 1:  # diff边缘掉落，adj=adj
            diff = edge_drop(adj, drop_percent=edge_drop_percent)
            diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))
            adj = normalize(adj + sp.eye(adj.shape[0]))

            adjes.append(adj.toarray())
            diffes.append(diff.toarray())
        elif method == 0:  # diff扩散，adj边缘掉落
            adj = edge_drop(adj, drop_percent=edge_drop_percent)
            adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
            diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)

            adjes.append(adj.toarray())
            diffes.append(diff)
        elif method == 2:  # method = 2 # A,diff 都是随机掉落
            diff = edge_drop(adj, drop_percent=edge_drop_percent)
            diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))

            adj = edge_drop(adj, drop_percent=edge_drop_percent)
            adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))

            adjes.append(adj.toarray())
            diffes.append(diff.toarray())
        else:
            adj = normalize(adj + sp.eye(adj.shape[0]))

            adjes.append(adj.toarray())
            diffes.append(adj.toarray())

    num_nodes = []
    time_end = time.time()
    print('Load over data-IVN, with {} graphs and time consuming {}'.format(len(adjes),time_end - time_start))


    return adjes, diffes, feat, labels, num_nodes



def load_data(dataset_str,tag = 'train'): # 0,1,2,3,4
    print('Start Loading data')
    # tag = 'test' #'train'
    feat=0;labels=0#初始化
    time_start = time.time()

    for dataset_str in [0,1,2,3,4,5]:
        node_path = '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/nodes/vector_' + str(dataset_str) + '.csv'
        # # edge_path = '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/edges/'+ str(dataset_str) + '/'
        ##### path = '../ROAD_split/Dataset' + str(nnodes) + '/' + Slice[div] + '/edges/'
        # node_path = '../../dataset-IVN/CHD/CHD_Split/Dataset50_40/'+tag+'/nodes/' + str(dataset_str) + '/'
        # files = glob.glob(r'{}*.csv'.format(node_path))  # Read the folder
        # num_png = len(files)  # Count the number of files in a folder
        # # print('文件个数：', num_png)
        # idx_features_labels = 0
        # for i in range(1,num_png+1):
        #     if tag == 'test':
        #         load_node = np.genfromtxt("{}{}.csv".format(node_path, str(i-1)), delimiter=',', dtype=np.float32)
        #     else:
        #         load_node = np.genfromtxt("{}{}.csv".format(node_path, str(i)), delimiter=',', dtype=np.float32)
        #
        #     if i == 1:
        #         idx_features_labels = load_node
        #     else:
        #         idx_features_labels = np.concatenate([idx_features_labels,load_node],0)

        idx_features_labels = np.genfromtxt(node_path, delimiter=',', dtype=np.dtype(np.float32))
        features = torch.FloatTensor(idx_features_labels[1:, :-1])
        label = torch.tensor(idx_features_labels[1:, -1])  # ,dtype = torch.int32
        features = torch.reshape(features, (-1, nnodes, features.shape[1]))
        label = torch.reshape(label, (-1, nnodes))
        if dataset_str == 0:
            feat = features
            labels = label
        else:
            feat = torch.cat([feat,features],0)
            labels = torch.cat([labels, label], 0)
    # features = normalize (features)

    time_end = time.time()
    print('Load over nodes of {} data-IVN, with {} graphs and time consuming {}'.format(tag,feat.shape[0],time_end - time_start))
    adjes = [];
    diffes = []
    for dataset_str in [0,1,2,3,4,5]:
        # # node_path = '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/nodes/vector_' + str(dataset_str) + '.csv'
        edge_path ='../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/edges/'+ str(dataset_str) + '/'


        ''' train/test_count: the number of files in each folder '''
        files = os.listdir(edge_path)  # Read the folder
        num_png = len(files)  # Count the number of files in a folder
        print('num_png', num_png)
        for i in range(1, num_png+1):
            # if i == 1 + 32 * 500:
            #     break

            if tag == 'test':
                edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i)), delimiter=',', dtype=np.int32)
            else:
                edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i)), delimiter=',', dtype=np.int32)

            adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),shape=(nnodes, nnodes), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # build symmetric adjacency matrix

            # adj = normalize(adj + sp.eye(adj.shape[0]))
            # # adj = sparse_mx_to_torch_sparse_tensor(adj)
            # diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)
            #
            # adjes.append(adj.toarray())
            # diffes.append(diff)
            if method == 1:  # diff边缘掉落，adj=adj
                diff = edge_drop(adj, drop_percent=edge_drop_percent)
                diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))
                adj = normalize(adj + sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(diff.toarray())
            elif method == 0:  # diff扩散，adj边缘掉落
                adj = edge_drop(adj, drop_percent=edge_drop_percent)
                adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
                diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)

                adjes.append(adj.toarray())
                diffes.append(diff)
            elif method == 2:  # method = 2 # A,diff 都是随机掉落
                diff = edge_drop(adj, drop_percent=edge_drop_percent)
                diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))

                adj = edge_drop(adj, drop_percent=edge_drop_percent)
                adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(diff.toarray())
            else:
                # adj = edge_drop(adj, drop_percent=0.2)
                adj = normalize(adj + sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(adj.toarray())

    num_nodes = []
    # return adj, diff, feat, labels, num_nodes
    time_end = time.time()
    print('Load over data-IVN, with {} graphs and time consuming {}'.format(len(adjes),time_end - time_start))
    # myexam: Load over data-IVN, with time consuming 132.79820036888123
    # py310: Load over data-IVN, with time consuming 167.32215857505798

    return adjes, diffes, feat, labels, num_nodes

def load_extra_data(dataset_str,tag = 'train'): # 0,1,2,3,4
    print('Start Loading data')
    # tag = 'test' #'train'
    feat=0;labels=0#初始化
    time_start = time.time()

    for dataset_str in [1,2,4,5]:

        #####       '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/nodes/vector_' + str(dataset_str) + '.csv'
        node_path = '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/'+tag+'/nodes/0_' + str(dataset_str) + '/'
        files = glob.glob(r'{}*.csv'.format(node_path))  # Read the folder
        num_png = len(files)  # Count the number of files in a folder
        print('文件个数：', num_png)
        idx_features_labels = 0
        for i in range(1,num_png+1):
            if tag == 'test':
                load_node = np.genfromtxt("{}{}.csv".format(node_path, str(i)), delimiter=',', dtype=np.float32)
            else:
                load_node = np.genfromtxt("{}{}.csv".format(node_path, str(i)), delimiter=',', dtype=np.float32)

            if i == 1:
                idx_features_labels = load_node
            else:
                idx_features_labels = np.concatenate([idx_features_labels,load_node],0)

        # idx_features_labels = np.genfromtxt(node_path, delimiter=',', dtype=np.dtype(np.float32))
        features = torch.FloatTensor(idx_features_labels[:, :-1])
        label = torch.tensor(idx_features_labels[:, -1])  # ,dtype = torch.int32
        features = torch.reshape(features, (-1, nnodes, features.shape[1]))
        label = torch.reshape(label, (-1, nnodes))
        if dataset_str == 1:
            feat = features
            labels = label
        else:
            feat = torch.cat([feat,features],0)
            labels = torch.cat([labels, label], 0)
    # features = normalize (features)

    time_end = time.time()
    print('Load over nodes of {} data-IVN, with {} graphs and time consuming {}'.format(tag,feat.shape[0],time_end - time_start))
    adjes = [];
    diffes = []
    for dataset_str in [1,2,4,5]:
        # node_path = '../../dataset-IVN/CHD/CHD_Split/Dataset50_40/'+tag+'/nodes/0_/vectors' + str(dataset_str) + '.csv'
        edge_path = '../../dataset-IVN/ROAD_mydata/ROAD_split/Dataset' + str(nnodes) + '/' + tag + '/edges/0_' + str(dataset_str) + '/'

        ''' train/test_count: the number of files in each folder '''
        files = os.listdir(edge_path)  # Read the folder
        num_png = len(files)  # Count the number of files in a folder

        for i in range(1, num_png+1):
            # if i == 1 + 32 * 500:
            #     break
                # print('i',i)
            if tag == 'test':
                edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i)), delimiter=',', dtype=np.int32)
            else:
                edges = np.genfromtxt("{}{}.csv".format(edge_path, str(i)), delimiter=',', dtype=np.int32)

            adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),shape=(nnodes, nnodes), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # build symmetric adjacency matrix
            # adj = normalize(adj + sp.eye(adj.shape[0]))
            # # adj = sparse_mx_to_torch_sparse_tensor(adj)
            # diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)
            #
            # adjes.append(adj.toarray())
            # diffes.append(diff)
            if method == 1:  # diff边缘掉落，adj=adj
                diff = edge_drop(adj, drop_percent=edge_drop_percent)
                diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))
                adj = normalize(adj + sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(diff.toarray())
            elif method == 0:   # diff扩散，adj边缘掉落
                adj = edge_drop(adj, drop_percent=edge_drop_percent)
                adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
                diff = compute_ppr_IVN(adj.toarray(), alpha=0.2)

                adjes.append(adj.toarray())
                diffes.append(diff)
            elif method == 2: #method = 2 # A,diff 都是随机掉落
                diff = edge_drop(adj, drop_percent=edge_drop_percent)
                diff = normalize(sp.csr_matrix(diff) + sp.eye(diff.shape[0]))

                adj = edge_drop(adj, drop_percent=edge_drop_percent)
                adj = normalize(sp.csr_matrix(adj)+ sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(diff.toarray())
            else:
                adj = normalize(adj+ sp.eye(adj.shape[0]))

                adjes.append(adj.toarray())
                diffes.append(adj.toarray())

    num_nodes = []
    # return adj, diff, feat, labels, num_nodes
    time_end = time.time()
    print('Load over data-IVN, with {} graphs and time consuming {}'.format(len(adjes),time_end - time_start))
    # myexam: Load over data-IVN, with time consuming 132.79820036888123
    # py310: Load over data-IVN, with time consuming 167.32215857505798

    return adjes, diffes, feat, labels, num_nodes



if __name__ == '__main__':
    # MUTAG, PTC_MR, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K,
    adj, diff, feat, labels = load('PTC_MR')
    print('done')

