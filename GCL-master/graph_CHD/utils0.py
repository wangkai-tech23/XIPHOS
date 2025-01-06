import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import random

def edge_drop(sp_adj, drop_percent=0.2):
    adj = sp_adj.toarray()
    percent = drop_percent / 2
    n = adj.shape[0]

    edge_num = int((np.count_nonzero(adj) - 0) / 2)  # int(len(row_idx) / 2)  # 87
    add_drop_num = int(edge_num * percent)  # 8
    # print('edge_num', edge_num, 'add_drop_num', add_drop_num)
    # sp_adj = sp.csr_matrix(adj)
    row_idx, col_idx = sp_adj.nonzero()

    # index_list = []
    # for i in range(len(row_idx)):
    #     index_list.append((row_idx[i], col_idx[i]))

    edge_idx = [i for i in range(edge_num)]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,...,87]
    drop_idx = random.sample(edge_idx, add_drop_num)
    # print('drop_idx', drop_idx)
    for i in drop_idx:
        adj[row_idx[i]][col_idx[i]] = 0
        adj[col_idx[i]][row_idx[i]] = 0
    l = [(i, j) for i in range(n) for j in range(i)]
    add_list = random.sample(l, add_drop_num)
    # print('add_list:', add_list)

    for i in add_list:
        adj[i[0]][i[1]] = 1
        adj[i[1]][i[0]] = 1
    return adj
def compute_ppr_IVN(adj, alpha=0.2, self_loop=True):
    a = adj # nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    #
    # b = (1 - alpha) * at
    # c = np.eye(a.shape[0]) - b
    # dd = inv(c)
    # eee = np.dot(c,dd)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    #
    # b = (1 - alpha) * at
    # c = np.eye(a.shape[0]) - b
    # dd = inv(c)
    # eee = np.dot(c,dd)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)