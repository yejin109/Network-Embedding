import time
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset

import JOB.config as config

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
unique_tag = np.array(list(map(chr, range(ord('a'), ord('z') + 1))))
unique_tag = np.append(unique_tag, np.char.add(unique_tag, unique_tag)).tolist()


def indexer(data: pd.DataFrame):
    attrs_size = dict()
    for column in tqdm(data.columns, desc='Indexing'):
        for idx, value in enumerate(np.unique(data[column])):
            data.loc[data[column] == value, column] = idx
        attrs_size[str(column)] = len(np.unique(data[column]))
    return data, attrs_size


def node_type_encoder(data: pd.DataFrame):
    start = time.time()
    node_to_type = dict()
    type_to_node = dict()
    for column_idx, column in enumerate(data.columns):
        type_to_node[unique_tag[column_idx]] = str(column)

    tags = unique_tag[:len(data.columns)]
    tags = pd.DataFrame([tags], columns=data.columns).reindex(index=range(len(data.index)), method='ffill')
    result = tags + data.astype(int).astype(str)

    for column_idx, column in enumerate(result.columns):
        node_to_type[str(column)] = np.unique(result[column])
    print(f'Node Type Encoding : {time.time()-start: .3f}s')
    return result, node_to_type, type_to_node


def graph_builder(data: pd.DataFrame):
    large_graph = nx.Graph()
    for attr_edge in tqdm(config.attr_edges, desc='Graph Building'):
        for i, j in itertools.combinations(attr_edge, 2):
            graph = nx.from_pandas_edgelist(data, source=i, target=j)
            large_graph.add_edges_from(graph.edges)
    return large_graph


def adj_mat_processor(graph):
    adj_mat = nx.adjacency_matrix(graph)
    adj_mat = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
    adj_mat = mat_normalizer(adj_mat)
    adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
    return adj_mat


def mat_normalizer(mat):
    row_sum = np.array(mat.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mat = r_mat_inv.dot(mat)
    return mat


def input_processor(data: pd.Series, node: list):
    gcn_input = np.where(np.isin(node, data), 1, 0)
    gcn_input = torch.LongTensor(gcn_input).to(device)
    return gcn_input


def splitter(data: pd.DataFrame):
    train_x, val_x, train_y, val_y = train_test_split(data.iloc[:, :-1], data['target'], shuffle=True)
    train_x.reset_index(drop=True, inplace=True)
    val_x.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)
    val_y.reset_index(drop=True, inplace=True)
    return train_x, val_x, train_y, val_y


def sparse_mx_to_torch_sparse_tensor(sparse_mat):
    sparse_mat = sparse_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mat.row, sparse_mat.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mat.data)
    shape = torch.Size(sparse_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


class DataContainer(Dataset):
    def __init__(self, data: pd.DataFrame, target, sample_size, nodes: list):
        super(DataContainer, self).__init__()
        samples = np.random.choice(np.arange(len(data)), sample_size)
        self.data = data.iloc[samples, :]
        self.target = torch.Tensor(target[samples].str.replace('\D', '').astype(int).to_numpy()).to(device)
        self.nodes = nodes

    def __getitem__(self, index):
        item = input_processor(self.data.iloc[index, :], self.nodes)
        return item, self.target[index]

    def __len__(self):
        return self.target.size(0)
