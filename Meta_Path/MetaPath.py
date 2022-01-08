import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# tags = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
unique_tag = np.array(list(map(chr, range(ord('a'), ord('z') + 1))))
unique_tag = np.append(unique_tag, np.char.add(unique_tag, unique_tag)).tolist()


def node_type_encoder(data: pd.DataFrame):
    node_to_type = dict()
    type_to_node = dict()
    for column_idx, column in enumerate(data.columns):
        type_to_node[unique_tag[column_idx]] = str(column)

    tags = unique_tag[:len(data.columns)]
    tags = pd.DataFrame([tags], columns=data.columns).reindex(index=range(len(data.index)), method='ffill')
    result = tags + data.astype(int).astype(str)

    for column_idx, column in enumerate(result.columns):
        node_to_type[str(column)] = np.unique(result[column])

    return result, node_to_type, type_to_node


def indexer(data: pd.DataFrame):

    for column in tqdm(data.columns):
        for idx, value in enumerate(np.unique(data[column])):
            data.loc[data[column] == value, column] = idx
    return data


def component_ranger(data: pd.DataFrame):
    size = []
    for column in data.columns:
        size.append(len(np.unique(data[column])))
    table = pd.DataFrame([size, data.columns], index=['Size', 'Cols']).transpose()
    order = table.sort_values(by='Size', axis=0, ascending=False)['Cols']
    data = data[order]
    return data


def meta_path_generator(data: pd.DataFrame, attrs_type: list, path_num, type_to_node: dict):
    """
    :param data: log data
    :param attrs_type: list which type of attributes are used. Meta-path-like list is available
    :param path_num: the number of path
    :param type_to_node: decoder for attrs_type, resulting from @node_type_encoder
    :return:
    """
    paths = []
    graphs = []
    attrs = list(map(lambda x: type_to_node[x], attrs_type))

    for attr_idx in range(len(attrs) - 1):
        graph = nx.from_pandas_edgelist(data, source=attrs[attr_idx], target=attrs[attr_idx + 1])
        graphs.append(graph)

    for path_idx in tqdm(range(path_num), desc='Path Generating'):
        path = []
        for element_idx, element_name in enumerate(attrs):
            if element_idx == 0:
                path.append(np.random.choice(data[element_name], 1)[0])
            else:
                path.append(np.random.choice([n for n in graphs[element_idx - 1][path[::-1][0]]], 1)[0])
        paths.append(pd.Series(path).str.replace(r'\D', '').astype(int).tolist())

    paths = np.array(paths)
    return paths


def path_generator(data: pd.DataFrame, path_num):
    """
    Basically, this can be used when the data is "ordered" or walk data.
    No need to specify the order of walk compared to @meta_path_generator
    :param data: (graph) walk data
    :param path_num: the number of path
    :return: list of path
    """
    paths = []
    graphs = []
    attrs = list(data.columns)

    for attr_idx in range(len(attrs) - 1):
        graph = nx.from_pandas_edgelist(data, source=attrs[attr_idx], target=attrs[attr_idx + 1])
        graphs.append(graph)

    for path_idx in tqdm(range(path_num), desc='Path Generating'):
        path = []
        for element_idx, element_name in enumerate(attrs):
            if element_idx == 0:
                path.append(np.random.choice(data[element_name], 1)[0])
            else:
                path.append(np.random.choice([n for n in graphs[element_idx - 1][path[::-1][0]]], 1)[0])
        paths.append(pd.Series(path).str.replace(r'\D', '').astype(int).tolist())

    paths = np.array(paths)
    print(f'Label 1 : {np.sum(paths[:, len(attrs)-1] == 1)} of {path_num}')
    return paths