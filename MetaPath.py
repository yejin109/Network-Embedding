import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

tags = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def meta_path(x: pd.DataFrame, attr_sets, attr_type: str):
    result = dict()
    if attr_type == 'user':
        attr_seed = 'person_rn'
    else:
        attr_seed = 'contents_rn'

    for attr_set in tqdm(attr_sets, desc=attr_type):
        paths = []

        attrs = [attr_seed]
        attrs.extend(attr_set)
        path_name = attrs + attrs[-2::-1]

        data = x[attrs]

        for column_idx, column in enumerate(data.columns):
            tag = tags[column_idx]
            data.loc[:, column] = np.char.add(data[column].values.astype(str), tag)
        graphs = []

        for attr_idx in range(len(attrs) - 1):
            graph = nx.from_pandas_edgelist(data, source=attrs[attr_idx], target=attrs[attr_idx + 1])
            graphs.append(graph)
        graphs = graphs + graphs[::-1]

        for path_idx in range(100):
            path_key = ''
            path = []
            for element_idx, element_name in enumerate(path_name):
                if element_idx == 0:
                    path.append(np.random.choice(data[element_name], 1)[0])
                else:
                    path.append(np.random.choice([n for n in graphs[element_idx - 1][path[::-1][0]]], 1)[0])
                if element_idx != len(path_name) - 1:
                    path_key += f'{element_name}/'
                else:
                    path_key += element_name
            paths.append(pd.Series(path).str.replace(r'\D', '').astype(int).tolist())

        result[path_key] = paths
    return result
