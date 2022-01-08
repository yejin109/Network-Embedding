import pandas as pd

from JOB import config
from JOB.utils.MetaPath import node_type_encoder, indexer, component_ranger, meta_path_generator

raw_train = pd.read_csv('../data/train.csv')
raw_train = raw_train.drop(config.drop_attrs, axis=1)

train = indexer(raw_train.copy())

train = component_ranger(train)

train, node_to_type, type_to_node = node_type_encoder(train)

train_y = train['target']
train = pd.concat((train.drop('target', axis=1), train_y), axis=1)

meta_paths = meta_path_generator(train, ['a', 'b', 'a'], 100, type_to_node)
attributes = list(train.columns)
print()
