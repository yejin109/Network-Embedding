import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from JOB import config
from JOB.src.GCN import GCN
from JOB.utils.util_GCN import indexer, node_type_encoder, graph_builder, adj_mat_processor, DataContainer, splitter

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

raw_train = pd.read_csv('data/train.csv')
raw_train = raw_train.drop(config.drop_attrs, axis=1)

train, attrs_size = indexer(raw_train.copy())
train, node_to_type, type_to_node = node_type_encoder(train)

graph = graph_builder(train)
adj_mat = adj_mat_processor(graph)

node_num = len(graph.nodes)
embedding_dim = 128
hidden_dim = 16
output_dim = 1
learning_rate = 1e-3

epochs = 30
batch_size = 1
train_sample_size = 600
val_sample_size = 50

train_x, val_x, train_y, val_y = splitter(train)

model = GCN(node_num, embedding_dim, hidden_dim, output_dim).to(device)
criteria = nn.MSELoss().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

train_loss_per_epoch = []
train_score_per_epoch = []
val_loss_per_epoch = []
val_score_per_epoch = []
for epoch in tqdm(range(epochs), desc='Epoch'):
    model.train()
    train_dataset = DataContainer(train_x, train_y, train_sample_size, list(graph.nodes))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss_per_iter = []
    train_pred_per_iter = []
    train_true_per_iter = []

    for sequence, label in train_data_loader:
        sequence = torch.squeeze(sequence)
        optimizer.zero_grad()
        output = model(sequence, adj_mat)
        train_loss = criteria(output, label)
        train_loss.backward()
        optimizer.step()
        train_loss_per_iter.append(train_loss.detach().cpu().numpy())
        pred = np.around(output.detach().cpu().numpy()).sum()/28
        train_pred_per_iter.append(np.where(pred > 0.5, 1, 0))
        train_true_per_iter.append(label.detach().cpu().numpy())

    train_loss_per_epoch.append(np.mean(train_loss_per_iter))
    train_score_per_epoch.append(f1_score(train_true_per_iter, train_pred_per_iter))

    model.eval()
    with torch.no_grad():
        val_dataset = DataContainer(val_x, val_y, val_sample_size, list(graph.nodes))
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        val_loss_per_iter = []
        val_pred_per_iter = []
        val_true_per_iter = []

        for sequence, label in val_data_loader:
            sequence = torch.squeeze(sequence)
            output = model(sequence, adj_mat)
            val_loss = criteria(output, label)
            val_loss_per_iter.append(val_loss.detach().cpu().numpy())
            pred = np.around(output.detach().cpu().numpy()).sum() / 28
            val_pred_per_iter.append(np.where(pred > 0.5, 1, 0))
            val_true_per_iter.append(label.detach().cpu().numpy())

        val_loss_per_epoch.append(np.mean(val_loss_per_iter))
        val_score_per_epoch.append(f1_score(val_true_per_iter, val_pred_per_iter))

    if epoch % 5 == 0:
        print(f' Train Loss : {np.mean(train_loss_per_iter): .3f} /'
              f' Train Score : {f1_score(train_true_per_iter, train_pred_per_iter): .3f} /'
              f' Val Loss : {np.mean(val_loss_per_iter): .3f} /'
              f' Val Score : {f1_score(val_true_per_iter, val_pred_per_iter): .3f}')


plt.figure()
plt.plot(train_score_per_epoch, label='Train')
plt.plot(val_score_per_epoch, label='Val')
plt.title('F1-SCORE')
plt.legend()
# plt.savefig(f'result/submit/{config.version}_score.png')
plt.show()

plt.figure()
plt.plot(train_loss_per_epoch, label='Train')
plt.plot(val_loss_per_epoch, label='Val')
plt.title('Loss')
plt.legend()
# plt.savefig(f'result/submit/{config.version}_loss.png')
plt.show()

raw_test = pd.read_csv('data/test.csv')
raw_test = raw_test.drop(config.drop_attrs, axis=1)

test, attrs_size = indexer(raw_test.copy())
dummy = pd.Series(np.ones(len(test)).astype(str))

model.eval()
preds = []
with torch.no_grad():
    test_dataset = DataContainer(test, dummy, len(test), list(graph.nodes))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for sequence, _ in tqdm(test_data_loader):
        sequence = torch.squeeze(sequence)
        output = model(sequence, adj_mat)
        pred = np.around(output.detach().cpu().numpy()).sum() / 28
        preds.append(np.where(pred > 0.5, 1, 0))

submission = pd.read_csv('data/sample_submission.csv')
preds = np.array(preds)
submission['target'] = preds

submission.to_csv(f'result/submit/GCN_V1.csv', index=False)
print()
