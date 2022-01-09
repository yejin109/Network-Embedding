import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, node_num, embedding_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.emb = nn.Embedding(node_num, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x, adj_mat):
        embeddings = self.emb(x)
        output = self.fc1(torch.spmm(adj_mat, embeddings))
        output = self.relu(output)
        output = self.fc2(torch.spmm(adj_mat, output)).squeeze()
        output = torch.mul(output, x)
        return output
