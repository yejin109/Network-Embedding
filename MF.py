import numpy as np


class MF:
    def __init__(self, embedding_size, nodes_num, learning_rate, adjacency_matrix):
        self.d = embedding_size
        self.n = nodes_num
        self.lr = learning_rate
        self.adj_mat = adjacency_matrix
        self.theta_r = self.adj_mat != 0

        # Embedding Initialization
        np.random.seed(7)
        self.U = np.random.rand(self.d, self.n)
        np.random.seed(7)
        self.V = np.random.rand(self.d, self.n)

    def iterating(self, index: tuple):
        i = index[0]
        j = index[1]
        u_i = self.U[:, i].copy()
        v_j = self.V[:, j].copy()

        error = np.transpose(u_i) @ v_j - self.adj_mat[i, j]
        loss = error ** 2

        gradient = error
        self.U[:, i] = u_i - self.lr * (gradient * v_j)
        self.V[:, j] = v_j - self.lr * (gradient * u_i)
        return loss
