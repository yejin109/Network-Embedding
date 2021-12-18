import numpy as np


class Optimizer:
    def __init__(self, optimizer_type, input_layer_size, hidden_layer_size, output_layer_size, batch_size, learning_rate):
        self.il = input_layer_size
        self.hl = hidden_layer_size
        self.ol = output_layer_size
        self.bs = batch_size
        self.lr = learning_rate

        self.opt = optimizer_type
        self.epsilon = 1e-8

        if optimizer_type == 'Adagrad':
            self.G_ki = np.zeros((self.il, self.hl))
            self.G_ij = np.zeros((self.hl, self.ol))
        elif optimizer_type == 'Adam':
            self.m_ki = np.zeros((self.il, self.hl))
            self.v_ki = np.zeros((self.il, self.hl))

            self.m_ij = np.zeros((self.hl, self.ol))
            self.v_ij = np.zeros((self.hl, self.ol))

            self.b_1 = 0.9
            self.b_2 = 0.999

    def step(self, w_ij, w_ki, E_wij, E_wki):
        if self.opt == 'SGD':
            w_ij -= self.lr * np.sum(E_wij, axis=0) / self.bs
            w_ki -= self.lr * np.sum(E_wki, axis=0) / self.bs
        elif self.opt == 'Adagrad':
            self.G_ki += np.sum(E_wki ** 2, axis=0) / self.bs
            self.G_ij += np.sum(E_wij ** 2, axis=0) / self.bs
            w_ki -= self.lr / np.sqrt(self.G_ki + self.epsilon) * np.sum(E_wki, axis=0) / self.bs
            w_ij -= self.lr / np.sqrt(self.G_ij + self.epsilon) * np.sum(E_wij, axis=0) / self.bs
        elif self.opt == 'Adam':
            self.m_ki = self.b_1 * self.m_ki + (1 - self.b_1) * np.sum(E_wki, axis=0) / self.bs
            self.v_ki = self.b_2 * self.v_ki + (1 - self.b_2) * np.sum(E_wki ** 2, axis=0) / self.bs
            w_ki -= self.lr / (np.sqrt(self.v_ki / (1 - self.b_2)) + self.epsilon) * self.m_ki / (1 - self.b_1)

            self.m_ij = self.b_1 * self.m_ij + (1 - self.b_1) * np.sum(E_wij, axis=0) / self.bs
            self.v_ij = self.b_2 * self.v_ij + (1 - self.b_2) * np.sum(E_wij ** 2, axis=0) / self.bs
            w_ij -= self.lr / (np.sqrt(self.v_ij / (1 - self.b_2)) + self.epsilon) * self.m_ij / (1 - self.b_1)
        return w_ij, w_ki
