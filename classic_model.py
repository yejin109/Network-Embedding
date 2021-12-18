import numpy as np


class NeuralNet:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, batch_size,
                 activation_type='sigmoid', optimizer_type='SGD'):
        self.il = input_layer_size
        self.hl = hidden_layer_size
        self.ol = output_layer_size
        self.lr = learning_rate
        self.bs = batch_size
        self.loss_per_iter = []
        self.validation_per_iter = []
        self.epsilon = 1e-8
        self.opt = optimizer_type

        # Weight Initialization
        np.random.seed(146823)
        self.w_ki = np.random.rand(self.il, self.hl)
        self.w_ki = np.tile(self.w_ki, [self.bs, 1, 1])

        np.random.seed(784290)
        self.w_ij = np.random.rand(self.hl, self.ol)
        self.w_ij = np.tile(self.w_ij, [self.bs, 1, 1])

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

        if activation_type == 'sigmoid':
            self.activation = lambda u: 1/(1+np.exp(-u))

    def propagating(self, batch, label, is_update=True):
        # Input Later
        x = batch.reshape((batch.shape[1], batch.shape[0], 1))
        # Hidden Layer
        h_i = self.activation(np.transpose(self.w_ki, [0, 2, 1]) @ x)

        # Output Layer
        y_i = self.activation(np.transpose(self.w_ij, [0, 2, 1]) @ h_i)

        # Error function Partial Derivatives
        E_yj = (y_i - label)
        E_uj = E_yj * y_i * (1 - y_i)
        E_wij = E_uj * h_i
        E_hi = E_uj * self.w_ij
        E_ui = E_hi * h_i * (1 - h_i)
        E_wki = x @ np.transpose(E_ui, [0, 2, 1])
        if is_update:
            if self.opt == 'SGD':
                self.w_ij -= self.lr * np.sum(E_wij, axis=0) / self.bs
                self.w_ki -= self.lr * np.sum(E_wki, axis=0) / self.bs
            elif self.opt == 'Adagrad':
                self.G_ki += np.sum(E_wki**2, axis=0) / self.bs
                self.G_ij += np.sum(E_wij**2, axis=0) / self.bs
                self.w_ki -= self.lr / np.sqrt(self.G_ki + self.epsilon) * np.sum(E_wki, axis=0) / self.bs
                self.w_ij -= self.lr / np.sqrt(self.G_ij + self.epsilon) * np.sum(E_wij, axis=0) / self.bs
            elif self.opt == 'Adam':
                self.m_ki = self.b_1 * self.m_ki + (1 - self.b_1) * np.sum(E_wki, axis=0) / self.bs
                self.v_ki = self.b_2 * self.v_ki + (1 - self.b_2) * np.sum(E_wki**2, axis=0) / self.bs
                self.w_ki -= self.lr / (np.sqrt(self.v_ki/(1-self.b_2)) + self.epsilon) * self.m_ki/(1-self.b_1)

                self.m_ij = self.b_1 * self.m_ij + (1 - self.b_1) * np.sum(E_wij, axis=0) / self.bs
                self.v_ij = self.b_2 * self.v_ij + (1 - self.b_2) * np.sum(E_wij ** 2, axis=0) / self.bs
                self.w_ij -= self.lr / (np.sqrt(self.v_ij/(1-self.b_2)) + self.epsilon) * self.m_ij/(1-self.b_1)

            self.loss_per_iter.append(E_yj**2/2)
        else:
            self.validation_per_iter.append(E_yj**2/2)


class NetworkEmbedding:
    def __init__(self, embedding_size, nodes_num, learning_rate, adjacency_matrix):
        self.d = embedding_size
        self.n = nodes_num
        self.lr = learning_rate
        self.adj_mat = adjacency_matrix
        self.theta_r = self.adj_mat != 0

        # Embedding Initialization
        np.random.seed(146823)
        self.U = np.random.rand(self.d, self.n)
        np.random.seed(635871)
        self.V = np.random.rand(self.d, self.n)

    def iterating(self, index: tuple):
        error = np.transpose(self.U)@self.V-self.adj_mat
        loss = error**2
        loss = np.average(loss[self.theta_r])
        i = index[0]
        j = index[1]
        u_i = self.U[:, i]
        v_j = self.V[:, j]
        gradient = self.adj_mat[i, j] - np.transpose(u_i)@v_j
        self.U[:, i] = u_i + self.lr*(gradient*v_j)
        self.V[:, j] = v_j + self.lr*(gradient*u_i)
        return loss


class LogisticClassifier:
    def __init__(
            self, nodes_num: int, target_index: list, target_label: list,
            threshold: float, embedding_size: int, learning_rate: float):
        self.nn = nodes_num
        self.train_idx = target_index
        self.train_label = target_label
        self.train_num = len(self.train_idx)
        self.thr = threshold
        self.em_size = embedding_size
        self.lr = learning_rate
        self.loss_per_iter = []

        np.random.seed(9846)
        self.beta = np.random.rand(self.em_size+1, 1)

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def cost(self, predict):
        label = np.array(self.train_label)
        m = self.train_num
        ce = label*np.log(predict) + (1-label)*np.log(1-predict)
        return -1/m*np.sum(ce)

    def train(self, x):
        # Train number = N'
        # x Shape : (Embedding size, train num)

        # (1, N')
        predict = self.sigmoid(np.transpose(self.beta)@x)
        label = np.array(self.train_label)

        cost = self.cost(predict)
        update_term = 1/self.train_num*np.sum((predict-label)*x)
        self.beta -= self.lr*update_term
        self.loss_per_iter.append(cost)

    def test(self, x):
        predict = self.sigmoid(np.transpose(self.beta)@x)
        predicted_label = np.where(predict > self.thr, True, False)
        return predicted_label

