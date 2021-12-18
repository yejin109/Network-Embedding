import numpy as np


class LINE:
    def __init__(self, flr, slr, dim, adjacency_matrix, adjacency_matrix_index, degree, negative_num, method='Pure'):
        self.adj_mat = adjacency_matrix
        self.adj_mat_idx = adjacency_matrix_index
        self.nodes_num = self.adj_mat.shape[0]
        self.fst_lr = flr
        self.snd_lr = slr
        self.d = dim
        self.loss_per_iter = []
        self.calcul_num = 0
        self.ns_num = negative_num
        self.method = method

        np.random.seed(790695)
        # np.random.seed(6578)
        self.fst_emb = np.random.normal(0, 1, (self.nodes_num, self.d))
        np.random.seed(15389)
        # np.random.seed(568)
        self.snd_target = np.random.normal(0, 1, (self.nodes_num, self.d))
        np.random.seed(6135)
        # np.random.seed(123)
        self.snd_context = np.random.normal(0, 1, (self.nodes_num, self.d))

        if method == 'NS':
            self.noise_dist = np.zeros((self.nodes_num, self.nodes_num))
            self.negative_count = np.zeros((self.nodes_num, self.nodes_num))
            negative_sample = np.argwhere(adjacency_matrix == 0)
            for context, target in negative_sample:
                if context != target:
                    self.noise_dist[context, target] = (degree[context])**(1/8)

            self.noise_dist = self.noise_dist / np.sum(self.noise_dist, axis=0)

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def ns_optimizing(self, target, context, dot_prod_mat, t_j):
        sigmoid = self.sigmoid(dot_prod_mat[target, context])

        sigmoid = sigmoid.reshape((sigmoid.size, 1))
        derivative = (sigmoid - t_j)
        obj = - np.sum(np.log(self.sigmoid(np.where(t_j == 1, 1, -1) * dot_prod_mat[target, context]
                                       .reshape(sigmoid.size, 1))))
        return derivative, obj

    def propagating(self):
        # First Order Proximity : (34,34) / (context, target)
        prior_fst_emb = self.fst_emb.copy()
        fst_dot_prod_mat = self.fst_emb @ self.fst_emb.T

        # Second Order Proximity: (34,34) / (context, target)
        prior_snd_target = self.snd_target.copy()
        prior_snd_context = self.snd_context.copy()
        snd_dot_prod_mat = self.snd_target @ self.snd_context.T

        # Loss Value
        fst_obj = []
        snd_obj = []

        case_num = len(self.adj_mat_idx)
        # Update 식에서 Unweighted graph 의 경d우 weight를 호출할 필요가 없어 생략하였다.
        for idx in np.random.choice(case_num, case_num, replace=False):
            target, context = self.adj_mat_idx[idx]

            # Negative Sampling
            ns = np.random.choice(np.arange(self.nodes_num), self.ns_num, replace=False, p=self.noise_dist[:, target])
            self.negative_count[target, ns] += 1

            # pn = self.noise_dist[target, ns]
            # pn = np.append(1, pn).reshape((self.ns_num+1, 1))
            indicator = np.append(context, ns)
            t_j = np.zeros((self.ns_num+1, 1))
            t_j[0] = 1

            fst_derivative, fst_obj_indi = self.ns_optimizing(target, indicator, fst_dot_prod_mat, t_j)
            fst_obj.append(fst_obj_indi)
            fst_prior = prior_fst_emb[indicator]
            self.fst_emb[target] -= self.fst_lr * np.sum(fst_derivative * fst_prior, axis=0)
            self.fst_emb[indicator] -= self.fst_lr * fst_derivative * prior_fst_emb[target]

            snd_derivative, snd_obj_indi = self.ns_optimizing(target, indicator, snd_dot_prod_mat, t_j)
            snd_obj.append(snd_obj_indi)
            snd_context_prior = prior_snd_context[indicator]
            self.snd_target[target] -= self.snd_lr * np.sum(snd_derivative * snd_context_prior, axis=0)
            self.snd_context[indicator] -= self.snd_lr * snd_derivative * prior_snd_target[target]

        return np.average(fst_obj), np.average(snd_obj)
