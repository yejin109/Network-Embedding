import numpy as np


class DeepWalk:
    def __init__(self, lr, ws, wl, wpv, dim, adjacency_matrix, linked, method='Hierarchical'):
        self.adj_mat = adjacency_matrix
        self.nodes_num = self.adj_mat.shape[0]
        self.linked = linked
        self.lr = lr
        self.wl = wl
        self.ws = ws
        self.wpv = wpv
        self.d = dim
        self.loss_per_iter = []
        self.calcul_num = 0
        self.method = method
        self.database = np.ones((self.nodes_num, self.wl, self.wpv)).astype(int)*100
        self.min_depth = np.floor(np.log2(self.nodes_num)).astype(int)
        self.min_depth_num = self.nodes_num - (self.nodes_num - 2 ** self.min_depth) * 2

        np.random.seed(612)
        self.node_order = np.random.choice(self.nodes_num, self.nodes_num, replace=False)
        self.t_l = dict()
        self.node_to_leaf_dict = dict()
        for nn in range(self.nodes_num):
            # Shuffling
            self.node_to_leaf_dict[self.node_order[nn]] = nn
            # No Shuffling
            # self.node_to_leaf_dict[nn] = nn
            # Huffman
            # self.node_to_leaf_dict[degree_sorted[nn]] = nn

        self.node_path_dict = dict()
        self.ch = dict()

        np.random.seed(42)
        self.phi = np.random.rand(self.nodes_num, self.d)
        if method == 'Pure':
            self.second_phi = np.random.rand(self.d, self.nodes_num)
        elif method == 'Hierarchical':
            np.random.seed(756)
            self.inner_unit_matrix = np.random.rand(self.d, self.nodes_num - 1)

    def show_leaves(self):
        print('-' * 30)
        print('leaves Info')
        for key in self.node_to_leaf_dict.keys():
            print(f'No.{self.node_to_leaf_dict[key]} leaf : {key}th Node')
        print('-' * 30)

    def show_path(self, node: int, is_print=False):
        leaf = self.node_to_leaf_dict[node]
        binary = self.leaf_to_binary(leaf)
        decimals = [0]
        for l in range(len(binary) - 1):
            lth_binary = binary[:l + 1]
            lth_decimal_base = np.arange(len(lth_binary))
            lth_decimal_base = np.sum(2 ** lth_decimal_base)
            lth_decimal = int(lth_binary, 2)
            lth_decimal += lth_decimal_base
            decimals.append(lth_decimal)
        if leaf >= self.min_depth_num:
            decimals = decimals[:len(decimals) - 1]
            last_index = len(binary) - 1
            last_binary = binary[:last_index]
            last_decimal = int(last_binary, 2) + np.ceil(np.log2((self.nodes_num-self.min_depth_num)/2)).astype(int)
            decimals.append(last_decimal)
        if is_print:
            print(f'No.{node} Node path : {decimals}')
        return decimals

    def random_walking(self, node: int):
        current_node = node
        random_walks = [current_node]
        link_dict = self.linked

        for t in range(self.wl - 1):
            candidates = link_dict[current_node]
            candidates = np.setdiff1d(candidates, np.array([node]))
            if candidates.size == 0:
                next_node = node
            else:
                next_node = np.random.choice(candidates, candidates.size, 1)[0]
            random_walks.append(next_node)
            current_node = next_node
        return random_walks

    def leaf_to_binary(self, leaf: int):
        if leaf < self.min_depth_num:
            binary = np.binary_repr(leaf)
            while len(binary) < self.min_depth:
                binary = '0' + binary
        else:
            binary = np.binary_repr(self.min_depth_num) + '0'
            surplus = leaf - self.min_depth_num
            int_val = int(binary, 2)
            int_val = int_val + surplus
            binary = bin(int_val)
            binary = binary[2:]
        return binary

    def show_direction(self, node: int):
        leaf = self.node_to_leaf_dict[node]
        binary = self.leaf_to_binary(leaf)
        binary = binary.replace('0', 'L')
        direction = binary.replace('1', 'R')
        return direction

    def sigmoid(self, value):
        value = value
        return 1 / (1 + np.exp(-value))

    def pure_update_function(self, opened_window, random_walks, v_j, w_mat, in_hi_vec, in_vec, target_node):
        w_p_mat = self.second_phi.copy()
        # Hidden to Output Vector; (34,1)
        hi_ou_vec = np.transpose(self.second_phi) @ in_hi_vec

        # Softmax; (34,1)
        exponential = np.exp(hi_ou_vec)
        total_sum = np.sum(exponential)
        y_j = exponential / total_sum

        # k loop : pairwise output loop
        for k in opened_window:
            context_node = random_walks[k]
            if context_node == v_j:
                continue
            u_j_star = np.zeros(self.nodes_num)
            u_j_star = u_j_star.reshape((self.nodes_num, 1))
            u_j_star[context_node] = 1

            # (34,1)
            t_j = u_j_star.reshape((self.nodes_num, 1))

            u_j_star = np.transpose(w_mat[context_node]).reshape((self.d, 1))
            u_j_star = np.transpose(w_p_mat) @ u_j_star

            # (34,1)
            e_j = y_j - t_j

            # (2,34)
            e_sp = in_hi_vec @ np.transpose(e_j)

            # (2,1)
            eh_i = self.second_phi @ e_j

            # (34,2)
            e_fp = in_vec @ np.transpose(eh_i)
            e_fp = e_fp[target_node]

            # Error Function Partial Derivatives
            self.phi[target_node] -= self.lr * e_fp
            self.second_phi -= self.lr * e_sp
            loss = np.log(total_sum) - u_j_star
            self.loss_per_iter.append(loss[context_node])
            self.calcul_num += 1

    def hierarchical_update_function(self, opened_window, random_walks, v_j, inner_unit_mat, in_hi_vec, target_node):
        # k loop : pairwise output loop
        for k in opened_window:
            context_node = random_walks[k]
            channel = self.ch[context_node]
            if context_node == v_j:
                continue

            # H-softmax
            p_l_decimal = self.node_path_dict[context_node]

            # (2, paths)
            p_l = inner_unit_mat[:, p_l_decimal]

            # (paths, 1)
            path_l = self.sigmoid(channel*np.transpose(p_l) @ in_hi_vec)
            loss = -np.log(path_l)

            e_v_l_h = channel*(path_l-1)
            e_v_l_h = np.transpose(e_v_l_h)
            e_v_l_h = np.tile(e_v_l_h, (2, 1))

            eh_l = e_v_l_h * p_l
            e_v_l = e_v_l_h * in_hi_vec

            self.inner_unit_matrix[:, p_l_decimal] -= self.lr * e_v_l
            self.phi[target_node] -= self.lr * np.sum(eh_l, axis=1)
            self.loss_per_iter.append(np.sum(loss))
            self.calcul_num += 1

    def skip_gram(self, random_walks: list):
        random_walks = np.array(random_walks)

        index_array = [wl for wl in range(self.wl)]

        padding = [np.nan] * self.ws
        index_array = np.array(padding + index_array + padding)

        # j loop : input loop
        for j, v_j in enumerate(random_walks):
            # Update연산과 Loss 연산이 서로 영향끼치지 않도록 설정한 것입니다.
            phi_mat = self.phi.copy()

            # One Hot Encoding; (34,1)
            target_node = v_j
            in_vec = np.zeros(self.nodes_num, dtype=np.int32).reshape((self.nodes_num, 1))
            in_vec[target_node] = 1

            # Input to Hidden Vector; (2,1)
            in_hi_vec = np.transpose(phi_mat[target_node]).reshape((self.d, 1))

            # window로 들여다 볼 범위 설정
            opened_window = index_array[j:j + self.ws * 2 + 1]
            opened_window = opened_window[~np.isnan(opened_window)]
            opened_window = opened_window.astype(dtype=np.int32)

            if self.method == 'Hierarchical':
                inner_unit_mat = self.inner_unit_matrix.copy()
                self.hierarchical_update_function(opened_window, random_walks, v_j,
                                                  inner_unit_mat, in_hi_vec, target_node)
            elif self.method == 'Pure':
                self.pure_update_function(opened_window, random_walks, v_j, phi_mat, in_hi_vec, in_vec, target_node)

    def generate_db(self):
        # (node, walks, walk per vertex) DB
        for seed in range(self.nodes_num):
            for i in range(self.wpv):
                rw = self.random_walking(seed)
                for random_walk_idx, random_walk in enumerate(rw):
                    self.database[seed, random_walk_idx, i] = random_walk
            # Path in a Binary Tree
            self.node_path_dict[seed] = self.show_path(seed)
            leaf = self.node_to_leaf_dict[seed]
            binary = self.leaf_to_binary(leaf)
            target = [[int(path)] for path in binary]
            target = np.array(target)
            t_l = np.where(target == 0, 1, 0)
            ch = np.where(target == 0, 1, -1)
            self.t_l[seed] = t_l
            self.ch[seed] = ch

    def propagating(self):
        # i loop : walk per vertex loop
        for i in range(self.wpv):
            order = np.random.choice(self.nodes_num, self.nodes_num, replace=False)
            for v_i in range(self.nodes_num):
                v_i = order[v_i]
                rw_sequence = self.database[v_i, :, i]
                self.skip_gram(rw_sequence)
