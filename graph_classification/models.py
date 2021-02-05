import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, DenseGINConv, GCNConv, SAGEConv, DenseGCNConv, DenseSAGEConv
from torch_geometric.utils import get_laplacian, to_dense_batch, to_dense_adj


class MLP(nn.Module):
    def __init__(self, size_list, batch_norm=False, dropout=0., activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        for i in range(len(size_list) - 1):
            self.mlp.append(nn.Linear(in_features=int(size_list[i]), out_features=int(size_list[i + 1])))
            if i != len(size_list) - 2:
                self.mlp.append(activation)

                if batch_norm is True:
                    self.mlp.append(nn.BatchNorm1d(num_features=size_list[i + 1]))

                self.mlp.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.mlp:
            if 'Batch' in layer.__class__.__name__:
                if len(x.size()) == 2:
                    x = layer(x)
                else:
                    x = layer(x.view(-1, x.size(-1))).view(x.size())
            else:
                x = layer(x)
        return x


class MWISPool(nn.Module):
    def __init__(self, conv_layer_type, mlp_list=None, in_channels_list=None,
                 out_channels_list=None, batch_norm_list=None, dropout_list=None,
                 activation=nn.ReLU(), mode='global_entropy'):
        super(MWISPool, self).__init__()

        self.pool = nn.ModuleList()
        self.mode = mode

        if conv_layer_type == 'GINConv':
            for i in range(len(mlp_list)):
                self.pool.append(GINConv(nn=mlp_list[i]))
                if i != len(mlp_list) - 1:
                    self.pool.append(activation)
                else:
                    self.pool.append(nn.Sigmoid())
        elif conv_layer_type == 'GCNConv':
            for i in range(len(in_channels_list)):
                self.pool.append(GCNConv(in_channels=in_channels_list[i], out_channels=out_channels_list[i]))

                if i != len(in_channels_list) - 1:
                    self.pool.append(activation)
                else:
                    self.pool.append(nn.Sigmoid())

                if batch_norm_list[i] and i != len(in_channels_list) - 1:
                    self.pool.append(nn.BatchNorm1d(num_features=out_channels_list[i]))

                if i != len(in_channels_list) - 1:
                    self.pool.append(nn.Dropout(dropout_list[i]))
        elif conv_layer_type == 'SAGEConv':
            for i in range(len(in_channels_list)):
                self.pool.append(SAGEConv(in_channels=in_channels_list[i], out_channels=out_channels_list[i]))

                if i != len(in_channels_list) - 1:
                    self.pool.append(activation)
                else:
                    self.pool.append(nn.Sigmoid())

                if batch_norm_list[i] and i != len(in_channels_list) - 1:
                    self.pool.append(nn.BatchNorm1d(num_features=out_channels_list[i]))

                if i != len(in_channels_list) - 1:
                    self.pool.append(nn.Dropout(dropout_list[i]))

    def forward(self, x=None, edge_index=None, batch=None, **kwargs):
        if 'adj' in kwargs.keys():
            edge_index, batch = self.to_edge_index(kwargs['adj'], kwargs['mwis'])
            adj = kwargs['adj']
        else:
            adj = to_dense_adj(edge_index, batch)

        if 'x_dense' in kwargs.keys():
            x = self.to_sparse_signal(kwargs['x_dense'], batch)

        if edge_index.size(1) != 0:
            L_indices, L_values = get_laplacian(edge_index)
            batch_nodes = batch.size(0)
            L = torch.sparse.FloatTensor(L_indices, L_values, torch.Size([batch_nodes, batch_nodes]))

            w = torch.norm(torch.matmul(L, x), dim=-1)

            w = self.entropy_weights(w, adj, batch, mode=self.mode)
        else:
            norm = torch.norm(x, dim=1)
            w = norm / norm

        for i, layer in enumerate(self.pool):
            if i == 0:
                prob = self.pool[i](w, edge_index)
            else:
                if isinstance(self.pool[i], GINConv) or isinstance(self.pool[i], GCNConv) or isinstance(self.pool[i], SAGEConv):
                    prob = self.pool[i](prob, edge_index)
                else:
                    prob = self.pool[i](prob)

        prob_dense, prob_mask = to_dense_batch(prob, batch)
        w_dense, w_mask = to_dense_batch(w, batch)
        gammas = w_dense.sum(dim=1)

        loss_thresholds = self.calculate_loss_thresholds(w_dense, prob_dense, adj, gammas)

        loss = loss_thresholds.sum() / adj.size(0)

        mwis = self.conditional_expectation(
            w_dense.detach(),
            prob_dense.detach(),
            adj,
            loss_thresholds.detach(),
            gammas.detach(),
            prob_mask.detach()
        )
        # construction of pooled adj and signal using mwis
        x_dense, _ = to_dense_batch(x, batch)
        x_dense_pooled, adj_pooled = self.graph_reconstruction(mwis, x_dense, adj)

        return x_dense_pooled, adj_pooled, loss, mwis

    def calculate_loss_thresholds(self, w, x, adj, gammas):
        loss_thresholds = []
        batch_size = adj.size(0)
        for b in range(batch_size):
            loss_thresholds.append(
                self.loss_fn(w[b], x[b], adj[b], gammas[b]).unsqueeze(0)
            )
        loss_thresholds = torch.cat(loss_thresholds)
        return loss_thresholds

    @staticmethod
    def loss_fn(w, x, adj, gamma=0):
        term1 = -torch.matmul(w.t(), x)

        term2 = torch.matmul(torch.matmul(x.t(), adj), x).sum()

        return gamma + term1 + term2

    def conditional_expectation(self, w, probability_vector, adj, loss_threshold, gammas, mask):
        s = time.time()
        sorted_prob_vector = torch.sort(probability_vector, descending=True, dim=1)

        selected = [set() for _ in range(adj.size(0))]
        rejected = [set() for _ in range(adj.size(0))]

        prob_vector_copy = probability_vector.clone()

        for b in range(adj.size(0)):
            for i in range(len(sorted_prob_vector.values[b])):
                node_index = sorted_prob_vector.indices[b][i].item()
                neighbors = torch.where(adj[b][node_index] == 1)[0]
                if len(neighbors) == 0:
                    selected[b].add(node_index)
                    continue
                if node_index not in rejected[b] and node_index not in selected[b]:
                    temp_prob_vector = prob_vector_copy.clone()
                    temp_prob_vector[b, node_index] = 1
                    temp_prob_vector[b, neighbors] = 0

                    loss = self.loss_fn(w[b], temp_prob_vector[b], adj[b], gammas[b])

                    if loss <= loss_threshold[b]:
                        selected[b].add(node_index)
                        for n in neighbors.tolist():
                            rejected[b].add(n)

                        prob_vector_copy[b, node_index] = 1
                        prob_vector_copy[b, neighbors] = 0

            mwis = np.array(list(selected[b]))
            masked_mwis = mwis[mwis < len(mask[b][mask[b] == True])]
            selected[b] = list(masked_mwis)

        return selected

    @staticmethod
    def graph_reconstruction(mwis, x_dense, adj):
        adj_2 = torch.matrix_power(adj, 2)
        adj_2 = adj_2 - torch.diag_embed(torch.diagonal(adj_2, dim1=1, dim2=2))
        adj_2 = torch.clamp(adj_2, 0, 1)
        adj_pooled_size = max(map(len, mwis))
        adj_pooled = torch.zeros((adj.size(0), adj_pooled_size, adj_pooled_size), dtype=adj.dtype).to(adj.device)
        x_dense_pooled = torch.zeros((adj.size(0), adj_pooled_size, x_dense.size(-1)),
                                     dtype=x_dense.dtype).to(x_dense.device)
        # TODO: shift x_dense on adj
        for b in range(adj.size(0)):
            adj_pooled[b][:len(mwis[b]), :len(mwis[b])] = adj_2[b][mwis[b]][:, mwis[b]]
            x_dense_pooled[b][:len(mwis[b])] = x_dense[b][mwis[b]]
        return x_dense_pooled, adj_pooled

    @staticmethod
    def to_edge_index(adj, mwis):
        batch = []
        start_index = 0
        upper_row = []
        lower_row = []

        for b in range(adj.size(0)):
            indices = torch.where(adj[b] == 1)
            u = (indices[0] + start_index).tolist()
            l = (indices[1] + start_index).tolist()
            size = len(mwis[b])
            upper_row.extend(u)
            lower_row.extend(l)
            batch.extend([b] * size)
            start_index += size

        edge_index = torch.tensor([upper_row, lower_row], dtype=torch.int64).to(adj.device)
        batch = torch.tensor(batch, dtype=torch.long).to(adj.device)

        return edge_index, batch

    @staticmethod
    def to_sparse_signal(x, batch):
        x_sparse = []

        for b in range(x.size(0)):
            x_sparse.append(x[b][:len(batch[batch == b])])

        x_sparse = torch.cat(x_sparse).to(x.device)

        return x_sparse

    @staticmethod
    def entropy_weights(w, adj, batch, mode):
        batch_size = (batch[-1] + 1).item()

        entropy_weights = torch.zeros(w.size(), dtype=w.dtype).to(w.device)

        for b in range(batch_size):
            weights = w[batch == b]
            weights = torch.exp(-weights)

            if mode == 'global_entropy':
                weights = torch.softmax(weights, dim=0)
            elif mode == 'local_entropy':
                # repeat weights
                weights_matrix = weights.unsqueeze(0).repeat(weights.size(0), 1)
                # hadamard product with adj
                local_probs = torch.diag_embed(weights) + weights_matrix * adj[b][:weights.size(0), :weights.size(0)]
                weights = torch.diag(torch.softmax(local_probs, dim=1))
            else:
                raise Exception('[!] Wrong mode!')

            entropy = -weights * torch.log(weights)
            entropy_weights[batch == b] = entropy

        return entropy_weights


class ShannonCapacityPooling(nn.Module):
    def __init__(self, forward_expansion, hidden_dim, channel_noise_threshold):
        super(ShannonCapacityPooling, self).__init__()
        self.threshold = channel_noise_threshold
        self.conv1 = DenseGINConv(
            nn=MLP(input_dim=1, forward_expansion=forward_expansion, output_dim=hidden_dim)
        )
        self.conv2 = DenseGINConv(
            nn=MLP(input_dim=hidden_dim, forward_expansion=1, output_dim=hidden_dim)
        )
        self.conv3 = DenseGINConv(
            nn=MLP(input_dim=hidden_dim, forward_expansion=1, output_dim=1)
        )

    def forward(self, x=None, edge_index=None, batch=None, **kwargs):
        if 'adj' in kwargs.keys():
            adj = kwargs['adj']
            _, batch = self.to_edge_index(kwargs['adj'], kwargs['mwis'])
        else:
            adj = to_dense_adj(edge_index, batch)
        if 'x_dense' in kwargs.keys():
            x = self.to_sparse_signal(kwargs['x_dense'], batch)

        x_dense, x_dense_mask = to_dense_batch(x, batch)
        N = adj.size(1)
        # q = x_dense.unsqueeze(1).repeat(1, N, 1, 1)
        # q_t = q.transpose(dim0=1, dim1=2)
        channel = torch.norm(
            x_dense.unsqueeze(1).repeat(1, N, 1, 1) - x_dense.unsqueeze(1).repeat(1, N, 1, 1).transpose(dim0=1, dim1=2),
            dim=-1) * adj
        channel = torch.where(
            channel > self.threshold * channel.max(),
            torch.tensor(0, dtype=channel.dtype, device=channel.device),
            channel
        )
        channel = torch.where(channel > 0, torch.tensor(1, dtype=channel.dtype, device=channel.device), channel)
        weights = torch.ones((adj.size(0), adj.size(1), 1), dtype=x_dense.dtype, device=adj.device)

        prob = torch.relu(self.conv1(weights, channel))
        prob = torch.relu(self.conv2(prob, channel))
        prob = torch.sigmoid(self.conv3(prob, channel))

        gammas = weights.sum(dim=1)

        loss_thresholds = self.calculate_loss_thresholds(weights, prob, adj, gammas)

        loss = loss_thresholds.sum() / adj.size(0)

        mwis = self.conditional_expectation(
            weights.detach(),
            prob.detach(),
            adj,
            loss_thresholds.detach(),
            gammas.detach(),
            x_dense_mask.detach()
        )
        # construction of pooled adj and signal using mwis
        x_dense_pooled, adj_pooled = self.graph_reconstruction(mwis, x_dense, adj)

        return x_dense_pooled, adj_pooled, loss, mwis

    def calculate_loss_thresholds(self, w, x, adj, gammas):
        loss_thresholds = []
        batch_size = adj.size(0)
        for b in range(batch_size):
            loss_thresholds.append(
                self.loss_fn(w[b], x[b], adj[b], gammas[b]).unsqueeze(0)
            )
        loss_thresholds = torch.cat(loss_thresholds)
        return loss_thresholds

    @staticmethod
    def loss_fn(w, x, adj, gamma=0):
        term1 = -torch.matmul(w.t(), x)

        term2 = torch.matmul(torch.matmul(x.t(), adj), x).sum()

        return gamma + term1 + term2

    def conditional_expectation(self, w, probability_vector, adj, loss_threshold, gammas, mask):
        sorted_prob_vector = torch.sort(probability_vector, descending=True, dim=1)

        selected = [set() for _ in range(adj.size(0))]
        rejected = [set() for _ in range(adj.size(0))]

        prob_vector_copy = probability_vector.clone()

        for b in range(adj.size(0)):
            for i in range(len(sorted_prob_vector.values[b])):
                node_index = sorted_prob_vector.indices[b][i].item()
                neighbors = torch.where(adj[b][node_index] == 1)[0]
                if len(neighbors) == 0:
                    selected[b].add(node_index)
                    continue
                if node_index not in rejected[b] and node_index not in selected[b]:
                    temp_prob_vector = prob_vector_copy.clone()
                    temp_prob_vector[b, node_index] = 1
                    temp_prob_vector[b, neighbors] = 0

                    loss = self.loss_fn(w[b], temp_prob_vector[b], adj[b], gammas[b])

                    if loss <= loss_threshold[b]:
                        selected[b].add(node_index)
                        for n in neighbors.tolist():
                            rejected[b].add(n)

                        prob_vector_copy[b, node_index] = 1
                        prob_vector_copy[b, neighbors] = 0

            mwis = np.array(list(selected[b]))
            masked_mwis = mwis[mwis < len(mask[b][mask[b] == True])]
            selected[b] = list(masked_mwis)

        return selected

    @staticmethod
    def graph_reconstruction(mwis, x_dense, adj):
        adj_2 = torch.matrix_power(adj, 2)
        adj_2 = adj_2 - torch.diag_embed(torch.diagonal(adj_2, dim1=1, dim2=2))
        adj_2 = torch.clamp(adj_2, 0, 1)
        adj_pooled_size = max(map(len, mwis))
        adj_pooled = torch.zeros((adj.size(0), adj_pooled_size, adj_pooled_size), dtype=adj.dtype).to(adj.device)
        x_dense_pooled = torch.zeros((adj.size(0), adj_pooled_size, x_dense.size(-1)),
                                     dtype=x_dense.dtype).to(x_dense.device)
        # TODO: shift x_dense on adj
        for b in range(adj.size(0)):
            adj_pooled[b][:len(mwis[b]), :len(mwis[b])] = adj_2[b][mwis[b]][:, mwis[b]]
            x_dense_pooled[b][:len(mwis[b])] = x_dense[b][mwis[b]]
        return x_dense_pooled, adj_pooled

    @staticmethod
    def to_edge_index(adj, mwis):
        batch = []
        start_index = 0
        upper_row = []
        lower_row = []
        for b in range(adj.size(0)):
            indices = torch.where(adj[b] == 1)
            u = (indices[0] + start_index).tolist()
            l = (indices[1] + start_index).tolist()
            size = len(mwis[b])
            upper_row.extend(u)
            lower_row.extend(l)
            batch.extend([b] * size)
            start_index += size

        edge_index = torch.tensor([upper_row, lower_row], dtype=torch.int64).to(adj.device)
        batch = torch.tensor(batch, dtype=torch.long).to(adj.device)

        return edge_index, batch

    @staticmethod
    def to_sparse_signal(x, batch):
        x_sparse = []

        for b in range(x.size(0)):
            x_sparse.append(x[b][:len(batch[batch == b])])

        x_sparse = torch.cat(x_sparse).to(x.device)

        return x_sparse


class GNN(nn.Module):
    def __init__(self, in_channels, forward_expansion, out_channels, hidden_channels=32, mode='variation'):
        super(GNN, self).__init__()

        self.mode = mode

        self.conv1 = GINConv(nn=MLP(in_channels, forward_expansion, hidden_channels))

        self.pool1 = MWISPool(forward_expansion, hidden_channels)

        self.conv2 = DenseGINConv(nn=MLP(hidden_channels, forward_expansion, hidden_channels))

        self.pool2 = MWISPool(forward_expansion, hidden_channels)

        self.conv3 = DenseGINConv(nn=MLP(hidden_channels, forward_expansion, hidden_channels))

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x_pooled, adj_pooled, loss_pool1, mwis1 = self.pool1(x, edge_index, batch, mode=self.mode)

        x = torch.relu(self.conv2(x_pooled, adj_pooled))
        x_pooled, adj_pooled, loss_pool2, mwis2 = self.pool2(x_dense=x, adj=adj_pooled, mwis=mwis1, mode=self.mode)

        x = self.conv3(x_pooled, adj_pooled)

        x = x.mean(dim=1)
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        # print(x)
        return torch.log_softmax(x, dim=-1), loss_pool1 + loss_pool2


class Net(nn.Module):
    def __init__(self, config_file, input_dim, num_classes):
        super(Net, self).__init__()

        self.net = nn.ModuleList()
        for layer_num, layer in enumerate(config_file.keys()):
            layer_type = config_file[layer]['layer_type']
            layer_activation = config_file[layer]['activation']
            layer_activation = self.get_activation(layer_activation)

            if layer_type in ['GINConv', 'DenseGINConv']:
                size_list = config_file[layer]['mlp']['architecture']
                if layer_num == 0:
                    size_list.insert(0, input_dim)
                batch_norm = config_file[layer]['mlp']['batch_norm']
                dropout = float(config_file[layer]['mlp']['dropout'])
                mlp = MLP(size_list=size_list, batch_norm=batch_norm, dropout=dropout, activation=layer_activation)
                if layer_type == 'GINConv':
                    self.net.append(GINConv(nn=mlp))
                elif layer_type == 'DenseGINConv':
                    self.net.append(DenseGINConv(nn=mlp))

                if layer_num != len(config_file.keys()) - 1:
                    self.net.append(layer_activation)
            elif layer_type in ['GCNConv', 'DenseGCNConv', 'SAGEConv', 'DenseSAGEConv']:
                if layer_num == 0:
                    in_channels = input_dim
                else:
                    in_channels = int(config_file[layer]['in_channels'])
                out_channels = int(config_file[layer]['out_channels'])
                batch_norm = config_file[layer]['batch_norm']
                dropout = float(config_file[layer]['dropout'])
                if layer_type == 'GCNConv':
                    self.net.append(GCNConv(in_channels=in_channels, out_channels=out_channels))
                elif layer_type == 'DenseGCNConv':
                    self.net.append(DenseGCNConv(in_channels=in_channels, out_channels=out_channels))
                elif layer_type == 'SAGEConv':
                    self.net.append(SAGEConv(in_channels=in_channels, out_channels=out_channels))
                elif layer_type == 'DenseSAGEConv':
                    self.net.append(DenseSAGEConv(in_channels=in_channels, out_channels=out_channels))

                if layer_num != len(config_file.keys()) - 1:
                    self.net.append(layer_activation)

                    if batch_norm is True:
                        self.net.append(nn.BatchNorm1d(num_features=out_channels))

                    self.net.append(nn.Dropout(p=dropout))
            elif layer_type == 'MWISPool':
                mlp_list = []
                in_channels_list = []
                out_channels_list = []
                batch_norm_list = []
                dropout_list = []
                pooling_layer_type = 'GINConv'
                pooling_mode = config_file[layer]['entropy_mode']
                for pooling_layer in config_file[layer]['pooling_layers'].keys():
                    layer_info = config_file[layer]['pooling_layers'][pooling_layer]
                    if layer_info['layer_type'] == 'GINConv':
                        pooling_mlp_size = layer_info['mlp']['architecture']
                        pooling_mlp_batch_norm = layer_info['mlp']['batch_norm']
                        pooling_mlp_dropout = float(layer_info['mlp']['dropout'])
                        pooling_mlp = MLP(size_list=pooling_mlp_size,
                                          batch_norm=pooling_mlp_batch_norm,
                                          dropout=pooling_mlp_dropout,
                                          activation=layer_activation)
                        mlp_list.append(pooling_mlp)
                    elif layer_info['layer_type'] in ['GCNConv', 'SAGEConv']:
                        pooling_layer_type = layer_info['layer_type']
                        pooling_in_channels = int(layer_info['in_channels'])
                        pooling_out_channels = int(layer_info['out_channels'])
                        pooling_batch_norm = layer_info['batch_norm']
                        pooling_dropout = float(layer_info['dropout'])

                        in_channels_list.append(pooling_in_channels)
                        out_channels_list.append(pooling_out_channels)
                        batch_norm_list.append(pooling_batch_norm)
                        dropout_list.append(pooling_dropout)

                self.net.append(MWISPool(pooling_layer_type, mlp_list, in_channels_list, out_channels_list,
                                         batch_norm_list, dropout_list, layer_activation, pooling_mode))

        last_layer = config_file[list(config_file.keys())[-1]]

        if last_layer['layer_type'] in ['GINConv', 'DenseGINConv']:
            hidden_channels = int(last_layer['mlp']['architecture'][-1])
        else:
            hidden_channels = int(last_layer['out_channels'])

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        pool_losses = []
        mwis_list = []
        for layer in self.net:
            if 'Conv' in layer.__class__.__name__:
                if 'Dense' in layer.__class__.__name__:
                    x = layer(x_pooled, adj_pooled)
                else:
                    x = layer(x, edge_index)

            elif 'MWIS' in layer.__class__.__name__:
                if len(mwis_list) == 0:
                    x_pooled, adj_pooled, loss_pool, mwis = layer(x, edge_index, batch)
                    # print('here')
                    pool_losses.append(loss_pool)
                    mwis_list.append(mwis)
                else:
                    x_pooled, adj_pooled, loss_pool, mwis = layer(x_dense=x, adj=adj_pooled, mwis=mwis_list[-1])

                    pool_losses.append(loss_pool)
                    mwis_list.append(mwis)
            else:
                x = layer(x)

        x = x.mean(dim=1)
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)

        return torch.log_softmax(x, dim=-1), sum(pool_losses)

    @staticmethod
    def get_activation(activation_name):
        if activation_name == 'ReLU':
            activation = nn.ReLU()
        elif activation_name == 'LeakyReLU':
            activation = nn.LeakyReLU()
        elif activation_name == 'PReLU':
            activation = nn.PReLU()
        elif activation_name == 'RReLU':
            activation = nn.RReLU()
        else:
            activation = None

        return activation
