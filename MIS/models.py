import numpy as np
import torch
import inspect
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch, to_dense_adj


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


class MISSolver(nn.Module):
    def __init__(self):
        super(MISSolver, self).__init__()

        self.conv1 = GINConv(nn=MLP([1, 64, 64, 64]), train_eps=True)
        self.conv2 = GINConv(nn=MLP([64, 64, 64, 64]), train_eps=True)
        self.conv3 = GINConv(nn=MLP([64, 64, 64, 64]), train_eps=True)
        self.conv4 = GINConv(nn=MLP([64, 64, 64, 64]), train_eps=True)
        self.conv5 = GINConv(nn=MLP([64, 64, 64, 64]), train_eps=True)
        self.conv6 = GINConv(nn=MLP([64, 64, 64, 1]), train_eps=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.bn5 = nn.BatchNorm1d(num_features=64)

    def forward(self, w, edge_index, batch):
        prob = torch.relu(self.bn1(self.conv1(w.unsqueeze(1), edge_index)))
        prob = torch.relu(self.bn2(self.conv2(prob, edge_index)))
        prob = torch.relu(self.bn3(self.conv3(prob, edge_index)))
        # prob = torch.relu(self.bn4(self.conv4(prob, edge_index)))
        prob = torch.relu(self.bn5(self.conv5(prob, edge_index)))

        prob = torch.sigmoid(self.conv6(prob, edge_index))

        prob_dense, prob_mask = to_dense_batch(prob, batch)
        w_dense, w_mask = to_dense_batch(w, batch)
        gammas = w_dense.sum(dim=1)

        adj = to_dense_adj(edge_index, batch)

        loss_thresholds = self.calculate_loss_thresholds(w_dense, prob_dense, adj, gammas)

        loss = loss_thresholds.sum() / adj.size(0)

        mis = self.conditional_expectation(
            w_dense.detach(),
            prob_dense.detach(),
            adj,
            loss_thresholds.detach(),
            gammas.detach(),
            prob_mask.detach()
        )

        return loss, mis

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

        if term1 == 0. and term2 == 0.:
            print(w)
            print(x)
            print(inspect.stack()[1].function)
            exit()
        # assert term1 == -term2

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
