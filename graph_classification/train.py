import os.path as osp
from math import ceil
import json
import time
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from graph_classification.models import GNN, Net
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, get_laplacian

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS').shuffle()
max_num_nodes = max(dataset.data.num_nodes)
average_nodes = int(dataset.data.x.size(0) / len(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

with open('../configs/config_files/arch1-conf1.json', 'rb') as handle:
    net_config = json.load(handle)

model = Net(net_config, input_dim=dataset.num_features, num_classes=dataset.num_classes).to(device)

# model = GNN(in_channels=dataset.num_features,
#             forward_expansion=8,
#             out_channels=dataset.num_classes,
#             hidden_channels=32,
#             mode='entropy').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

nll_loss = torch.nn.NLLLoss()


def train(epoch):
    model.train()
    loss_all = 0

    for i, data in enumerate(train_loader):
        # print(f'{i} / {len(train_loader)}')
        data = data.to(device)
        optimizer.zero_grad()

        out, loss_pool = model(data.x, data.edge_index, data.batch)

        loss_classification = nll_loss(out, data.y.view(-1))

        loss = loss_classification + 0.01 * loss_pool

        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred, loss_pool = model(data.x, data.edge_index, data.batch)
        loss_classification = nll_loss(pred, data.y.view(-1))
        loss = loss_classification + 0.01 * loss_pool
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct / len(loader.dataset)


best_val_acc = test_acc = 0
best_val_loss = float('inf')
patience = start_patience = 50
for epoch in range(1, 15000):
    s = time.time()
    train_loss = train(epoch)
    _, train_acc = test(train_loader)
    val_loss, val_acc = test(val_loader)
    if val_loss < best_val_loss:
        test_loss, test_acc = test(test_loader)
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print('Epoch: {:03d}, '
          'Train Loss: {:.3f}, Train Acc: {:.3f}, '
          'Val Loss: {:.3f}, Val Acc: {:.3f}, '
          'Test Loss: {:.3f}, Test Acc: {:.3f}, time: {:.1f}'.format(epoch, train_loss,
                                                                     train_acc, val_loss,
                                                                     val_acc, test_loss,
                                                                     test_acc,
                                                                     time.time() - s))