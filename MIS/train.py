import os
import os.path as osp
from models import MISSolver
from torch_geometric.datasets import Planetoid
import torch.optim as optim
import torch
import time


os.makedirs('checkpoints', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
name = 'Cora'
dataset = Planetoid(path, name=name)
num_nodes = dataset.data.train_mask.size(0)
edge_index = dataset.data.edge_index
w = torch.ones(num_nodes, dtype=torch.float32)
batch = torch.zeros(num_nodes, dtype=torch.long)

model = MISSolver().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# defining the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       patience=10,
                                                       factor=0.1,
                                                       verbose=True)
mis_best = 0
for epoch in range(1000):
    s = time.time()

    loss, mis = model(w.to(device), edge_index.to(device), batch.to(device))

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    scheduler.step(loss.item())

    l_mis = len(mis[0])
    if l_mis > mis_best:
        mis_best = l_mis
        torch.save(model.state_dict(), f'checkpoints/{name}.pkl')

    print('[*] Epoch : {}, Loss : {:.3f}, |MIS| = {}, Best : {} Time : {:.1f}'
          .format(epoch, loss.item(), l_mis, mis_best, time.time() - s))

