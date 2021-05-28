import os.path as osp
from math import ceil
import json
import time
import torch
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from graph_classification.models import Net
from torch_geometric.transforms import OneHotDegree


OTHER_DATASET_NAMES = ['NCI1', 'NCI109', 'PROTEINS_full', 'REDDIT-BINARY']
#'DD',
# 'PROTEINS', 'ENZYMES', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
#                  'FRANKENSTEIN', 'MUTAG', 'Mutagenicity'
DATASET_NAMES = ['IMDB-MULTI']

BATCH_SIZE = 20
EPOCHS = 200
FOLDS = 10
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.1
EARLY_STOPPING_PATIENCE = 50


# Training, evaluating, and testing a model on each dataset, with each config, for three folds
for dataset_name in DATASET_NAMES:
    print(f'#################### [*] Loading dataset {dataset_name}')
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_name)
    dataset = TUDataset(path, name=dataset_name, transform=OneHotDegree(88), use_node_attr=True, use_edge_attr=True).shuffle()

    n = (len(dataset) + 9) // 10

    input_dim = dataset.num_features
    num_classes = dataset.num_classes

    # Loading config files
    config_file_names = sorted(os.listdir('../configs/config_files'))
    for config_name in config_file_names:
        # if 'arch1' in config_name:
        #     continue
        print(f'******************** [*] Loading config file : {config_name}')
        with open('../configs/config_files/' + config_name, 'rb') as handle:
            config = json.load(handle)

        # Iterating through the folds
        for fold in range(FOLDS):
            print(f'@@@@@@@@@@@@@@@@@ [*] Dataset {dataset_name}, Config: {config_name}, Fold: {fold}')
            wait = None

            dataset = dataset.shuffle()

            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

            device = torch.device('cpu')

            # Model, Optimizer and Loss definitions
            model = Net(config, input_dim=input_dim, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                   patience=SCHEDULER_PATIENCE,
                                                                   factor=SCHEDULER_FACTOR,
                                                                   verbose=True)
            nll_loss = torch.nn.NLLLoss()

            best_val_loss = float('inf')

            for epoch in range(EPOCHS):
                # Training the model
                s_time = time.time()
                train_loss = 0.
                train_corrects = 0
                model.train()
                for i, data in enumerate(train_loader):
                    s = time.time()
                    data = data.to(device)
                    optimizer.zero_grad()
                    out, loss_pool = model(data.x, data.edge_index, data.batch)
                    loss_classification = nll_loss(out, data.y.view(-1))
                    loss = loss_classification + 0.01 * loss_pool

                    loss.backward()
                    train_loss += loss.item()
                    train_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
                    optimizer.step()
                    # print(f'{i}/{len(train_loader)}, {time.time() - s}')

                train_loss /= len(train_loader)
                train_acc = train_corrects / len(train_dataset)
                scheduler.step(train_loss)

                # Validation
                val_loss = 0.
                val_corrects = 0
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        s = time.time()
                        data = data.to(device)
                        out, loss_pool = model(data.x, data.edge_index, data.batch)
                        loss_classification = nll_loss(out, data.y.view(-1))
                        loss = loss_classification + 0.01 * loss_pool
                        val_loss += loss.item()
                        val_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
                        # print(f'{i}/{len(val_loader)}, {time.time() - s}')

                val_loss /= len(val_loader)
                val_acc = val_corrects / len(val_dataset)

                # Test
                test_loss = 0.
                test_corrects = 0
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        s = time.time()
                        data = data.to(device)
                        out, loss_pool = model(data.x, data.edge_index, data.batch)
                        loss_classification = nll_loss(out, data.y.view(-1))
                        loss = loss_classification + 0.01 * loss_pool
                        test_loss += loss.item()
                        test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
                        # print(f'{i}/{len(val_loader)}, {time.time() - s}')

                test_loss /= len(test_loader)
                test_acc = test_corrects / len(test_dataset)

                elapse_time = time.time() - s_time
                log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
                      'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, Elapsed Time: {:.1f}'\
                    .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, elapse_time)
                print(log)

                # Early-Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    # saving the model with best validation loss
                    torch.save(model.state_dict(), f'checkpoints/{dataset_name}_{config_name[:-5]}_{fold}.pkl')
                else:
                    wait += 1
                # early stopping
                if wait == EARLY_STOPPING_PATIENCE:
                    print('======== Early stopping! ========')
                    break

