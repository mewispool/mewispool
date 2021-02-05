import json

config_name = input("Enter config name : ")

network_num_layer = int(input("How many layers does the network have ? : "))

network_architecture = {}

for i in range(network_num_layer):
    layer = input(f"Enter the type of layer {i + 1} : ")
    layer_activation = input(f"Enter the activation function of layer {i + 1} : ")

    network_architecture[i] = {'layer_type': layer}
    network_architecture[i]['activation'] = layer_activation

    if layer in ['GINConv', 'DenseGINConv']:
        mlp = input("Enter the MLP layer dimensions, e.g, 16 32 32 8: ")
        mlp = list(map(int, mlp.split(' ')))
        batch_norm = input("Does the MLP have batch normalization ? (y/n) ")
        batch_norm = True if batch_norm == 'y' else False
        dropout = input("Does the MLP have dropout ? (y/n) ")
        dropout = 0.2 if dropout == 'y' else 0.

        network_architecture[i]['mlp'] = {'architecture': mlp,
                                          'batch_norm': batch_norm,
                                          'dropout': dropout}
    elif layer in ['GCNConv', 'DenseGCNConv', 'SAGEConv', 'DenseSAGEConv']:
        in_channels = int(input(f"Enter {layer} in_channels parameter : "))
        out_channels = int(input(f"Enter {layer} out_channels parameter : "))

        batch_norm = input(f"Does the {layer} have batch normalization ? (y/n) ")
        batch_norm = True if batch_norm == 'y' else False
        dropout = input(f"Does the {layer} have dropout ? (y/n) ")
        dropout = 0.2 if dropout == 'y' else 0.

        network_architecture[i]['in_channels'] = in_channels
        network_architecture[i]['out_channels'] = out_channels
        network_architecture[i]['batch_norm'] = batch_norm
        network_architecture[i]['dropout'] = dropout
    elif layer == 'MWISPool':
        network_architecture[i]['pooling_layers'] = {}

        pooling_mode = input("Enter the entropy type for pooling (1: local_entropy, 2: global_entropy) : ")
        pooling_mode = 'local_entropy' if pooling_mode == '1' else 'global_entropy'

        network_architecture[i]['entropy_mode'] = pooling_mode

        pool_layer = input("Enter the type of pooling layer : ")
        pool_hidden_sizes = input(f"Enter the hidden dimensions of conv layers "
                                  f"(you should enter numbers starting and ending with 1),e.g, 1 16 32 1: ")
        pool_hidden_sizes = pool_hidden_sizes.split(' ')
        pool_hidden_sizes = list(map(int, pool_hidden_sizes))

        for j in range(len(pool_hidden_sizes) - 1):
            network_architecture[i]['pooling_layers'][j] = {}
            if pool_layer == 'GINConv':
                network_architecture[i]['pooling_layers'][j]['layer_type'] = pool_layer
                pool_mlp = input(f"Fill the blankets for the MLP : "
                                 f"[{pool_hidden_sizes[j]}, -, -, {pool_hidden_sizes[j + 1]}] : ")
                pool_mlp = list(map(int, pool_mlp.split(' ')))
                pool_mlp.insert(0, pool_hidden_sizes[j])
                pool_mlp.append(pool_hidden_sizes[j + 1])
                pool_mlp_batch_norm = input("Does the MLP have batch normalization ? (y/n) ")
                pool_mlp_batch_norm = True if pool_mlp_batch_norm == 'y' else False
                pool_mlp_dropout = input("Does the MLP have dropout ? (y/n) ")
                pool_mlp_dropout = 0.2 if pool_mlp_dropout == 'y' else 0.
                network_architecture[i]['pooling_layers'][j]['mlp'] = {'architecture': pool_mlp,
                                                                       'batch_norm': pool_mlp_batch_norm,
                                                                       'dropout': pool_mlp_dropout}
            elif pool_layer in ['GCNConv', 'SAGEConv']:
                network_architecture[i]['pooling_layers'][j]['layer_type'] = pool_layer
                pool_in_channels = pool_hidden_sizes[j]
                pool_out_channels = pool_hidden_sizes[j + 1]

                pool_batch_norm = input(f"Does the {pool_layer} {j + 1} have batch normalization ? (y/n) ")
                pool_batch_norm = True if pool_batch_norm == 'y' else False
                pool_dropout = input(f"Does the {pool_layer} {j + 1} have dropout ? (y/n) ")
                pool_dropout = 0.2 if pool_dropout == 'y' else 0.

                network_architecture[i]['pooling_layers'][j]['in_channels'] = pool_in_channels
                network_architecture[i]['pooling_layers'][j]['out_channels'] = pool_out_channels
                network_architecture[i]['pooling_layers'][j]['batch_norm'] = pool_batch_norm
                network_architecture[i]['pooling_layers'][j]['dropout'] = pool_dropout
            else:
                raise Exception('[!] Wrong layer name...')
    else:
        raise Exception('[!] Wrong layer name...')

# print(json.dumps(network_architecture, sort_keys=False, indent=4))
with open(f'config_files/{config_name}.json', 'w') as outfile:
    json.dump(network_architecture, outfile, sort_keys=False, indent=4)
