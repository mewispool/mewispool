#!/usr/bin/env bash

python3 -m venv mewispool_env
source mewispool_env/bin/activate

pip3 install torch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric

python3 graph_classification/train.py
