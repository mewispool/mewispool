#!/usr/bin/env bash

if [ ! -d "mewispool_env" ]; then
  python3 -m venv mewispool_env
  source mewispool_env/bin/activate
  pip3 install torch
  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
  pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
  pip install torch-geometric
fi
source mewispool_env/bin/activate

python3 MIS/train.py
