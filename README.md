# FARS

FARS: Frequency-Guided Radial Subgraph Learning for Drug Repositioning

# Requirments

```text
python==3.10
torch>=2.0
torch-geometric>=2.6
torch-scatter
torch-sparse
numpy
scipy
pandas
networkx
tqdm
scikit-learn
```

Note: If there is a problem with `torch-sparse` or `torch-scatter` installation, please install the wheel version matching your PyTorch/CUDA from:
https://pytorch-geometric.com/whl/

# Datasets

- Gdataset
- Cdataset
- LRSSL

Place dataset files under `data/` with the same format as the original project.

# How to use

```bash
pip install -r requirements.txt
python run_fars_final.py
```

Example:

```bash
python run_fars_final.py --dataset-list lrssl --seed-list 2047 --fold-start 0 --num-folds 10 --epochs 60 --batch-size 64 --test-batch-size 512 --preprocess-workers 2 --attention-type gatv2 --radial-layers 2
```

