# FARS

Code for **FARS: Frequency-Aware Radial Subgraph Learning for Drug Repositioning**.

## Installation

Use Python 3.10 and install the required packages:

```bash
python -m pip install -r requirements.txt
```

Reference environment: Python 3.10.11, PyTorch 2.6.0, PyTorch Geometric 2.7.0,
and NumPy 2.2.6. Install a CUDA-compatible PyTorch build to use a GPU.
The runner automatically uses CUDA when available, otherwise the CPU.

## Files

```text
FARS/
  run_fars_final.py
  requirements.txt
  code/
    model_fars_final.py
    model_fars_blocks.py
    kafnets.py
    train_eval.py
    util_functions.py
  drug_data/
    Gdataset.mat
    Cdataset.mat
    lrssl.txt
```

Original benchmark files are included in `drug_data/`. Graph caches are
generated in `data/` and training logs in `logs/`. All paths are relative
to the project directory, independently of the current working directory.

## Usage

Check the available arguments:

```bash
python run_fars_final.py --help
```

Run one fold and one epoch on Gdataset:

```bash
python run_fars_final.py --dataset-list Gdataset --seed-list 20 --num-folds 1 --epochs 1
```

Run the full experiment:

```bash
python run_fars_final.py
```

The defaults use Gdataset, Cdataset, and LRSSL; 10 folds; 60 epochs;
2-hop subgraphs; three encoder layers with dimensions 256/128/64;
GATv2; two radial layers; and a 1:1 positive-to-sampled-negative ratio.
The default seeds are `20 34 42 43 61 70 83 1024 2014 2047`.
The full command runs 300 dataset/seed/fold combinations.
Use `lrssl` (lowercase) as the LRSSL dataset argument.
