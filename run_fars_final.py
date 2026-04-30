import argparse
import builtins
import datetime
import gc
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from train_eval import train_epochs
from model_fars_final import FARSRadialFrequencyFinal
from util_functions import (
    MyDataset,
    format_association_retain_tag,
    load_k_fold,
    normalize_association_retain_ratio,
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def configure_reproducibility(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def extract_subgraph(split_data, fold_idx, seed_value, data_name, hop, preprocess_workers, preprocess_chunksize, association_retain_ratio=1.0):
    if data_name == 'Gdataset':
        print('Using Gdataset with 10% testing...')
    elif data_name == 'Cdataset':
        print('Using Cdataset with 10% testing...')
    else:
        print('Using LRSSL with 10% testing...')

    (
        adj_train, train_labels, train_u_indices, train_v_indices,
        test_labels, test_u_indices, test_v_indices,
    ) = split_data

    val_test_appendix = str(fold_idx) + '_kfold'
    seed_appendix = 'seed_{}'.format(seed_value)
    cache_parts = ['data', data_name]
    if association_retain_ratio < 1.0:
        cache_parts.append(format_association_retain_tag(association_retain_ratio))
    cache_parts.extend([seed_appendix, val_test_appendix])

    train_indices = (train_u_indices, train_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    train_file_path = os.path.join(*(cache_parts + ['train']))
    test_file_path = os.path.join(*(cache_parts + ['test']))

    train_graph = MyDataset(
        train_file_path,
        adj_train,
        train_indices,
        train_labels,
        hop,
        preprocess_workers=preprocess_workers,
        preprocess_chunksize=preprocess_chunksize,
    )
    test_graph = MyDataset(
        test_file_path,
        adj_train,
        test_indices,
        test_labels,
        hop,
        preprocess_workers=preprocess_workers,
        preprocess_chunksize=preprocess_chunksize,
    )
    return train_graph, test_graph


def main():
    parser = argparse.ArgumentParser(description='FARS final 20260321 release runner')
    parser.add_argument('--dataset-list', nargs='*', default=['Gdataset', 'Cdataset', 'lrssl'])
    parser.add_argument('--seed-list', type=int, nargs='*', default=[20, 34, 42, 43, 61, 70, 83, 1024, 2014, 2047])
    parser.add_argument('--association-retain-ratio', type=float, default=1.0)
    parser.add_argument('--fold-start', type=int, default=0)
    parser.add_argument('--num-folds', type=int, default=10)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--preprocess-workers', type=int, default=2)
    parser.add_argument('--preprocess-chunksize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--dropout_n', type=float, default=0.4)
    parser.add_argument('--dropout_e', type=float, default=0.1)
    parser.add_argument('--valid_interval', type=int, default=1)
    parser.add_argument('--attention-type', default='gatv2', choices=['gat', 'gatv2', 'gin'])
    parser.add_argument('--radial-layers', type=int, default=2)
    parser.add_argument('--disable-pin-memory', action='store_true', default=False)
    parser.add_argument('--force-undirected', action='store_true', default=False)
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine'])
    parser.add_argument('--scheduler-t-max', type=int, default=0)
    parser.add_argument('--scheduler-eta-min', type=float, default=1e-6)
    parser.add_argument('--loss-type', default='bce', choices=['bce', 'focal', 'asl'])
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--asl-gamma-neg', type=float, default=4.0)
    parser.add_argument('--asl-gamma-pos', type=float, default=1.0)
    parser.add_argument('--asl-clip', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    args.model_type = 'final'
    args.association_retain_ratio = normalize_association_retain_ratio(args.association_retain_ratio)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configure_reproducibility(args.seed)

    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    retain_tag = format_association_retain_tag(args.association_retain_ratio)
    log_name = f'{ts}_{args.model_type}_release_{retain_tag}_fold{args.num_folds}_epoch{args.epochs}.txt'
    log_path = os.path.join(logs_dir, log_name)

    old_stdout = sys.stdout
    with open(log_path, 'w', encoding='utf-8') as log_fp:
        sys.stdout = TeeStream(old_stdout, log_fp)
        try:
            print(args)
            print('log file', log_path)

            for data_name in args.dataset_list:
                print('################ dataset=', data_name, '################')
                print('################ retain_ratio=', '{:.0%}'.format(args.association_retain_ratio), '################')
                for seed in args.seed_list:
                    configure_reproducibility(seed)
                    print('============= seed=', str(seed), '==================')
                    split_data_dict = load_k_fold(
                        data_name,
                        seed,
                        association_retain_ratio=args.association_retain_ratio,
                    )

                    fold_stop = builtins.min(10, args.fold_start + args.num_folds)
                    for k in range(args.fold_start, fold_stop):
                        args.current_seed = seed
                        args.current_fold = k
                        print('------------ fold', str(k + 1), '--------------')
                        train_graphs, test_graphs = extract_subgraph(
                            split_data_dict[k],
                            k,
                            seed,
                            data_name,
                            args.hop,
                            args.preprocess_workers,
                            args.preprocess_chunksize,
                            association_retain_ratio=args.association_retain_ratio,
                        )

                        model = FARSRadialFrequencyFinal(
                            train_graphs,
                            latent_dim=[256, 128, 64],
                            dropout_n=args.dropout_n,
                            dropout_e=args.dropout_e,
                            force_undirected=args.force_undirected,
                            radial_layers=args.radial_layers,
                            attention_type=args.attention_type,
                        )

                        print('Used #train graphs: %d, #test graphs: %d' % (len(train_graphs), len(test_graphs)))
                        train_epochs(train_graphs, test_graphs, model, args)

                        del model
                        del train_graphs
                        del test_graphs
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        finally:
            sys.stdout = old_stdout


if __name__ == '__main__':
    main()




