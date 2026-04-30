import gc
import os
import torch
import time
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import multiprocessing as mp

from tqdm import tqdm
from scipy import io
from torch_geometric.data import Data, InMemoryDataset, Dataset
from sklearn.model_selection import KFold

import warnings

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

_SUBGRAPH_AROW = None
_SUBGRAPH_ACOL = None
_SUBGRAPH_HOP = None


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class MyDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, hop, drug_embedding=None, disease_embedding=None,
                 preprocess_workers=None, preprocess_chunksize=64):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.hop = hop
        self.attach_pairs = drug_embedding is not None and disease_embedding is not None
        self.n_drug = None if drug_embedding is None else drug_embedding.shape[0]
        self.n_disease = None if disease_embedding is None else disease_embedding.shape[0]
        self.preprocess_workers = preprocess_workers
        self.preprocess_chunksize = preprocess_chunksize
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._release_preprocess_state()

    def _release_preprocess_state(self):
        self.Arow = None
        self.Acol = None
        self.links = None
        self.labels = None

    @property
    def processed_file_names(self):
        suffix = 'pairs' if self.attach_pairs else 'plain'
        name = 'data_h{}_{}.pt'.format(self.hop, suffix)
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(
            self.Arow, self.Acol, self.links, self.labels, self.hop,
            attach_pairs=self.attach_pairs,
            num_workers=self.preprocess_workers,
            chunksize=self.preprocess_chunksize
        )
        try:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        finally:
            del data_list
            if 'data' in locals():
                del data
            if 'slices' in locals():
                del slices
            gc.collect()


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h, drug_embedding=None, disease_embedding=None):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.n_drug = None if drug_embedding is None else drug_embedding.shape[0]
        self.n_disease = None if disease_embedding is None else disease_embedding.shape[0]

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, g_label, self.h)
        data = construct_pyg_graph(*tmp[0:6], interaction_pair=(tmp[6], tmp[7]))
        if self.n_drug is not None and self.n_disease is not None:
            data.interaction_pairs1 = torch.tensor(i, dtype=torch.long)
            data.interaction_pairs2 = torch.tensor(j, dtype=torch.long)
        return data


def _init_subgraph_worker(Arow, Acol, hop):
    global _SUBGRAPH_AROW, _SUBGRAPH_ACOL, _SUBGRAPH_HOP
    _SUBGRAPH_AROW = Arow
    _SUBGRAPH_ACOL = Acol
    _SUBGRAPH_HOP = hop


def _extract_subgraph_worker(task):
    i, j, g_label = task
    return subgraph_extraction_labeling(
        (i, j), _SUBGRAPH_AROW, _SUBGRAPH_ACOL, g_label, _SUBGRAPH_HOP
    )


def links2subgraphs(Arow, Acol, links, labels, hop, attach_pairs=False, num_workers=None, chunksize=64):
    # Extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    start = time.time()
    if num_workers is None:
        num_workers = max(1, min(8, mp.cpu_count() - 1))
    chunksize = max(1, int(chunksize))
    total_tasks = len(labels)
    task_iter = (
        (int(i), int(j), int(g_label))
        for i, j, g_label in zip(links[0], links[1], labels)
    )

    g_list = []
    if num_workers <= 1:
        extracted_iter = (
            subgraph_extraction_labeling((i, j), Arow, Acol, g_label, hop)
            for i, j, g_label in task_iter
        )
        for tmp in tqdm(extracted_iter, total=total_tasks):
            interaction_pair = (tmp[6], tmp[7]) if attach_pairs else None
            g_list.append(construct_pyg_graph(*tmp[0:6], interaction_pair=interaction_pair))
    else:
        with mp.Pool(
            processes=num_workers,
            initializer=_init_subgraph_worker,
            initargs=(Arow, Acol, hop)
        ) as pool:
            extracted_iter = pool.imap(_extract_subgraph_worker, task_iter, chunksize=chunksize)
            for tmp in tqdm(extracted_iter, total=total_tasks):
                interaction_pair = (tmp[6], tmp[7]) if attach_pairs else None
                g_list.append(construct_pyg_graph(*tmp[0:6], interaction_pair=interaction_pair))

    end = time.time()
    print("Time elapsed for subgraph extraction + graph transform: {}s".format(end - start))

    return g_list


def subgraph_extraction_labeling(ind, Arow, Acol, label=1, h=1):
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h + 1):
        if len(u_fringe) == 0 or len(v_fringe) == 0:
            break

        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited

        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    # Remove link between target nodes
    subgraph[0, 0] = 0
    # Prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    max_node_label = 2 * 8 + 1  # To construct initialize label trick matrix

    return u, v, r, node_labels, max_node_label, label, ind[0], ind[1]


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y, interaction_pair=None):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_attr=edge_type, y=y)
    if interaction_pair is not None:
        i, j = interaction_pair
        data.interaction_pairs1 = torch.tensor(i, dtype=torch.long)
        data.interaction_pairs2 = torch.tensor(j, dtype=torch.long)

    return data


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def row_normalize_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def build_similarity_graph(sim, num_neighbor=5):
    sim = np.asarray(sim)
    if num_neighbor <= 0 or num_neighbor > sim.shape[0]:
        num_neighbor = sim.shape[0]

    neighbor = np.argpartition(-sim, kth=num_neighbor - 1, axis=1)[:, :num_neighbor]
    row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
    col_index = neighbor.reshape(-1)

    edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(np.int64))
    values = torch.from_numpy(sim[row_index, col_index]).float()
    return edge_index, values, (sim.shape[0], sim.shape[0])


class SimilarityData:
    def __init__(self, drug_embedding, disease_embedding, drug_edge, disease_edge):
        self.drug_embedding = drug_embedding
        self.disease_embedding = disease_embedding
        self.drug_edge = drug_edge
        self.disease_edge = disease_edge


class GlobalTopologyData:
    def __init__(self, edge_index, node_type, num_drug, num_disease):
        self.edge_index = edge_index
        self.node_type = node_type
        self.num_drug = int(num_drug)
        self.num_disease = int(num_disease)


class PairDataset(Dataset):
    def __init__(self, pairs, labels, num_drug, num_disease):
        super(PairDataset, self).__init__(root=None)
        self.pairs = torch.as_tensor(pairs, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.n_drug = int(num_drug)
        self.n_disease = int(num_disease)

    def len(self):
        return int(self.labels.numel())

    def get(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        data = Data(y=label.view(1))
        data.interaction_pairs1 = pair[0].view(1)
        data.interaction_pairs2 = pair[1].view(1)
        return data


def build_global_topology_data(adj_matrix):
    drug_idx, disease_idx = ssp.find(adj_matrix)[:2]
    num_drug, num_disease = adj_matrix.shape
    disease_idx = disease_idx + num_drug

    src = np.concatenate([drug_idx, disease_idx])
    dst = np.concatenate([disease_idx, drug_idx])
    edge_index = torch.as_tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    node_type = torch.cat(
        [
            torch.zeros(num_drug, dtype=torch.long),
            torch.ones(num_disease, dtype=torch.long),
        ],
        dim=0,
    )
    return GlobalTopologyData(edge_index, node_type, num_drug, num_disease)


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    # In case some nodes are isolated
    g.add_nodes_from(range(len(data.x)))
    edge_types = {(u, v): data.edge_attr[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g


def normalize_association_retain_ratio(association_retain_ratio):
    ratio = float(association_retain_ratio)
    if ratio > 1.0:
        ratio /= 100.0
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(
            'association_retain_ratio must be in (0, 1] or (0, 100]. Got {}'.format(
                association_retain_ratio
            )
        )
    return ratio


def format_association_retain_tag(association_retain_ratio):
    ratio = normalize_association_retain_ratio(association_retain_ratio)
    return 'retain{:03d}'.format(int(round(ratio * 100)))


def load_k_fold(data_name, seed, with_similarity=False, association_retain_ratio=1.0):
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_root_candidates = [
        os.path.join(root_path, 'drug_data'),
        os.path.join(os.path.dirname(root_path), 'drug_data'),
    ]
    data_root = None
    for candidate in data_root_candidates:
        if os.path.isdir(candidate):
            data_root = candidate
            break
    if data_root is None:
        raise FileNotFoundError(
            "drug_data directory not found. Expected one of: {}".format(
                ", ".join(data_root_candidates)
            )
        )

    drug_sim = disease_sim = None

    if data_name == 'lrssl':
        # txt dataset
        path = os.path.join(data_root, '{}.txt'.format(data_name))
        matrix = pd.read_table(path, index_col=0).values
        if with_similarity:
            raise ValueError('FARS-SA currently supports similarity inputs only for .mat datasets.')
    elif data_name in ['Gdataset', 'Cdataset']:
        path = os.path.join(data_root, '{}.mat'.format(data_name))
        # mat dataset
        data = io.loadmat(path)
        matrix = data['didr'].T
        if with_similarity:
            if 'drug' not in data or 'disease' not in data:
                raise KeyError('Expected drug/disease similarity matrices in {}.'.format(path))
            drug_sim = row_normalize_matrix(data['drug'])
            disease_sim = row_normalize_matrix(data['disease'])
    else:
        # csv dataset
        path = os.path.join(data_root, '{}.csv'.format(data_name))
        data = pd.read_csv(path, header=None)
        matrix = data.values.T
        if with_similarity:
            raise ValueError('FARS-SA currently supports similarity inputs only for .mat datasets.')

    association_retain_ratio = normalize_association_retain_ratio(association_retain_ratio)

    drug_num, disease_num = matrix.shape[0], matrix.shape[1]
    drug_id, disease_id = np.nonzero(matrix)

    total_positive = len(drug_id)
    num_len = int(np.ceil(total_positive * association_retain_ratio))
    if num_len < 10:
        raise ValueError(
            'Retained positive associations ({}) are fewer than 10; cannot run 10-fold CV.'.format(
                num_len
            )
        )

    if num_len < total_positive:
        retain_rng = np.random.default_rng(seed)
        retained_idx = np.sort(retain_rng.permutation(total_positive)[:num_len])
        drug_id = drug_id[retained_idx]
        disease_id = disease_id[retained_idx]

    print(
        'association retain ratio {:.0%}: kept {} / {} positive associations'.format(
            association_retain_ratio,
            len(drug_id),
            total_positive,
        )
    )

    neutral_flag = 0
    labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)
    observed_labels = [1] * len(drug_id)
    labels[drug_id, disease_id] = np.array(observed_labels)
    labels = labels.reshape([-1])

    # Number of test and validation edges
    num_train = int(np.ceil(0.9 * len(drug_id)))
    num_test = int(np.ceil(0.1 * len(drug_id)))
    print("num_train {}".format(num_train),
          "num_test {}".format(num_test))

    print("num_train, num_test's ratio is", 0.9, 0.1)

    # Negative sampling
    neg_drug_idx, neg_disease_idx = np.where(matrix == 0)
    neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])
    np.random.seed(6)
    np.random.shuffle(neg_pairs)
    neg_idx = np.array([dr * disease_num + di for dr, di in neg_pairs])

    # Positive sampling
    pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])
    pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])

    split_data_dict = {}
    count = 0
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_data, test_data in kfold.split(pos_idx):
        # Train dataset contains positive and negative
        idx_pos_train = np.array(pos_idx)[np.array(train_data)]

        idx_neg_train = neg_idx[0:len(idx_pos_train)]  # Training dataset pos:neg = 1:1
        idx_train = np.concatenate([idx_pos_train, idx_neg_train], axis=0)

        pairs_pos_train = pos_pairs[np.array(train_data)]
        pairs_neg_train = neg_pairs[0:len(pairs_pos_train)]
        pairs_train = np.concatenate([pairs_pos_train, pairs_neg_train], axis=0)

        # Test dataset contains positive and negative
        idx_pos_test = np.array(pos_idx)[np.array(test_data)]
        idx_neg_test = neg_idx[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test)]
        idx_test = np.concatenate([idx_pos_test, idx_neg_test], axis=0)

        pairs_pos_test = pos_pairs[np.array(test_data)]
        pairs_neg_test = neg_pairs[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test)]
        pairs_test = np.concatenate([pairs_pos_test, pairs_neg_test], axis=0)

        # Internally shuffle training set
        rand_idx = list(range(len(idx_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_train = idx_train[rand_idx]
        pairs_train = pairs_train[rand_idx]

        u_train_idx, v_train_idx = pairs_train.transpose()
        u_test_idx, v_test_idx = pairs_test.transpose()

        # Create labels
        train_labels = labels[idx_train]
        test_labels = labels[idx_test]

        # Make training adjacency matrix
        rating_mx_train = np.zeros(drug_num * disease_num, dtype=np.float32)
        rating_mx_train[idx_train] = labels[idx_train]
        rating_mx_train = ssp.csr_matrix(rating_mx_train.reshape(drug_num, disease_num))
        split_data_dict[count] = [rating_mx_train, train_labels, u_train_idx, v_train_idx, \
                                  test_labels, u_test_idx, v_test_idx]
        count += 1

    if with_similarity:
        return split_data_dict, drug_sim, disease_sim
    return split_data_dict


if __name__ == '__main__':
    split_data_dict = load_k_fold('Gdataset', 1)

