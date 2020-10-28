"""Generates idx_dict.npy & split_idxs.npy

Assumes
    train_K == 5 (can change via argument)
    test_K == 10 (i.e. use all test data)
    N == 5 (number of unique classes in each task)

Chose num_eval_tasks = 8
"""
import argparse
import numpy as np
import os
import pickle
import random
import torch


def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class AbstractMeta(object):

    def __init__(self, grouped_xs):
        '''
        Args:
            grouped_xs: size-num_labels list of np arrays (each a spec)
        '''
        self.grouped_xs = grouped_xs

    def __len__(self):
        return len(self.grouped_xs)

    def __getitem__(self, idx):
        return self.grouped_xs[idx]

    def get_idxs(self, N=5, train_K=1, test_K=10, verbose=False):
        '''
        Args:
            N: number of classes/labels to sample
        
        Return:
            character_indices: classes to use
            all_curr_idxs
            new_train_idxs
            new_test_idxs
        '''
        train_samples = []
        test_samples = []
        character_indices = np.random.choice(len(self), N, replace=False)
        if verbose:
            print(character_indices)

        all_curr_idxs = []
        for base_idx, idx in enumerate(character_indices):
            xs = self.grouped_xs[idx] # list of strings
            
            curr_idxs = np.arange(len(xs))
            np.random.shuffle(curr_idxs)
            all_curr_idxs.append(curr_idxs)
            
            for i, x_idx in enumerate(curr_idxs):
                x = xs[x_idx]
                x = {'character_idx':idx, 'x':x}
                new_x = {}
                new_x.update(x) # used to copy orig x to new mem i think
                new_x['base_idx'] = base_idx
                if i < train_K:
                    train_samples.append(new_x)
                elif i < train_K+test_K:
                    test_samples.append(new_x)

        new_train_idxs = np.arange(len(train_samples))
        np.random.shuffle(new_train_idxs) # need shuffle here bc shuffle=False in eval
        new_test_idxs = np.arange(len(test_samples))
        np.random.shuffle(new_test_idxs)
        if verbose:
            test_samples = [test_samples[i] for i in new_test_idxs]
            print([t['base_idx'] for t in test_samples])

        return character_indices, all_curr_idxs, new_train_idxs, new_test_idxs


class MetaFolder(AbstractMeta):
    '''dataset for recipe task'''
    def __init__(self):
        data_dir = '../data/recipe'
        fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
        fids = sorted(list(fid_to_label.keys()))
        label_list = sorted(list(set(fid_to_label.values())))
        label_to_int = {}
        for i, label in enumerate(label_list):
            label_to_int[label] = i
        paths = [os.path.join(data_dir, 'images', 'img%s.jpg' % fid) for fid in fids]
        targets = [label_to_int[fid_to_label[fid]] for fid in fids]
        num_labels = len(np.unique(targets))
        grouped_x1s = [[] for _ in range(num_labels)]
        for path, target in zip(paths, targets):
            grouped_x1s[target].append(path)
        AbstractMeta.__init__(self, grouped_x1s)


def split_meta(all_meta, train=0.8, validation=0.1, test=0.1, split_idxs_path=None):
    '''
    Split all_meta into 3 meta-datasets of tasks (disjoint characters)
    '''
    if train >= 0.1:
        n_train = int(train * len(all_meta))
    else:
        n_train = int(0.1 * len(all_meta))
    n_val = int(validation * len(all_meta))
    n_test = int(test * len(all_meta))
    indices = np.arange(len(all_meta))
    np.random.shuffle(indices)
    np.save(split_idxs_path, indices)
    
    all_train = [all_meta[i] for i in indices[:n_train]]
    if train < 0.1:
        new_train_frac = train/0.1
        all_train = [l[:int(len(l)*new_train_frac)] for l in all_train]
    all_val = [all_meta[i] for i in indices[-(n_val+n_test):-n_test]]
    all_test = [all_meta[i] for i in indices[-n_test:]]
    print(indices[-n_test:])
    train = AbstractMeta(all_train)
    val = AbstractMeta(all_val)
    test = AbstractMeta(all_test)
    return train, val, test


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--train-shots', default=5, type=int, help='Train shots')
    parser.add_argument('--test-shots', default=10, type=int, help='Train shots')
    parser.add_argument('--train', default=0.8, type=float, help='Percentage of train')
    parser.add_argument('--eval-tasks', default=8, type=int, help='Number of eval tasks')
    parser.add_argument('--verbose', action='store_true', help='', default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    idx_dir = '../data/recipe/idxs'
    if not os.path.exists(idx_dir):
        os.makedirs(idx_dir)
    idx_dict_path = os.path.join(idx_dir, 'idx_dict_%d_%d_%d.npy' % (args.train_shots, args.eval_tasks, args.seed))
    print(idx_dict_path)
    split_idxs_path = os.path.join(idx_dir, 'split_idxs_%d.npy' % args.seed)
    if os.path.exists(split_idxs_path) and os.path.exists(idx_dict_path):
        exit()

    all_meta = MetaFolder()
    meta_train, meta_val, meta_test = split_meta(all_meta, args.train, 0.1, split_idxs_path=split_idxs_path)

    idx_dict = {'meta_train': [], 'meta_val': [], 'meta_test': []}
    for _ in range(args.eval_tasks):
        character_indices, all_curr_idxs, new_train_idxs, new_test_idxs = meta_train.get_idxs(train_K=args.train_shots, test_K=args.test_shots)
        idx_dict['meta_train'].append({
            'character_indices': character_indices,
            'all_curr_idxs': all_curr_idxs,
            'new_train_idxs': new_train_idxs,
            'new_test_idxs': new_test_idxs
        })
        character_indices, all_curr_idxs, new_train_idxs, new_test_idxs = meta_val.get_idxs(train_K=args.train_shots, test_K=args.test_shots)
        idx_dict['meta_val'].append({
            'character_indices': character_indices,
            'all_curr_idxs': all_curr_idxs,
            'new_train_idxs': new_train_idxs,
            'new_test_idxs': new_test_idxs
        })
        character_indices, all_curr_idxs, new_train_idxs, new_test_idxs = meta_test.get_idxs(train_K=args.train_shots, test_K=args.test_shots, verbose=args.verbose)
        idx_dict['meta_test'].append({
            'character_indices': character_indices,
            'all_curr_idxs': all_curr_idxs,
            'new_train_idxs': new_train_idxs,
            'new_test_idxs': new_test_idxs
        })
    np.save(idx_dict_path, idx_dict)


if __name__ == "__main__":
    main()
