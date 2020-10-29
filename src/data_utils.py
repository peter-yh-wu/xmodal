import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
import pickle
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def r_at_k(embs1, embs2):
    """embs1->embs2 Retrieval

    R@K is Recall@K (high is good). For median rank, low is good.
    Details described in https://arxiv.org/pdf/1411.2539.pdf.

    Args:
        embs1: embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        embs2: embeddings, np array with shape shape (num_data, emb_dim)

    Return:
        r1: R@1
        r5: R@5
        r10: R@10
        medr: median rank
    """
    ranks = np.zeros(len(embs1))
    mnorms = np.sqrt(np.sum(embs1**2,axis=1)[None]).T
    embs1 = embs1 / mnorms
    tnorms = np.sqrt(np.sum(embs2**2,axis=1)[None]).T
    embs2 = embs2 / tnorms

    for index in range(len(embs1)):
        im = embs1[index].reshape(1, embs1.shape[1])
        d = np.dot(im, embs2.T).flatten()
        inds = np.argsort(d)[::-1]
        rank = 1e20
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, r50, r100, medr)


image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_image(path):
    input_image = Image.open(path)
    input_tensor = image_preprocess(input_image)
    return input_tensor


def np_transform(path):
    return np.load(path)


def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def collate_recipe(batch):
    '''collate function for recipe task'''
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    max_len_2 = 0
    for (_, _, x2, _) in batch:
        max_len_2 = max(len(x2), max_len_2)
    for (x1, y1, x2, y2) in batch:
        x1s.append(x1)
        y1s.append(y1)
        x2 = np.pad(x2, ((0,max_len_2-len(x2)), (0,0)), mode='constant', constant_values=0.0)
        x2s.append(x2)
        y2s.append(y2)
    return torch.tensor(x1s), torch.tensor(y1s), torch.tensor(x2s), torch.tensor(y2s)


class FewShotTwo(Dataset):
    '''
    Dataset for K-shot N-way classification for 2 modalities
    '''
    def __init__(self, x1s, x2s, parent=None, verbose=False):
        self.x1s = x1s
        self.x2s = x2s
        self.parent = parent
        self.verbose = verbose

    def __len__(self):
        return len(self.x1s)

    def __getitem__(self, idx):
        x1 = self.x1s[idx]['x']
        orig_x2 = self.x2s[idx]['x']
        if self.parent.transform_x1 is not None:
            x1 = self.parent.transform_x1(x1)
        if self.parent.transform_x2 is not None:
            x2 = self.parent.transform_x2(orig_x2)
        else:
            x2 = orig_x2
        y1 = self.x1s[idx]['y']
        y2 = self.x2s[idx]['base_idx']
        if self.verbose:
            print('x: ', orig_x2, ', y: ', y1, ', base_idx: ', y2)
        if self.parent.transform_y1 is not None:
            y1 = self.parent.transform_y1(y1)
        if self.parent.transform_y2 is not None:
            y2 = self.parent.transform_y2(y2)
        return x1, y1, x2, y2


class BothDataset(Dataset):
    def __init__(self, grouped_x1s, grouped_x2s, align_idxs=None, \
            transform_x1=None, transform_x2=None, \
            transform_y1=None, transform_y2=None, is_align=True):
        '''
        grouped_xis organized by shared class label
        assumes grouped x1s & grouped x2s are paired
        '''
        if align_idxs is None:
            non_align_idxs = None
        else:
            non_align_idxs = [[] for _ in grouped_x1s]
            for list_i, l in enumerate(grouped_x1s):
                align_idxs_set = set(list(align_idxs[list_i]))
                for i in range(len(l)):
                    if i not in align_idxs_set:
                        non_align_idxs[list_i].append(i)
        self.transform_x1 = transform_x1
        self.transform_x2 = transform_x2
        self.transform_y1 = transform_y1
        self.transform_y2 = transform_y2

        if align_idxs is not None:
            if is_align:
                grouped_x1s = [[l[i] for i in idxs] for idxs, l in zip(align_idxs, grouped_x1s)]
                grouped_x2s = [[l[i] for i in idxs] for idxs, l in zip(align_idxs, grouped_x2s)]
            else:
                grouped_x1s = [[l[i] for i in idxs] for idxs, l in zip(non_align_idxs, grouped_x1s)]
                grouped_x2s = [[l[i] for i in idxs] for idxs, l in zip(non_align_idxs, grouped_x2s)]

        self.x1s = [item for sublist in grouped_x1s for item in sublist]
        self.x2s = [item for sublist in grouped_x2s for item in sublist]
        grouped_ys = [[i]*len(l) for i, l in enumerate(grouped_x1s)]
        self.ys = [item for sublist in grouped_ys for item in sublist]

    def __len__(self):
        return len(self.x1s)
    
    def __getitem__(self, index):
        x1 = self.x1s[index]
        y1 = self.ys[index]
        x2 = self.x2s[index]
        y2 = self.ys[index]
        if self.transform_x1 is not None:
            x1 = self.transform_x1(x1)
        if self.transform_x2 is not None:
            x2 = self.transform_x2(x2)
        if self.transform_y1 is not None:
            y1 = self.transform_y1(y1)
        if self.transform_y2 is not None:
            y2 = self.transform_y2(y2)
        return x1, y1, x2, y2


class AbstractMetaTwo(object):
    '''
    Note only supports get_random_task and not get_task_split
    '''
    def __init__(self, grouped_x1s, grouped_x2s, align_idxs, \
            transform_x1=None, transform_x2=None, \
            transform_y1=None, transform_y2=None):
        '''
        grouped_xis organized by shared class label
        assumes grouped x1s & grouped x2s are paired
        '''
        self.grouped_x1s = grouped_x1s
        print(len(self.grouped_x1s))
        self.grouped_x2s = grouped_x2s
        self.align_idxs = align_idxs
        if self.align_idxs is None:
            self.non_align_idxs = None
        else:
            self.non_align_idxs = [[] for _ in self.grouped_x1s]
            for list_i, l in enumerate(self.grouped_x1s):
                align_idxs_set = set(list(self.align_idxs[list_i]))
                for i in range(len(l)):
                    if i not in align_idxs_set:
                        self.non_align_idxs[list_i].append(i)
        self.transform_x1 = transform_x1
        self.transform_x2 = transform_x2
        self.transform_y1 = transform_y1
        self.transform_y2 = transform_y2

    def __len__(self):
        return len(self.grouped_x1s)

    def __getitem__(self, idx):
        return self.grouped_x1s[idx], self.grouped_x2s[idx]

    def get_random_task(self, N=5, K=1, is_align=True):
        train_task, __ = self.get_random_task_split(N, train_K=K, test_K=0, is_align=is_align)
        return train_task

    def get_random_task_split(self, N=5, train_K=4, test_K=2, is_align=True, verbose=False):
        train_samples1 = []
        test_samples1 = []
        train_samples2 = []
        test_samples2 = []
        character_indices = np.random.choice(len(self), N, replace=False)
        for base_idx, idx in enumerate(character_indices):
            x1s = self.grouped_x1s[idx]
            x2s = self.grouped_x2s[idx]
            if self.align_idxs is None:
                curr_idxs1 = np.random.choice(len(x1s), train_K + test_K, replace=False)
            elif is_align:
                curr_idxs1 = np.random.choice(self.align_idxs[idx], train_K + test_K, replace=False)
            else:
                curr_idxs1 = np.random.choice(self.non_align_idxs[idx], train_K + test_K, replace=False)
            for i, x1_idx in enumerate(curr_idxs1):
                x1 = x1s[x1_idx]
                x2 = x2s[x1_idx]
                new_x1 = {'x':x1, 'y':idx, 'base_idx':base_idx}
                new_x2 = {'x':x2, 'y':idx, 'base_idx':base_idx}
                if i < train_K:
                    train_samples1.append(new_x1)
                    train_samples2.append(new_x2)
                else:
                    test_samples1.append(new_x1)
                    test_samples2.append(new_x2)
        train_task = FewShotTwo(train_samples1, train_samples2, parent=self)
        test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
        return train_task, test_task

    def get_task_split(self, character_indices, all_curr_idxs,
                        new_train_idxs,
                        new_test_idxs,
                        train_K=1, test_K=10, verbose=False):
        train_samples1 = []
        test_samples1 = []
        train_samples2 = []
        test_samples2 = []
        for base_idx, idx in enumerate(character_indices):
            x1s = self.grouped_x1s[idx]
            x2s = self.grouped_x2s[idx]
            
            curr_idxs = all_curr_idxs[base_idx]
            
            for i, x1_idx in enumerate(curr_idxs):
                x1 = x1s[x1_idx]
                x2 = x2s[x1_idx]
                new_x1 = {'x':x1, 'y':idx, 'base_idx':base_idx}
                new_x2 = {'x':x2, 'y':idx, 'base_idx':base_idx}
                if i < train_K:
                    train_samples1.append(new_x1)
                    train_samples2.append(new_x2)
                elif i < train_K+test_K:
                    test_samples1.append(new_x1)
                    test_samples2.append(new_x2)
        train_samples1 = [train_samples1[i] for i in new_train_idxs]
        train_samples2 = [train_samples2[i] for i in new_train_idxs]
        test_samples1 = [test_samples1[i] for i in new_test_idxs]
        test_samples2 = [test_samples2[i] for i in new_test_idxs]
        train_task = FewShotTwo(train_samples1, train_samples2, parent=self)
        test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
        return train_task, test_task


class MetaFolderTwo(AbstractMetaTwo):
    '''dataset for recipe task'''
    def __init__(self, *args, **kwargs):
        data_dir = '../data/recipe'
        fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
        fids = sorted(list(fid_to_label.keys()))
        label_list = sorted(list(set(fid_to_label.values())))
        label_to_int = {}
        for i, label in enumerate(label_list):
            label_to_int[label] = i
        paths = [os.path.join(data_dir, 'new_imgs', '%s.npy' % fid) for fid in fids]
        targets = [label_to_int[fid_to_label[fid]] for fid in fids]
        num_labels = len(np.unique(targets))
        grouped_x1s = [[] for _ in range(num_labels)]
        for path, target in zip(paths, targets):
            grouped_x1s[target].append(path)
        paths = [os.path.join(data_dir, 'text_embs', '%s.npy' % fid) for fid in fids]
        grouped_x2s = [[] for _ in range(num_labels)]
        for path, target in zip(paths, targets):
            grouped_x2s[target].append(path)
        AbstractMetaTwo.__init__(self, grouped_x1s, grouped_x2s, None, *args, **kwargs)


def split_meta_both(all_meta, train=0.8, validation=0.1, test=0.1, seed=0, batch_size=64, num_workers=4, mk_super=False, dept=False, verbose=False, collate_fn=None):
    idx_dir = '../data/recipe/idxs'
    split_idxs_path = os.path.join(idx_dir, 'split_idxs_%d.npy' % seed)

    indices = np.load(split_idxs_path)
    if train >= 0.1:
        n_train = int(train * len(all_meta))
    else:
        n_train = int(0.1 * len(all_meta))
    n_val = int(validation * len(all_meta))
    n_test = int(test * len(all_meta))

    all_train = [all_meta[i] for i in indices[:n_train]]
    if train < 0.1:
        new_train_frac = train/0.1
        all_train = [l[:int(len(l)*new_train_frac)] for l in all_train]
    all_val = [all_meta[i] for i in indices[-(n_val+n_test):-n_test]]
    all_test = [all_meta[i] for i in indices[-n_test:]]
    print(indices[-n_test:])

    train_x1s = [e[0] for e in all_train]
    val_x1s = [e[0] for e in all_val]
    test_x1s = [e[0] for e in all_test]
    train_x2s = [e[1] for e in all_train]
    val_x2s = [e[1] for e in all_val]
    test_x2s = [e[1] for e in all_test]

    train_align_idxs = None
    if not dept:
        if os.path.exists(os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed)):
            train_align_idxs = load_pkl(os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed))
        else:
            train_align_idxs = [list(np.random.choice(len(l), int(len(l)/2), replace=False)) for l in train_x1s]
            save_pkl(train_align_idxs, os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed))

        if verbose:
            num_align = sum([len(l) for l in train_align_idxs])
            num_train = sum([len(l) for l in train_x1s])
            print('num_train_clf: ', num_train-num_align)

    train = AbstractMetaTwo(train_x1s, train_x2s, align_idxs=train_align_idxs,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)
    val = AbstractMetaTwo(val_x1s, val_x2s, align_idxs=None,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)
    test = AbstractMetaTwo(test_x1s, test_x2s, align_idxs=None,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)

    if mk_super:
        dataset = BothDataset(train_x1s, train_x2s, align_idxs=train_align_idxs, \
            transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, \
            transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2, \
            is_align=True)
        super_train = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)

        return train, val, test, super_train
    else:
        return train, val, test
