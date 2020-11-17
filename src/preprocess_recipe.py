import argparse
import json
import numpy as np
import os
import pickle
import re
import torch
import shutil
import random

from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from data_utils import transform_image


def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser('preprocess recipe')
    parser.add_argument('--min-data', type=int, help='min data', default=100)
    args = parser.parse_args()

    min_data = args.min_data

    data_dir = '../data/recipe'
    fid_to_label_path = os.path.join(data_dir, 'fid_to_label.pkl')
    fid_to_text_path = os.path.join(data_dir, 'fid_to_text.pkl')

    if True:
        # if not os.path.exists(fid_to_label_path) and not os.path.exists(fid_to_text_path):
        metadata_dir = os.path.join(data_dir, 'metadata')
        metadata_files = os.listdir(metadata_dir)
        metadata_files = [f for f in metadata_files if f.endswith('.json')]
        metadata_paths = [os.path.join(metadata_dir, f) for f in metadata_files]

        fid_to_label = {}
        fid_to_text = {}
        label_dict = {}
        for path_i, p in enumerate(metadata_paths):
            fid = metadata_files[path_i][4:-5]
            with open(p, 'r') as inf:
                metadata = json.load(inf)
            cuisines = metadata['attributes']['cuisine']
            cuisine = '_'.join(sorted(cuisines))
            courses = metadata['attributes']['course']
            course = '_'.join(sorted(courses))
            label = cuisine+'_'+course
            fid_to_label[fid] = label
            fid_to_text[fid] = ', '.join(metadata['ingredientLines'])
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        
        new_label_dict = {}
        num_data = 0
        label_counts = []
        for label in label_dict:
            if label_dict[label] >= min_data:
                new_label_dict[label] = label_dict[label]
                num_data += label_dict[label]
                label_counts.append(label_dict[label])
        print('%d Classes' % len(new_label_dict))
        print('%d Datapoints' % num_data)
        mean_labels = np.mean(label_counts)
        std_labels = np.std(label_counts)
        print('original mean labels per class:  %d' % mean_labels)
        print('original stdev labels per class: %d' % std_labels)

        label_set = set(new_label_dict.keys())
        curr_label_counts = {}
        max_labels = mean_labels+std_labels
        # max_labels = min(max_labels, 8*min_data)
        print('capping at %d' % max_labels)

        fids = sorted(list(fid_to_label.keys()))
        random.Random(0).shuffle(fids)

        new_fid_to_label = {}
        new_fid_to_text = {}
        for fid in fids:
            label = fid_to_label[fid]
            # if label in label_set and new_label_dict[label] <= max_labels:
            if label in label_set and (label not in curr_label_counts or curr_label_counts[label] <= max_labels):
                if label in curr_label_counts:
                    curr_label_counts[label] += 1
                else:
                    curr_label_counts[label] = 1
                new_fid_to_label[fid] = label
                new_fid_to_text[fid] = fid_to_text[fid]
        new_label_counts = sorted(list(curr_label_counts.values()))
        print('now %d Classes' % len(new_label_counts))
        print('now %d Datapoints' % np.sum(new_label_counts))
        print('new mean labels per class:  %d' % np.mean(new_label_counts))
        print('new stdev labels per class: %d' % np.std(new_label_counts))

        save_pkl(new_fid_to_label, fid_to_label_path)
        save_pkl(new_fid_to_text, fid_to_text_path)

    label2int_path = os.path.join(data_dir, 'label2int.pkl')
    if True:
        # if not os.path.exists(label2int_path):
        fid_to_label = load_pkl(fid_to_label_path)
        labels = sorted(list(set(fid_to_label.values())))
        label2int = {label: i for i, label in enumerate(labels)}
        save_pkl(label2int, label2int_path)

    text_emb_dir = os.path.join(data_dir, 'text_embs')
    if os.path.exists(text_emb_dir):
        shutil.rmtree(text_emb_dir)
    if True:
        # if not os.path.exists(text_emb_dir):
        os.makedirs(text_emb_dir)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model = model.cuda()
        model.eval()
        fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
        with torch.no_grad():
            for fid in tqdm(fid_to_text):
                text = fid_to_text[fid]
                text = text.lower()
                tokenized_text = tokenizer.tokenize(text)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens[:512]])
                tokens_tensor = tokens_tensor.cuda()
                outputs = model(tokens_tensor)
                encoded_layers = outputs[0][0] # (seq len, 768)
                emb_path = os.path.join(text_emb_dir, '%s.npy' % fid)
                np.save(emb_path, encoded_layers.cpu().numpy())

    new_img_dir = os.path.join(data_dir, 'new_imgs')
    if os.path.exists(new_img_dir):
        shutil.rmtree(new_img_dir)
    if True:
        # if not os.path.exists(new_img_dir):
        os.makedirs(new_img_dir)
        fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fids = sorted(list(fid_to_label.keys()))
        paths = [os.path.join(data_dir, 'images', 'img%s.jpg' % fid) for fid in fids]
        with torch.no_grad():
            for path in tqdm(paths):
                new_path = path.replace('img', '').replace('jpg', 'npy').replace('images', 'new_imgs')
                x = transform_image(path)
                x = x.numpy()
                np.save(new_path, x)

if __name__ == '__main__':
    main()
