import pickle, os, json
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm


label_name_mapper = {
    'BaseballPitch': 'baseball pitch',
    'BasketballDunk': 'basketball dunk',
    'Billiards': 'billiards',
    'CleanAndJerk': 'clean and jerk',
    'CliffDiving': 'cliff diving',
    'CricketBowling': 'cricket bowling',
    'CricketShot': 'cricket shot',
    'Diving': 'diving',
    'FrisbeeCatch': 'frisbee catch',
    'GolfSwing': 'golf swing',
    'HammerThrow': 'hammer throw',
    'HighJump': 'high jump',
    'JavelinThrow': 'javelin throw',
    'LongJump': 'long jump',
    'PoleVault': 'pole vault',
    'Shotput': 'shot put',
    'SoccerPenalty': 'soccer penalty',
    'TennisSwing': 'tennis swing',
    'ThrowDiscus': 'throw discus',
    'VolleyballSpiking': 'volleyball spiking',
}

def save_split_overlap_mask(dataset, split_dirname, split_idx):
    split_perct = split_dirname.split('_')[1]
    label_dirpath = f'/root/datasets/{dataset}/annotations'
    split_filepath = f'/root/datasets/{dataset}/{split_dirname}/split_{split_idx}.list'
    tgt_label_filepath = osp.join(label_dirpath, f'{dataset}_labels.csv')
    save_label_filepath = osp.join(label_dirpath, split_dirname, f'{dataset}_labels_overlap_{split_perct}-{split_idx}.csv')
    os.makedirs(osp.dirname(save_label_filepath), exist_ok=True)
    tgt_labels = pd.read_csv(tgt_label_filepath)

    with open(split_filepath, 'r') as fp:
        split_cls_names = fp.read().splitlines()

    if dataset == 'anet13':
        split_cls_names_mapped = split_cls_names
    elif dataset == 'thumos14':
        split_cls_names_mapped = [label_name_mapper[c] for c in split_cls_names]
    else:
        raise NotImplementedError(f'{dataset} class mapping is not implemented')

    matched_bool_list = []
    for i in range(len(tgt_labels)):
        matched_bool = tgt_labels.loc[i, 'name'] in split_cls_names_mapped
        matched_bool_list.append(matched_bool)
    tgt_labels['split_overlap'] = matched_bool_list
    tgt_labels.to_csv(save_label_filepath, index=False)

def get_split_overlap_labels(dataset, split_dirname, split_idx, save_flag=False):
    split_perct = split_dirname.split('_')[1]
    label_dirpath = f'/root/datasets/{dataset}/annotations'
    label_filepath = osp.join(label_dirpath, split_dirname, f'{dataset}_labels_overlap_{split_perct}-{split_idx}.csv')
    split_overlap_label_filepath = osp.join(label_dirpath, split_dirname, f'{dataset}_{split_perct}-{split_idx}_overlap_labels.csv')
    split_nonoverlap_label_filepath = osp.join(label_dirpath, split_dirname, f'{dataset}_{split_perct}-{split_idx}_nonoverlap_labels.csv')

    ## read label file
    label_df = pd.read_csv(label_filepath)

    ## split masking
    split_overlap_mask = label_df['split_overlap']
    split_overlap_labels = label_df[split_overlap_mask].reset_index(drop=True)
    split_nonoverlap_labels = label_df[~split_overlap_mask].reset_index(drop=True)
    split_overlap_labels['ori_id'] = split_overlap_labels['id']
    split_nonoverlap_labels['ori_id'] = split_nonoverlap_labels['id']
    split_overlap_labels['id'] = split_overlap_labels.index.values
    split_nonoverlap_labels['id'] = split_nonoverlap_labels.index.values

    if save_flag:
        ## save new label files
        split_overlap_labels.to_csv(split_overlap_label_filepath, index=False)
        split_nonoverlap_labels.to_csv(split_nonoverlap_label_filepath, index=False)

    return split_overlap_labels, split_nonoverlap_labels

def split_tal_annotations(dataset, split_dirname, split_idx, subset):
    split_perct = split_dirname.split('_')[1]
    split_name = f'{split_perct}-{split_idx}'
    label_dirpath = f'/root/datasets/{dataset}/annotations'
    label_filepath = osp.join(label_dirpath, f'{subset}_tal.json')
    split_overlap_annos_filepath = osp.join(label_dirpath, split_dirname, f'{subset}_{split_name}_tal.json')
    split_nonoverlap_annos_filepath = osp.join(label_dirpath, split_dirname, f'{subset}_non{split_name}_tal.json')
    split_both_annos_filepath = osp.join(label_dirpath, split_dirname, f'{subset}_half{split_name}_tal.json')

    with open(label_filepath, 'r') as fp:
        tal_annos = json.load(fp)['database']

    split_overlap_labels, split_nonoverlap_labels = get_split_overlap_labels(dataset, split_dirname, split_idx)
    split_overlap_label_ids, split_nonoverlap_label_ids = split_overlap_labels['ori_id'].values, split_nonoverlap_labels['ori_id'].values

    def switch_label_id(tgt_annos, labels_df):
        for anno in tgt_annos:
            new_label_id = labels_df[labels_df['ori_id'] == anno['label_id']]['id'].item()
            anno['label_id'] = new_label_id

    both_labels = {}
    overlap_annos, nonoverlap_annos, both_annos = {}, {}, {}
    for vid in tal_annos.keys():
        annos = tal_annos[vid]['annotations']
        label_ids = [anno['label_id'] for anno in annos]
        _overlap_mask = np.array([l in split_overlap_label_ids for l in label_ids])
        _nonoverlap_mask = np.array([l in split_nonoverlap_label_ids for l in label_ids])
        overlap_mask = _overlap_mask.any()
        nonoverlap_mask = _nonoverlap_mask.any()

        if nonoverlap_mask and overlap_mask:
            both_labels[vid] = np.unique([anno['label_name'] for anno in annos]).tolist()
            both_annos[vid] = tal_annos[vid]
        elif overlap_mask:
            switch_label_id(tal_annos[vid]['annotations'], split_overlap_labels)
            overlap_annos[vid] = tal_annos[vid]
        elif nonoverlap_mask:
            switch_label_id(tal_annos[vid]['annotations'], split_nonoverlap_labels)
            nonoverlap_annos[vid] = tal_annos[vid]
        else:
            raise NotImplementedError("unexpected case")

    with open(split_overlap_annos_filepath, 'w') as fp:
        json.dump({'database': overlap_annos}, fp)
    with open(split_nonoverlap_annos_filepath, 'w') as fp:
        json.dump({'database': nonoverlap_annos}, fp)
    with open(split_both_annos_filepath, 'w') as fp:
        json.dump({'database': both_annos}, fp)

if __name__ == '__main__':
    dataset_name = 'thumos14'
    # dataset_name = 'anet13'
    split_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    root_dir = f'/root/datasets/{dataset_name}/'
    split50_dirname = 'train_50_test_50'
    split75_dirname = 'train_75_test_25'

    # tgt_split_name = split50_dirname
    # split_idx = 0

    # for tgt_split_name in [split50_dirname, split75_dirname]:
    #     for split_idx in split_idxs:
    #         save_split_overlap_mask(dataset_name, tgt_split_name, split_idx)
    #         get_split_overlap_labels(dataset_name, tgt_split_name, split_idx, save_flag=True)
    #         split_tal_annotations(dataset_name, tgt_split_name, split_idx, 'training')
    #         split_tal_annotations(dataset_name, tgt_split_name, split_idx, 'validation')

    # tgt_split_name = split50_dirname
    tgt_split_name = split75_dirname
    for split_idx in tqdm(split_idxs):
        save_split_overlap_mask(dataset_name, tgt_split_name, split_idx)
        get_split_overlap_labels(dataset_name, tgt_split_name, split_idx, save_flag=True)
        split_tal_annotations(dataset_name, tgt_split_name, split_idx, 'training')
        split_tal_annotations(dataset_name, tgt_split_name, split_idx, 'validation')