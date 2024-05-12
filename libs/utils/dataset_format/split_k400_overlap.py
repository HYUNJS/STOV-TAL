import os, json
import os.path as osp
import pandas as pd
import numpy as np

import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
word_lemm = WordNetLemmatizer()

def stem_verb_noun(w):
    return word_lemm.lemmatize(word_lemm.lemmatize(w, pos='v'), pos='n')

def get_label_names(label_df):
    return label_df['name'].apply(lambda x: ' '.join([stem_verb_noun(w.lower()) for w in x.split(' ')])).values

def get_label_by_word(label_df):
    return label_df['name'].apply(lambda x: [stem_verb_noun(w.lower()) for w in x.split(' ')])

def get_k400_overlap_labels(dataset, save_flag=False):

    label_dirpath = f'/root/datasets/{dataset}/annotations'
    label_filepath = osp.join(label_dirpath, f'{dataset}_labels_overlapK400.csv')
    k400_overlap_label_filepath = osp.join(label_dirpath, f'{dataset}_K400_overlap_labels.csv')
    k400_nonoverlap_label_filepath = osp.join(label_dirpath, f'{dataset}_K400_nonoverlap_labels.csv')

    ## read label file
    label_df = pd.read_csv(label_filepath)

    ## k400 masking
    k400_overlap_mask = label_df['k400_overlap']
    k400_overlap_labels = label_df[k400_overlap_mask].reset_index(drop=True)
    k400_nonoverlap_labels = label_df[~k400_overlap_mask].reset_index(drop=True)
    k400_overlap_labels['ori_id'] = k400_overlap_labels['id']
    k400_nonoverlap_labels['ori_id'] = k400_nonoverlap_labels['id']
    k400_overlap_labels['id'] = k400_overlap_labels.index.values
    k400_nonoverlap_labels['id'] = k400_nonoverlap_labels.index.values

    if save_flag:
        ## save new label files
        k400_overlap_labels.to_csv(k400_overlap_label_filepath, index=False)
        k400_nonoverlap_labels.to_csv(k400_nonoverlap_label_filepath, index=False)

    return k400_overlap_labels, k400_nonoverlap_labels

def save_k400_overlap_labels_all():
    save_k400_overlap_mask('thumos14')
    save_k400_overlap_mask('fineaction')
    save_k400_overlap_mask('anet13')
    get_k400_overlap_labels('thumos14', save_flag=True)
    get_k400_overlap_labels('fineaction', save_flag=True)
    get_k400_overlap_labels('anet13', save_flag=True)

def save_k400_overlap_mask(dataset):
    k400_label_filepath = '/root/datasets/k400/annotations/k400_labels.csv'
    label_dirpath = f'/root/datasets/{dataset}/annotations'
    tgt_label_filepath = osp.join(label_dirpath, f'{dataset}_labels.csv')
    save_label_filepath = osp.join(label_dirpath, f'{dataset}_labels_overlapK400.csv')

    k400_labels = pd.read_csv(k400_label_filepath)
    tgt_labels = pd.read_csv(tgt_label_filepath)
    k400_label_names_by_word = get_label_by_word(k400_labels)
    tgt_label_names_by_word = get_label_by_word(tgt_labels)

    matched_bool_list, matched_idx_list, matched_name_list = [], [], []
    for i in range(len(tgt_label_names_by_word)):
        tgt_label_words = tgt_label_names_by_word[i]
        matched_bool, matched_idx, matched_name = False, -1, ''
        for ki, k400_label_words in enumerate(k400_label_names_by_word.values):
            word_mask = np.array(tgt_label_words).reshape(-1, 1) == np.array(k400_label_words).reshape(1, -1)
            if word_mask.any(axis=1).all():
                matched_bool = True
                matched_idx = ki
                matched_name = k400_labels.loc[ki, 'name']
        matched_bool_list.append(matched_bool)
        matched_idx_list.append(matched_idx)
        matched_name_list.append(matched_name)

    tgt_labels['k400_overlap'] = matched_bool_list
    tgt_labels['k400_cls_idx'] = matched_idx_list
    tgt_labels['k400_cls_name'] = matched_name_list

    tgt_labels.to_csv(save_label_filepath, index=False)

def split_tal_annotations(dataset, subset):
    label_dirpath = f'/root/datasets/{dataset}/annotations'
    label_filepath = osp.join(label_dirpath, f'{subset}_tal.json')
    k400_overlap_annos_filepath = osp.join(label_dirpath, f'{subset}_K400_tal.json')
    k400_nonoverlap_annos_filepath = osp.join(label_dirpath, f'{subset}_nonK400_tal.json')
    k400_both_annos_filepath = osp.join(label_dirpath, f'{subset}_halfK400_tal.json')

    with open(label_filepath, 'r') as fp:
        tal_annos = json.load(fp)['database']

    k400_overlap_labels, k400_nonoverlap_labels = get_k400_overlap_labels(dataset)
    # k400_overlap_label_ids, k400_nonoverlap_label_ids = k400_overlap_labels['id'].values, k400_nonoverlap_labels['id'].values
    k400_overlap_label_ids, k400_nonoverlap_label_ids = k400_overlap_labels['ori_id'].values, k400_nonoverlap_labels['ori_id'].values

    def switch_label_id(tgt_annos, labels_df):
        for anno in tgt_annos:
            new_label_id = labels_df[labels_df['ori_id'] == anno['label_id']]['id'].item()
            anno['label_id'] = new_label_id

    both_labels = {}
    overlap_annos, nonoverlap_annos, both_annos = {}, {}, {}
    for vid in tal_annos.keys():
        annos = tal_annos[vid]['annotations']
        label_ids = [anno['label_id'] for anno in annos]
        _overlap_mask = np.array([l in k400_overlap_label_ids for l in label_ids])
        _nonoverlap_mask = np.array([l in k400_nonoverlap_label_ids for l in label_ids])
        overlap_mask = _overlap_mask.any()
        nonoverlap_mask = _nonoverlap_mask.any()

        if nonoverlap_mask and overlap_mask:
            both_labels[vid] = np.unique([anno['label_name'] for anno in annos]).tolist()
            both_annos[vid] = tal_annos[vid]
        elif overlap_mask:
            switch_label_id(tal_annos[vid]['annotations'], k400_overlap_labels)
            overlap_annos[vid] = tal_annos[vid]
        elif nonoverlap_mask:
            switch_label_id(tal_annos[vid]['annotations'], k400_nonoverlap_labels)
            nonoverlap_annos[vid] = tal_annos[vid]
        else:
            raise NotImplementedError("unexpected case")

    with open(k400_overlap_annos_filepath, 'w') as fp:
        json.dump({'database': overlap_annos}, fp)
    with open(k400_nonoverlap_annos_filepath, 'w') as fp:
        json.dump({'database': nonoverlap_annos}, fp)
    with open(k400_both_annos_filepath, 'w') as fp:
        json.dump({'database': both_annos}, fp)

def split_tal_annotations_all():
    split_tal_annotations('thumos14', 'training')
    split_tal_annotations('thumos14', 'validation')
    split_tal_annotations('fineaction', 'training')
    split_tal_annotations('fineaction', 'validation')
    split_tal_annotations('anet13', 'training')
    split_tal_annotations('anet13', 'validation')

if __name__ == '__main__':
    pass

    # save_k400_overlap_labels_all()
    # split_tal_annotations_all()
    # split_tal_annotations('thumos14', 'training')
    # split_tal_annotations('thumos14', 'validation')
    # save_k400_overlap_mask('fineaction')
    # get_k400_overlap_labels('fineaction', save_flag=True)
    # split_tal_annotations('fineaction', 'training')
    # split_tal_annotations('fineaction', 'validation')

    # save_k400_overlap_mask('anet13')
    # get_k400_overlap_labels('anet13', save_flag=True)
    # split_tal_annotations('anet13', 'training')
    # split_tal_annotations('anet13', 'validation')
    
    save_k400_overlap_mask('uk600')
    get_k400_overlap_labels('uk600', save_flag=True)
    split_tal_annotations('uk600', 'validation')
