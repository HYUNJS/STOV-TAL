import os
import os.path as osp
import pandas as pd

def get_one_split_metric(tgt_split, tgt_ep_filename, tgt_ep, tgt_subset):
    metric_path = f'metric_{tgt_subset}_{tgt_split}-{{split_id}}_tal'
    dfs = []
    for split_id in range(10):
        tgt_cfg = f"{dataset}_{model}_prop_{split}_tmpl_split{split_id}_{ckpt_ver}"
        tgt_metric_path = metric_path.format(split_id=split_id)
        tgt_filepath = osp.join(tgt_dir, tgt_cfg, tgt_metric_path, f"{tgt_cfg}_{tgt_ep_filename}")
        df = pd.read_csv(tgt_filepath)
        df['split_id'] = split_id
        dfs.append(df)

    dfs = pd.concat(dfs)
    dfs['epoch'] = tgt_ep
    dfs['split'] = tgt_split
    dfs['subset'] = tgt_subset

    return dfs

def get_one_k400_metric(tgt_split, tgt_ep_filename, tgt_ep, tgt_subset):
    tgt_cfg = f"{dataset}_{model}_prop_{split}_{ckpt_ver}"
    tgt_split_ = tgt_split.replace('k400', 'K400')
    tgt_metric_path = f'metric_{tgt_subset}_{tgt_split_}_tal'
    tgt_filepath = osp.join(tgt_dir, tgt_cfg, tgt_metric_path, f"{tgt_cfg}_{tgt_ep_filename}")
    df = pd.read_csv(tgt_filepath)
    df['epoch'] = tgt_ep
    df['split'] = tgt_split
    df['subset'] = tgt_subset

    return df

def get_all_split():
    df_all = []

    tgt_subset = 'validation'
    tgt_split = f'non{split}'
    df = get_one_split_metric(tgt_split, ep5_filename, 5, tgt_subset)
    df_all.append(df)
    df = get_one_split_metric(tgt_split, ep15_filename, 15, tgt_subset)
    df_all.append(df)

    tgt_subset = 'validation'
    tgt_split = f'{split}'
    df = get_one_split_metric(tgt_split, ep5_filename, 5, tgt_subset)
    df_all.append(df)
    df = get_one_split_metric(tgt_split, ep15_filename, 15, tgt_subset)
    df_all.append(df)

    tgt_subset = 'training'
    tgt_split = f'non{split}'
    train_df = get_one_split_metric(tgt_split, ep15_filename, 15, tgt_subset)

    val_df = pd.concat(df_all)

    return train_df, val_df

def get_all_k400():
    df_all = []

    tgt_subset = 'validation'
    tgt_split = f'non{split}'
    df = get_one_k400_metric(tgt_split, ep15_filename, 15, tgt_subset)
    df_all.append(df)

    tgt_subset = 'validation'
    tgt_split = f'{split}'
    df = get_one_k400_metric(tgt_split, ep15_filename, 15, tgt_subset)
    df_all.append(df)

    tgt_subset = 'training'
    tgt_split = f'non{split}'
    train_df = get_one_k400_metric(tgt_split, ep15_filename, 15, tgt_subset)
    val_df = pd.concat(df_all)

    return train_df, val_df

def eda_split_all():
    train_df, val_df = get_all_split()
    df_gb = val_df.groupby(['epoch', 'split'])
    df_gb.get_group((15, 'non50')).mean()
    df_gb

def eda_k400():
    train_df, val_df = get_all_k400()
    val_df


if __name__ == '__main__':
    dataset_opts = ['thumos14', 'anet13']
    model_opts = ['clip', 'vifi']
    split_opts = ['50', '75', 'k400']
    dataset = 'thumos14'

    # model = 'clip'
    model = 'vifi'

    # split = '50'
    # split = '75'
    split = 'k400'

    tgt_dir = f'ckpt/TH_agn_{split}'
    ckpt_ver = '1'
    ep5_filename = 'epoch_005.csv'
    ep15_filename = 'epoch_015.csv'

    if split == 'k400':
        eda_k400()
    else:
        eda_split_all()