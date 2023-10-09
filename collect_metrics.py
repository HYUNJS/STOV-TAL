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

def get_split_val(epochs):
    tgt_subset = 'validation'
    df_all = []
    for epoch in epochs:
        ep_filename = f'epoch_{epoch:03d}.csv'
        tgt_split = f'non{split}'
        df = get_one_split_metric(tgt_split, ep_filename, epoch, tgt_subset)
        df_all.append(df)
        tgt_split = f'{split}'
        df = get_one_split_metric(tgt_split, ep_filename, epoch, tgt_subset)
        df_all.append(df)

    val_df = pd.concat(df_all)

    return val_df

def get_split_train(epoch):
    tgt_subset = 'training'
    tgt_split = f'non{split}'
    ep_filename = f'epoch_{epoch:03d}.csv'
    train_df = get_one_split_metric(tgt_split, ep_filename, epoch, tgt_subset)

    return train_df

def get_k400_val(epochs):
    tgt_subset = 'validation'
    df_all = []
    for epoch in epochs:
        ep_filename = f'epoch_{epoch:03d}.csv'
        tgt_split = f'non{split}'
        df = get_one_k400_metric(tgt_split, ep_filename, epoch, tgt_subset)
        df_all.append(df)
        tgt_split = f'{split}'
        df = get_one_k400_metric(tgt_split, ep_filename, epoch, tgt_subset)
        df_all.append(df)

    val_df = pd.concat(df_all)

    return val_df

def get_k400_train(epoch):
    tgt_subset = 'training'
    tgt_split = f'non{split}'
    ep_filename = f'epoch_{epoch:03d}.csv'
    train_df = get_one_k400_metric(tgt_split, ep_filename, epoch, tgt_subset)

    return train_df


def eda_split_all():
    epochs = [25, 30, 35, 40, 45]
    val_df = get_split_val(epochs)
    # train_df = get_split_train()
    df_gb = val_df.groupby(['epoch', 'split'])
    # df_gb.get_group((15, 'non50')).mean()
    df_gb

def eda_k400():
    epoch = 45
    # train_df = get_k400_train(epoch)
    val_df = get_k400_val(epoch)
    val_df


if __name__ == '__main__':
    dataset_opts = ['thumos14', 'anet13']
    model_opts = ['clip', 'vifi']
    split_opts = ['50', '75', 'k400']
    dataset_short_dict = {'anet13': 'AN', 'thumos14': 'TH'}
    # dataset = 'thumos14'
    dataset = 'anet13'

    # model = 'clip'
    model = 'vifi'
    # load_ema = False
    load_ema = True

    # split = '50'
    split = '75'
    # split = 'k400'
    use_PL = False

    dataset_short = dataset_short_dict[dataset]
    if use_PL:
        tgt_dir = f'ckpt/{dataset_short}_agn_PL_{split}'
    else:
        tgt_dir = f'ckpt/{dataset_short}_agn_{split}'

    # ckpt_ver = '1'
    # ckpt_ver = '2_ema'
    # ckpt_ver = '0_ema'
    ckpt_ver = '3'
    # ckpt_ver = '0_ema_th0.05'
    # epoch = 35
    epoch = 15
    topk = 1

    tgt_subset = 'validation'
    tgt_split = f"non{split}"
    # tgt_split = f"{split}"
    tgt_ep_filename = f'epoch_{epoch:03d}.csv'
    if topk == 1:
        metric_path = f'metric_CLIP_cls_{tgt_subset}_{tgt_split}-{{split_id}}_tal'
    else:
        metric_path = f'metric_CLIP_clsK2_{tgt_subset}_{tgt_split}-{{split_id}}_tal'
    if load_ema:
        metric_path += '_ema'

    df_list = []
    for split_id in range(10):
        if use_PL:
            tgt_cfg = f"{dataset}_{model}_prop_{split}_pseudo_tmpl_split{split_id}_{ckpt_ver}"
        else:
            tgt_cfg = f"{dataset}_{model}_prop_{split}_tmpl_split{split_id}_{ckpt_ver}"
        tgt_metric_path = metric_path.format(split_id=split_id)
        tgt_filepath = osp.join(tgt_dir, tgt_cfg, tgt_metric_path, f"{tgt_cfg}_{tgt_ep_filename}")
        df = pd.read_csv(tgt_filepath)
        df['split_id'] = split_id
        df_list.append(df)
    dfs = pd.concat(df_list).reset_index(drop=True)
    dfs_ = dfs.drop(columns='split_name')
    dfs.loc[len(dfs)] = ['avg', *dfs_.mean().values.tolist()]
    dfs
    # if split == 'k400':
    #     eda_k400()
    # else:
    #     eda_split_all()

    # eda_split_all()

    # ckpt_ver = '0_ema'
    # epochs = [35]
    # val_df_v0 = get_split_val(epochs)
    #
    # ckpt_ver = '2_ema'
    # epochs = [45]
    # val_df_v2 = get_split_val(epochs)
    #
    # ckpt_ver = '3_ema'
    # epochs = [55]
    # val_df_v3 = get_split_val(epochs)
    #
    # val_df_v3
    #
    # val_df = pd.concat([val_df_v0, val_df_v2, val_df_v3])
    # val_df_gb = val_df.groupby(['epoch', 'split'])
    # # val_df_gb.mean()
    #
    # ckpt_ver = '0_ema'
    # train_df_v0 = get_split_train(35)
    # ckpt_ver = '2_ema'
    # train_df_v2 = get_split_train(45)
    # train_df = pd.concat([train_df_v0, train_df_v2])
    # train_df
    # train_df_gb = train_df.groupby(['epoch', 'split'])
