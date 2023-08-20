import os
import os.path as osp
import pandas as pd

def merge_dfs(tgt_dir, eval_split):
    filenames = sorted(os.listdir(tgt_dir))
    print(filenames)

    cfg_names = []
    dfs = []
    for filename in filenames:
        df = pd.read_csv(osp.join(tgt_dir, filename))
        dfs.append(df)
        cfg_names.append(filename.replace('.csv', ''))
    eval_splits = [eval_split] * len(filenames)    
    dfs = pd.concat(dfs).reset_index(drop=True)
    print(cfg_names)    
    print(eval_splits)
    print(dfs)    
    dfs.insert(loc=0, column='cfg', value=cfg_names)
    dfs.insert(loc=1, column='eval-split', value=eval_splits)
    return dfs
    
tgt_dir_vificlip_all = 'ckpt/cls_agnostic/thumos_ViFiCLIP_prop_all_1'
tgt_dir_vificlip_k400 = 'ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1'
tgt_dir_vificlip_nonk400 = 'ckpt/cls_agnostic/thumos_ViFiCLIP_prop_nonK400_1'
tgt_dir_clip_all = 'ckpt/cls_agnostic/thumos_CLIP_prop_all_1'
tgt_dir_clip_k400 = 'ckpt/cls_agnostic/thumos_CLIP_prop_K400_1'
tgt_dir_clip_nonk400 = 'ckpt/cls_agnostic/thumos_CLIP_prop_nonK400_1'
tgt_dir_pseudo1 = 'ckpt/cls_agnostic_pseudo/thumos_ViFiCLIP_prop_pseudo_0-00_1'
tgt_dir_pseudo2 = 'ckpt/cls_agnostic_pseudo/thumos_ViFiCLIP_prop_pseudo_0-01_1'
tgt_dir_pseudo3 = 'ckpt/cls_agnostic_pseudo/thumos_ViFiCLIP_prop_pseudo_0-05_1'
tgt_dir_pseudo4 = 'ckpt/cls_agnostic_pseudo/thumos_ViFiCLIP_prop_pseudo_0-10_1'
tgt_dir_base = tgt_dir_pseudo4

tgt_dir_all = osp.join(tgt_dir_base, 'metric_validation_tal')
tgt_dir_K400 = osp.join(tgt_dir_base, 'metric_validation_K400_tal')
tgt_dir_nonK400 = osp.join(tgt_dir_base, 'metric_validation_nonK400_tal')
save_filepath = f'{tgt_dir_all}.csv'

df_all =merge_dfs(tgt_dir_all, 'all')
df_K400 = merge_dfs(tgt_dir_K400, 'K400')
df_nonK400 = merge_dfs(tgt_dir_nonK400, 'nonK400')
df_merged = pd.concat([df_all, df_K400, df_nonK400]).reset_index(drop=True)
df_merged.to_csv(save_filepath, index=False)
