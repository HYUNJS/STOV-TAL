import json, os
import pandas as pd
import os.path as osp
import numpy as np
from tqdm import tqdm

def get_vid_featlen_df(tgt_feat_rootdir):
    vid_featlen_list = []
    for feat_fn in tqdm(os.listdir(tgt_feat_rootdir)):
        feat_filepath = osp.join(tgt_feat_rootdir, feat_fn)
        feat_len = len(np.load(feat_filepath, mmap_mode='r'))
        vid = feat_fn[:-4]
        vid_featlen_list.append([vid, feat_len])
    
    df = pd.DataFrame(vid_featlen_list, columns=['video_id', 'feat_len'])
    
    return df
    
max_feat_num = 2304 * 28
uk600_feat_rootdir = '/root/datasets/uk600_feats/vificlip/F16_w16F_s4F_ep10_train/'
vinfo_ori_filepath = '/root/datasets/uk600/vinfo_training_ori.csv'
vinfo_ori = pd.read_csv(vinfo_ori_filepath)
vid_featlen_df = get_vid_featlen_df(uk600_feat_rootdir)
vinfo_with_featlen = vinfo_ori.merge(vid_featlen_df, on='video_id')

new_vinfo = vinfo_with_featlen[vinfo_with_featlen['feat_len'] <= max_feat_num].reset_index(drop=True)
new_vinfo.to_csv('/root/datasets/uk600/vinfo_training.csv', index=False)

# vinfo_with_featlen[vinfo_with_featlen['feat_len'] > max_feat_num].reset_index(drop=True)

# fineaction_feat_rootdir = '/root/datasets/fineaction_feats/vificlip/F16_w16F_s4F_ep10_trainval/'
# FA_vid_featlen_df = get_vid_featlen_df(fineaction_feat_rootdir)