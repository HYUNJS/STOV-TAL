import os, json
import pandas as pd
import os.path as osp
from tqdm import tqdm
# import multiprocessing as mp
from joblib import Parallel, delayed

def save_tgt_annos(g_id):
    print(f"Process split {g_id}")
    si, ei = g_id * num_vid_per_group, (g_id+1) * num_vid_per_group
    tgt_vids = curr_vids[si:ei]
    tgt_annos = {}
    last_vid = ''
    try:
        for vid in tqdm(tgt_vids):
            last_vid = vid
            row = vinfo.loc[vinfo['video_id'] == vid]
            if len(row) == 0:
                print(f"No row at {vid}")
                raise Exception("error")
            
            fps, duration = row['fps'].item(), row['duration'].item()
            num_frames = round(fps * duration)
            tgt_annos[vid] = {'duration': duration, 'num_frames': num_frames, 'fps': fps, 'subset': 'training'}
            
    except:
        print(f"error at {last_vid}") 
    
    print(f"Save split {g_id}")
    with open(osp.join(anno_root_dir, f'training_tal_{g_id:02d}.json'), 'w') as fp:
        json.dump({'database': tgt_annos}, fp)
    
vinfo_filepath = '/root/datasets/uk600/vinfo_training.csv'
vinfo = pd.read_csv(vinfo_filepath)

anno_root_dir = '/root/datasets/uk600/annotations'
feat_root_dir = '/root/datasets/uk600/vificlip_feats/F16_w16F_s4F_ep10_train'
# num_vid_per_group = 10000
# num_groups = 24
# num_vid_per_group = 85000
num_vid_per_group = 83100
num_groups = 4

curr_vids = list(vinfo['video_id'])
print(f"Total vids {len(curr_vids)}")
# curr_vids = os.listdir(feat_root_dir)[0:num_vid_per_group*num_groups]
# curr_vids = [fn[:-4] for fn in curr_vids]
# curr_vids

Parallel(n_jobs=16)(delayed(save_tgt_annos)(g_id) for g_id in range(num_groups))

# for g_id in range(num_groups):
#     si, ei = g_id * num_vid_per_group, (g_id+1) * num_vid_per_group
#     tgt_vids = curr_vids[si:ei]
#     tgt_annos = {}
#     for vid in tqdm(tgt_vids):
#         row = vinfo.loc[vinfo['video_id'] == vid]
#         fps, duration = row['fps'].item(), row['duration'].item()
#         num_frames = round(fps * duration)
#         tgt_annos[vid] = {'duration': duration, 'num_frames': num_frames, 'fps': fps, 'subset': 'training'}
    
#     with open(osp.join(anno_root_dir, f'training_tal_{g_id:02d}.json'), 'w') as fp:
#         json.dump(tgt_annos, fp)

