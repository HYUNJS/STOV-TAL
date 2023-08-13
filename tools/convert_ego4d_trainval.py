import os
import json
import os.path as osp

import torch
import numpy as np

"""
Directory structure before processing:

This folder
│  convert_ego4d_trainval.py
│  ego4d_label_map.txt
│  ... 
│
└───features
│    └───slowfast8x8_r100_k400
│    └───omnivore_video_swinl
│
└───annotations
│    └───moments_train.json
│    └───moments_val.json
│  ...
"""

def main(video_feat_dir, clip_feat_savedir, annot_out_path=None):

    # clip size / stride in feature extraction
    clip_size = 32
    stride = 16
    os.makedirs(clip_feat_savedir, exist_ok=True)

    with open(train_annot_path, 'r') as f:
        train_videos = json.load(f)['videos']
    with open(val_annot_path, 'r') as f:
        val_videos = json.load(f)['videos']
    videos = train_videos + val_videos

    label_map = dict()
    with open(label_map_path, 'r') as f:
        lines = [l.strip().split('\t') for l in f.readlines()]
        for v, k in lines:
            label_map[k] = int(v)

    database = dict()

    # parse video annotations
    for video in videos:
        vid = video['video_uid']
        print('Processing video {:s} ...'.format(vid))
        subset = video['split']
        if subset == 'train':
            subset = 'training'
        elif subset == 'val':
            subset = 'validation'

        # load video features
        video_feat_path = os.path.join(video_feat_dir, vid + '.pt')
        # skip video if feature does not exist
        if not os.path.exists(video_feat_path):
            print(f'> {video_feat_dir} feature missing')
        video_feat = torch.load(video_feat_path).numpy()
        
        # slowfast_path = os.path.join(slowfast_dir, vid + '.pt')
        # omnivore_path = os.path.join(omnivore_dir, vid + '.pt')
        # # skip video if feature does not exist
        # if not os.path.exists(slowfast_path):
        #     print('> slowfast feature missing')
        # if not os.path.exists(omnivore_path):
        #     print('> omnivore feature missing')
        #     continue
        # slowfast_video = torch.load(slowfast_path).numpy()
        # omnivore_video = torch.load(omnivore_path).numpy()

        # parse clip annotations
        clips = video['clips']
        for clip in clips:
            cid = clip['clip_uid']
            ss = max(float(clip['video_start_sec']), 0)
            es = float(clip['video_end_sec'])
            sf = max(int(clip['video_start_frame']), 0)
            ef = int(clip['video_end_frame'])
            duration = es - ss      # clip length in second
            frames = ef - sf        # clip length in frame
            fps = frames / duration
            if fps < 10 or fps > 100:
                continue
            
            # align event onsets and offsets with feature grid
            prepend_frames = sf % stride
            prepend_sec = prepend_frames / fps
            duration += prepend_sec
            frames += prepend_frames

            append_frames = append_sec = 0
            if (frames - clip_size) % stride:
                append_frames = stride - (frames - clip_size) % stride
                append_sec = append_frames / fps
                duration += append_sec
                frames += append_frames

            # save clip features
            si = (sf - prepend_frames) // stride
            ei = (ef + append_frames - clip_size) // stride
            if ei > len(video_feat):
                raise ValueError(f'end index exceeds {video_feat_dir} feature length')
            # if ei > len(slowfast_video):
            #     raise ValueError('end index exceeds slowfast feature length')
            # if ei > len(omnivore_video):
            #     raise ValueError('end index exceeds omnivore feature length')
            
            clip_feat = video_feat[si:ei]
            np.save(
                os.path.join(clip_feat_savedir, cid + '.npy'), 
                clip_feat.astype(np.float32),
            )
            # slowfast_clip = slowfast_video[si:ei]
            # omnivore_clip = omnivore_video[si:ei]
            # np.save(
            #     os.path.join(slowfast_out_dir, cid + '.npy'), 
            #     slowfast_clip.astype(np.float32),
            # )
            # np.save(
            #     os.path.join(omnivore_out_dir, cid + '.npy'), 
            #     omnivore_clip.astype(np.float32),
            # )
            
            annotations = []

            # parse annotations from different annotators
            annotators = clip['annotations']
            for annotator in annotators:
                
                # parse action items
                items = annotator['labels']
                for item in items:
                    # skip items not from primary categories
                    if not item['primary']:
                        continue
                    
                    ssi = item['video_start_time'] - ss + prepend_sec
                    esi = item['video_end_time'] - ss + prepend_sec
                    sfi = item['video_start_frame'] - sf + prepend_frames
                    efi = item['video_end_frame'] - sf + prepend_frames
                    
                    # filter out very short actions
                    if esi - ssi < 0.25:
                        continue
                    
                    label = item['label']
                    annotations += [{
                        'label': label,
                        'segment': [round(ssi, 2), round(esi, 2)],
                        # 'segment(frames)': [sfi, efi],
                        'label_id': label_map[label],
                    }]

            if len(annotations) == 0:
                continue

            database[cid] = {
                'subset': subset,
                'duration': round(duration, 2),
                'fps': round(fps, 2),
                'annotations': annotations,
            }

    if annot_out_path is not None:
        out = {'version': 'v1', 'database': database}
        with open(annot_out_path, 'w') as f:
            json.dump(out, f)

def pt2npy(egovlp_out_dir):
    fnames = [f for f in os.listdir(egovlp_out_dir+'_pt') if f.endswith('.pt')]
    print(fnames[0])
    for fname in fnames:
        np.save(osp.join(egovlp_out_dir, f"{fname[:-3]}.npy"), torch.load(osp.join(egovlp_out_dir+'_pt', fname)).numpy())

if __name__ == '__main__':
    root_dir = 'data/ego4d/v1/'
    # full-video features downloaded from Ego4D website
    slowfast_dir = osp.join(root_dir, 'video_features/slowfast8x8_r101_k400')
    omnivore_dir = osp.join(root_dir, 'video_features/omnivore_video_swinl')
    egovlp_dir = osp.join(root_dir, 'video_features/egovlp')

    # annotation files downloaded from Ego4D website
    train_annot_path = osp.join(root_dir, 'annotations/moments_train.json')
    val_annot_path = osp.join(root_dir, 'annotations/moments_val.json')

    # label mapping
    label_map_path = osp.join(root_dir, 'ego4d_label_map.txt')

    # where to save the processed features
    slowfast_out_dir = osp.join(root_dir, 'clip_features/slowfast')
    omnivore_out_dir = osp.join(root_dir, 'clip_features/omnivore_video')
    egovlp_out_dir = osp.join(root_dir, 'clip_features/egovlp')
    # os.makedirs(slowfast_out_dir, exist_ok=True)
    # os.makedirs(omnivore_out_dir, exist_ok=True)

    # where to save the processed annotations
    # annot_out_path = 'annotations/ego4d.json'

    # video_feat_dir = egovlp_dir
    # clip_feat_savedir = egovlp_out_dir
    # main(video_feat_dir, clip_feat_savedir)
    pt2npy(egovlp_out_dir)