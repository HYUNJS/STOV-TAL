import os, sys, json, torch, pickle, copy

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats, parse_split_name

@register_dataset("uk600")
class UK600Dataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        train_json_file,       # json file for annotations
        val_json_file,       # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        class_agnostic,
        tiou_thresholds,
        root_dir,
        **kwargs,
    ):
        if json_file == '':
            json_file = train_json_file if 'training' in split else val_json_file
        print(f'split: {split} | filepath: {json_file}')
        self.dataset_name = 'uk600'
        self.dataset_shortname = 'UK600'

        # file path
        feat_exist = os.path.exists(feat_folder)
        json_exist = os.path.exists(json_file)
        assert feat_exist and json_exist, f"Feat: {feat_exist} | json: {json_exist}"
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        ## cls-split
        subset_split_name = parse_split_name(json_file, None)
        self.split_name = subset_split_name.replace('val_', '').replace('train_', '')
        label_filepath = kwargs['label_filepaths'][self.dataset_name][self.split_name]
        self.label_df = pd.read_csv(label_filepath)
        self.cls_name_list = self.label_df['name'].tolist()

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        # self.label_dict = None
        self.crop_ratio = crop_ratio
        self.class_agnostic = class_agnostic
        self.tiou_thresholds = tiou_thresholds
        
        # load database and select the subset
        # dict_db, label_dict = self._load_json_db(self.json_file)
        # # proposal vs action categories
        # assert (num_classes == 1) or (len(label_dict) == num_classes)
        # self.data_list = dict_db
        # self.label_dict = label_dict
        dict_db = self._load_json_db(self.json_file)
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': self.dataset_name,
            'tiou_thresholds': tiou_thresholds,
            # 'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }
        
        ## load proposal filepath if True
        self.load_proposal_result = kwargs.get('load_proposal_result', False)
        if self.load_proposal_result:
            proposal_filepath = kwargs.get('proposal_filepath', '')
            assert proposal_filepath != ''
            
            with open(proposal_filepath, 'r') as fp:
                tgt_results = json.load(fp)['results']
            
            proposals = {}
            # for vid in tgt_results.keys():
            for vid in self.vid_list:
                if vid not in tgt_results or len(tgt_results[vid]) == 0:
                    proposal_per_vid = {'video_id': vid, 
                    'segments': torch.tensor(np.stack([[0.0, 0.1]]), dtype=torch.float32), 
                    'scores': torch.tensor(np.stack([1.0]), dtype=torch.float32)}
                    proposals[vid] = proposal_per_vid
                    continue
                    
                results = tgt_results[vid]
                segm_list = []
                score_list = []
                for row in results:
                    segm_list.append(row['segment'])
                    score_list.append(row['actionness'])

                proposal_per_vid = {'video_id': vid, 
                                    'segments': torch.tensor(np.stack(segm_list), dtype=torch.float32), 
                                    'scores': torch.tensor(np.stack(score_list), dtype=torch.float32)}
                proposals[vid] = proposal_per_vid
            
            self.proposals = proposals

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        num_annos = 0
        vid_list = []
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue
            
            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            
            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segm = act['segment']
                    label_id = 0 if self.class_agnostic else act['label_id']
                    segments.append(segm)
                    labels.append([label_id])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
                num_annos += len(labels)
            else:
                segments = None
                labels = None
            vid_list.append(key)
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )
        print(f"# video: {len(dict_db)} | # annos: {num_annos}")
        self.vid_list = vid_list

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # resize the features if needed
        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    segm_duration = seg[1].item() - seg[0].item()
                    if segm_duration <= 0:
                        continue
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item()) / segm_duration
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        ## add target proposals
        if self.load_proposal_result:
            vid = data_dict['video_id']
            data_dict['proposals'] = copy.deepcopy(self.proposals[vid])

        return data_dict
