import json, os
import os.path as osp
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
         for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def score_tiou_dist():
    ## score-tiou distribution
    plt.scatter(pl_scores, pl_tious, s=0.5)
    plt.xlabel('score')
    plt.ylabel('tiou')
    plt.title(f'{tgt_dataset} - split-{split_id} | thresh {score_thresh}')
    plt.savefig(osp.join(save_dir, f'PL_score-tiou_{tgt_dataset}_split-{split_id}.jpg'))
    plt.close()


def score_list_tiou_dist():
    ## score - tiou with score thresh
    score_thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    num_score_thresh = len(score_thresh_list)
    plt.figure(figsize=(num_score_thresh * 2, 6))
    for i, score_thresh in enumerate(score_thresh_list):
        plt.subplot(2, num_score_thresh, i + 1)
        score_mask = pl_scores < score_thresh
        filtered_tiou = pl_tious[score_mask]
        filtered_score = pl_scores[score_mask]
        plt.scatter(filtered_score, filtered_tiou, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('tiou')
            plt.title(f'{tgt_dataset}-s{split_id} | score<{score_thresh}')
        else:
            plt.title(f'score<{score_thresh}')

        plt.subplot(2, num_score_thresh, i + 1 + num_score_thresh)
        score_mask = pl_scores >= score_thresh
        filtered_tiou = pl_tious[score_mask]
        filtered_score = pl_scores[score_mask]
        plt.scatter(filtered_score, filtered_tiou, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('tiou')
            plt.title(f'{tgt_dataset}-s{split_id} | score>{score_thresh}')
        else:
            plt.title(f'score>{score_thresh}')
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f'PL_score_thresh-tiou_{tgt_dataset}_split-{split_id}.jpg'))
    plt.close()


def score_tiou_list_dist():
    ## score - tiou with tiou thresh
    tiou_thresh_list = [0.1, 0.3, 0.5, 0.7]
    num_tiou_thresh = len(tiou_thresh_list)
    for i, tiou_thresh in enumerate(tiou_thresh_list):
        plt.subplot(2, num_tiou_thresh, i + 1)
        tiou_mask = pl_tious < tiou_thresh
        filtered_tiou = pl_tious[tiou_mask]
        filtered_score = pl_scores[tiou_mask]
        plt.scatter(filtered_score, filtered_tiou, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('tiou')
            plt.title(f'{tgt_dataset}-s{split_id} | tiou<{tiou_thresh}')
        else:
            plt.title(f'tiou<{tiou_thresh}')

        plt.subplot(2, num_tiou_thresh, i + 1 + num_tiou_thresh)
        tiou_mask = pl_tious >= tiou_thresh
        filtered_tiou = pl_tious[tiou_mask]
        filtered_score = pl_scores[tiou_mask]
        plt.scatter(filtered_score, filtered_tiou, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('tiou')
            plt.title(f'{tgt_dataset}-s{split_id} | tiou>{tiou_thresh}')
        else:
            plt.title(f'tiou>{tiou_thresh}')
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f'PL_score-tiou_thresh_{tgt_dataset}_split-{split_id}.jpg'))
    plt.close()


def score_duration_dist():
    ## score - duration with tiou thresh
    tiou_thresh_list = [0.1, 0.3, 0.5, 0.7]
    num_tiou_thresh = len(tiou_thresh_list)
    for i, tiou_thresh in enumerate(tiou_thresh_list):
        plt.subplot(2, num_tiou_thresh, i + 1)
        tiou_mask = pl_tious < tiou_thresh
        filtered_duration = pl_duration[tiou_mask]
        filtered_score = pl_scores[tiou_mask]
        plt.scatter(filtered_score, filtered_duration, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('duration')
            plt.title(f'{tgt_dataset}-s{split_id} | tiou<{tiou_thresh}')
        else:
            plt.title(f'tiou<{tiou_thresh}')

        plt.subplot(2, num_tiou_thresh, i + 1 + num_tiou_thresh)
        tiou_mask = pl_tious >= tiou_thresh
        filtered_duration = pl_duration[tiou_mask]
        filtered_score = pl_scores[tiou_mask]
        plt.scatter(filtered_score, filtered_duration, s=0.5)
        if i == 0:
            plt.xlabel('score')
            plt.ylabel('duration')
            plt.title(f'{tgt_dataset}-s{split_id} | tiou>{tiou_thresh}')
        else:
            plt.title(f'tiou>{tiou_thresh}')
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f'PL_score-duration_thresh_{tgt_dataset}_split-{split_id}.jpg'))
    plt.close()


def score_avg_tiou():
    score_list, tiou_list = [], []
    for i in range(10):
        si, ei = i * 0.1, (i + 1) * 0.1
        tgt_mask = np.logical_and(pl_scores.values >= si, pl_scores.values < ei)
        tgt_tious = pl_tious[tgt_mask].values
        if len(tgt_tious) == 0:
            avg_tiou = 0.0
        else:
            avg_tiou = tgt_tious.mean()

        score_list.append((si + ei) / 2)
        tiou_list.append(avg_tiou)

    fig = plt.figure()
    plt.plot(score_list, tiou_list, marker='o')
    plt.xlabel('score')
    plt.ylabel('Avg. tiou')
    plt.title(f"PL quality analysis - Avg. tiou over score interval ({tgt_dataset})")
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f"PL_score-AvgTiou_{tgt_dataset}_split-{split_id}.jpg"))
    plt.close()
    
def score_avg_tiou_v2():
    score_list, tiou_list = [], []
    for i in range(6):
        si, ei = i * 0.1, (i + 1) * 0.1
        tgt_mask = np.logical_and(pl_scores.values >= si, pl_scores.values < ei)
        tgt_tious = pl_tious[tgt_mask].values
        if len(tgt_tious) == 0:
            avg_tiou = 0.0
        else:
            avg_tiou = tgt_tious.mean()

        score_list.append((si + ei) / 2)
        tiou_list.append(avg_tiou)

    fig = plt.figure(figsize=(5, 10))
    plt.plot(score_list, tiou_list, marker='o')
    plt.xlabel('Actionness')
    plt.ylabel('Avg. tIoU')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # plt.title(f"PL quality analysis - Avg. tiou over score interval ({tgt_dataset})")
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f"PL_score-AvgTiou_{tgt_dataset}_split-{split_id}_v2.jpg"), dpi=300)
    plt.close()


def read_data(tgt_dataset, ):
    data_root = f'./data/{tgt_dataset}'
    anno_root = osp.join(data_root, 'annotations')
    gt_anno_filepath = osp.join(anno_root, split_gt_anno_dir, gt_json_file)
    # pl_anno_filepath = osp.join(data_root, 'pseudo_annos_non50', pl_json_file)
    pl_anno_filepath = osp.join(data_root, split_pl_anno_dir, pl_json_file)

    with open(osp.join(gt_anno_filepath), 'r') as fp:
        gt_annos = json.load(fp)['database']
    with open(osp.join(pl_anno_filepath), 'r') as fp:
        pl_annos = json.load(fp)['database']

    tgt_vids = list(gt_annos.keys())
    gt_anno_df_list, pl_anno_df_list = [], []
    for vid in tqdm(tgt_vids):
        gt_anno = gt_annos[vid]['annotations']
        pl_anno = pl_annos[vid]['annotations']
        gt_anno_df = pd.DataFrame(gt_anno)
        pl_anno_df = pd.DataFrame(pl_anno).sort_values(by='score', ascending=False).reset_index(drop=True)
        if len(gt_anno) > 1:
            gt_anno

        gt_anno_df['vid'] = vid
        pl_anno_df['vid'] = vid
        pl_anno_df['duration'] = pl_anno_df['segment'].apply(lambda x: x[1] - x[0])

        gt_segment = np.stack(gt_anno_df['segment'].values)
        pl_segment = np.stack(pl_anno_df['segment'].values)
        num_pl = len(pl_segment)

        # ## filter out the highly overlapped predictions
        # tiou_arr_pl2pl = k_segment_iou(pl_segment, pl_segment)  # [#pl, #pl]
        # pl2pl_duplicate_mask = tiou_arr_pl2pl >= pl_tiou_thresh
        # pl2pl_duplicate_mask[np.arange(num_pl), np.arange(num_pl)] = False
        # dup_idxs = np.stack(np.nonzero(pl2pl_duplicate_mask), axis=1)
        # pl_wo_dup_mask = np.ones(num_pl).astype(bool)
        # pl_wo_dup_mask[dup_idxs[dup_idxs[:, 1] > dup_idxs[:, 0], 1]] = False
        # # if (pl_wo_dup_mask == False).sum() > 0:
        # #     pl_wo_dup_mask
        # pl_segment = pl_segment[pl_wo_dup_mask]
        # pl_anno_df = pl_anno_df[pl_wo_dup_mask].reset_index(drop=True)

        tiou_arr = k_segment_iou(pl_segment, gt_segment)  # [#pl, #gt]

        tious = tiou_arr.max(axis=1)
        pl_anno_df['tiou'] = tious
        # tious_idx = tiou_arr.argmax(axis=1)

        gt_anno_df_list.append(gt_anno_df)
        pl_anno_df_list.append(pl_anno_df)
        # pl_anno_df_list.append(pl_anno_df[['vid', 'score', 'tiou', 'duration']])

    gt_all_df = pd.concat(gt_anno_df_list).reset_index(drop=True)
    pl_all_df = pd.concat(pl_anno_df_list).reset_index(drop=True)
    gt_all_df.columns = [f"{col}_gt" if col != 'vid' else col for col in gt_all_df.columns]
    pl_all_df = pl_all_df.drop(columns=['label_id', 'segment'])
    pl_scores = pl_all_df['score']
    pl_tious = pl_all_df['tiou']
    pl_duration = pl_all_df['duration']
    
    return pl_scores, pl_tious, pl_duration

if __name__ == '__main__':
    # tgt_dataset = 'anet13'
    # tgt_dataset = 'thumos14'
    tgt_dataset = 'fineaction'
    split_id = 0
    score_thresh = 0.01
    # pl_tiou_thresh = 0.5
    # save_root = f'eda/eda_PLv1_th{pl_tiou_thresh}'
    # save_root = f'eda/eda_PL'
    save_root = f'eda/eda_PL_TH-FA'
    save_dir = osp.join(save_root, f"th{score_thresh}")
    os.makedirs(save_dir, exist_ok=True)

    # split_gt_anno_dir = 'train_50_test_50'
    # split_pl_anno_dir = 'PL_non50_vifi'
    # split_pl_anno_dir = 'pseudo_annos_non50'
    # gt_json_file = f'training_non50-{split_id}_tal.json'
    # pl_json_file = f"T-50-{split_id}_E-non50-{split_id}_th-0.01_CLS-ep10.json"
    # pl_json_file = f"T-50-{split_id}_E-non50-{split_id}_th-{score_thresh}.json"

    split_gt_anno_dir = ''
    split_pl_anno_dir = 'PL_nonK400_vifi'
    gt_json_file = f'training_nonK400_tal.json'
    # pl_json_file = f"T-K400_E-nonK400_th-0.05.json"
    pl_json_file = f"T-K400_E-nonK400_th-0.01.json"

    pl_scores, pl_tious, pl_duration

    ## save plot
    score_tiou_dist()
    score_tiou_list_dist()
    score_duration_dist()
    score_list_tiou_dist()
    score_avg_tiou()

    ## tmp

    pl_duration
    # pl_gt_all_df = pl_all_df.merge(gt_all_df, on='vid')
