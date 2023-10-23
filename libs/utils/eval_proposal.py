# Modified from official EPIC-Kitchens action detection evaluation code
# see https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py
import os, pickle, argparse, sys, json
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict
from tqdm import tqdm


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    num_dup = 0
    if len(ants) > 0 and 'label_id' in ants[0]:
        label_key = 'label_id'
    else:
        label_key = 'labelIndex'
    # label_key = 'label_id' if 'label_id' in ants[0] else 'labelIndex'
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event[label_key]
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event[label_key])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
        else:
            num_dup += 1
    # print(f"Remove duplicated annotations: {num_dup} / {len(ants)}")
    return valid_events, num_dup, len(ants)


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    num_dup_all, num_ori_all = {}, {}
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue
        # remove duplicated instances
        # ants, num_dup, num_ori = remove_duplicate_annotations(v['annotations'])
        ants, num_dup, num_ori = remove_duplicate_annotations(v['annotations'], 0)
        num_dup_all[k] = num_dup
        num_ori_all[k] = num_ori
        # video id
        vids += [k] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
    print(f"Total Removed Annotations - {sum(num_dup_all.values())} / {sum(num_ori_all.values())}")
    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    # json_db = json_db['database']
    json_db = json_db['results']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            # scores += [float(event['scores'])]
            scores += [float(event['score'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ant_file,
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_kx=(1, 5, 10),
        top_k=(100, 1000, 10000),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name=None,
        label_filepath=None,
        cls_agnostic_flag=False
    ):

        self.tiou_thresholds = tiou_thresholds
        self.top_kx = top_kx
        self.top_k = top_k
        self.ap = None
        self.num_workers = num_workers
        self.act_id2name = None
        if dataset_name is not None:
            self.dataset_name = dataset_name
            self._get_act_id2name(label_filepath)
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        self.ground_truth = load_gt_seg_from_json(
            ant_file, split=self.split, label=label, label_offset=label_offset)

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label'] = self.ground_truth['label'].replace(self.activity_index)

    def _get_act_id2name(self, label_filepath):
        if label_filepath is None:
            label_filepath = f'./data/{self.dataset_name}/annotations/{self.dataset_name}_labels.csv'
        label_mapper = pd.read_csv(label_filepath)
        self.act_id2name = {row['id']: row['name'] for _, row in label_mapper.iterrows()}

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_kx), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_kx,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]

        return recall

    def wrapper_compute_topk_recall(self, preds):
        """Computes Top-k recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topk_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]

        return recall

    def wrapper_compute_topkx_proposal_recall(self, preds, tiou_thresholds=[0.5], top_kx=(1, 5)):
        return compute_topkx_recall_detection(ground_truth=self.ground_truth, prediction=preds,
                                                tiou_thresholds=tiou_thresholds, top_k=top_kx)

    def wrapper_compute_topk_proposal_recall(self, preds, tiou_thresholds=[0.5], top_k=(50, 100, 300, 1000)):
        return compute_topk_recall_detection(ground_truth=self.ground_truth, prediction=preds,
                                                tiou_thresholds=tiou_thresholds, top_k=top_k)

    def eval_cls_specific(self, preds):
        # compute mAP
        ap = self.wrapper_compute_average_precision(preds)
        recallx = self.wrapper_compute_topkx_recall(preds)
        recall = self.wrapper_compute_topk_recall(preds)

        return ap, recallx, recall

    def eval_cls_agnostic(self, preds):
        ## compute proposal recall
        prop_recallx = self.wrapper_compute_topkx_proposal_recall(preds)
        prop_recall = self.wrapper_compute_topk_proposal_recall(preds)

        return prop_recallx, prop_recall

    def evaluate_proposal(self, preds, verbose=False):
        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })

        prop_recallx, prop_recall = self.eval_cls_agnostic(preds)
        prop_Rxs, prop_Rs = prop_recallx * 100, prop_recall * 100
        if verbose:
            prop_rec1x, prop_rec5x = prop_Rxs[0, 0], prop_Rxs[0, 1]
            prop_rec100, prop_rec300, prop_rec1000 = prop_Rs[0, 1], prop_Rs[0, 2], prop_Rs[0, 3]
            print(f'    pR@1x: {prop_rec1x:.3f} | pR@5x: {prop_rec5x:.3f} | pR@100: {prop_rec100:.3f}'
                f' | pR@300: {prop_rec300:.3f} | pR@1000: {prop_rec1000:.3f} | #preds: {len(preds)}')

        return prop_Rxs, prop_Rs
    
    def evaluate_mAP(self, preds, verbose=False, tgt_cls_arr=None):
        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        ## remap prediction label_id following self.activity_index
        gt_cls_list = list(self.activity_index.keys())
        pred_cls_in_gt_mask = preds['label'].apply(lambda x: x in gt_cls_list)
        preds = preds[pred_cls_in_gt_mask].reset_index(drop=True)
        preds['label'] = preds['label'].replace(self.activity_index)

        ap, recallx, recall = self.eval_cls_specific(preds)
        if tgt_cls_arr is not None:
            self.ap = self.ap[..., tgt_cls_arr]
            self.recallx = self.recallx[..., tgt_cls_arr]
            self.recall = self.recall[..., tgt_cls_arr]
        else:
            self.ap, self.recallx, self.recall = ap, recallx, recall
        mAP = self.ap.mean(axis=1) * 100
        mRecallx = self.recallx.mean(axis=2) * 100
        mRecall = self.recall.mean(axis=2) * 100
        if verbose:
            avg_mAP = mAP.mean()
            for idx, tiou in enumerate(self.tiou_thresholds):
                if tiou == 0.5:
                    mAP_50 = mAP[idx]
            
            print(f'    mAP@0.5: {mAP_50:.3f} | mAP@{self.tiou_thresholds[0]}:{self.tiou_thresholds[-1]}: {avg_mAP:.3f}')

        return mAP, mRecallx, mRecall

    def evaluate(self, preds, verbose=True, tgt_cls_arr=None, eval_agnostic=False):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        ## remap prediction label_id following self.activity_index
        gt_cls_list = list(self.activity_index.keys())
        pred_cls_in_gt_mask = preds['label'].apply(lambda x: x in gt_cls_list)
        preds = preds[pred_cls_in_gt_mask].reset_index(drop=True)
        preds['label'] = preds['label'].replace(self.activity_index)

        # always reset ap
        self.ap = None

        # make the label ids consistent
        preds['label'] = preds['label'].replace(self.activity_index)

        ap, recallx, recall = self.eval_cls_specific(preds)
        if eval_agnostic:
            prop_recallx, prop_recall = self.eval_cls_agnostic(preds)

        self.ap, self.recallx, self.recall = ap, recallx, recall

        if tgt_cls_arr is not None:
            self.ap = self.ap[..., tgt_cls_arr]
            self.recallx = self.recallx[..., tgt_cls_arr]
            self.recall = self.recall[..., tgt_cls_arr]

        mAP = self.ap.mean(axis=1)
        mRecallx = self.recallx.mean(axis=2)
        mRecall = self.recall.mean(axis=2)
        average_mAP = mAP.mean()

        # print results
        if verbose:
            # print the results
            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            num_pred_per_cls = preds[['video-id', 'label']].groupby('label').count()['video-id']
            num_gt_per_cls = self.ground_truth.groupby('label').count()['video-id']
            if self.act_id2name is not None:
                raise Exception("This version is not handling label_id shift due to self.activity_index")
                ## class-wise results
                for idx, tiou in enumerate(self.tiou_thresholds):
                    print('\n[tIoU = {:.2f}] - Class-wise Accuracy'.format(tiou))
                    ap = self.ap[idx] * 100
                    rec_k1x, rec_k5x, rec_k10x = self.recallx[idx, :] * 100
                    rec_k100, rec_k1000, rec_k10000 = self.recall[idx, :] * 100
                    for cls_idx in range(len(ap)):
                        _pred_cnt, _gt_cnt = num_pred_per_cls.get(cls_idx, 0), num_gt_per_cls.get(cls_idx, 0)
                        print(f'{self.act_id2name[cls_idx]} - AP: {ap[cls_idx]:.3f} | R@1x: {rec_k1x[cls_idx]:.3f} | R@5x: {rec_k5x[cls_idx]:.3f}'
                              f' |  R@10x: {rec_k10x[cls_idx]:.3f} | R@100: {rec_k100[cls_idx]:.3f} | R@1000: {rec_k1000[cls_idx]:.3f}'
                              f' | R@10000: {rec_k10000[cls_idx]:.3f} | #gt: {_gt_cnt} | #pred: {_pred_cnt}')
                ## top-5 results
                ascd_idx = np.argsort(ap)
                print(f'\n[High top-5]')
                for cls_idx in ascd_idx[:-6:-1]:
                    _pred_cnt, _gt_cnt = num_pred_per_cls.get(cls_idx, 0), num_gt_per_cls.get(cls_idx, 0)
                    print(
                        f'{self.act_id2name[cls_idx]} - AP: {ap[cls_idx]:.3f} | R@1x: {rec_k1x[cls_idx]:.3f} | R@5x: {rec_k5x[cls_idx]:.3f}'
                        f' |  R@10x: {rec_k10x[cls_idx]:.3f} | R@100: {rec_k100[cls_idx]:.3f} | R@1000: {rec_k1000[cls_idx]:.3f}'
                        f' | R@10000: {rec_k10000[cls_idx]:.3f} | #gt: {_gt_cnt} | #pred: {_pred_cnt}')
                print(f'\n[Low top-5]')
                for cls_idx in ascd_idx[:5]:
                    _pred_cnt, _gt_cnt = num_pred_per_cls.get(cls_idx, 0), num_gt_per_cls.get(cls_idx, 0)
                    print(
                        f'{self.act_id2name[cls_idx]} - AP: {ap[cls_idx]:.3f} | R@1x: {rec_k1x[cls_idx]:.3f} | R@5x: {rec_k5x[cls_idx]:.3f}'
                        f' |  R@10x: {rec_k10x[cls_idx]:.3f} | R@100: {rec_k100[cls_idx]:.3f} | R@1000: {rec_k1000[cls_idx]:.3f}'
                        f' | R@10000: {rec_k10000[cls_idx]:.3f} | #gt: {_gt_cnt} | #pred: {_pred_cnt}')
            ## average results
            block = ''
            for tiou, tiou_mAP, tiou_mRecallx, tiou_mRecall in zip(self.tiou_thresholds, mAP, mRecallx, mRecall):
                block += '\n|tIoU = {:.2f}: '.format(tiou)
                block += 'mAP = {:>4.3f} (%) '.format(tiou_mAP*100)
                for idx, k in enumerate(self.top_kx):
                    block += 'Recall@{:d}x = {:>4.3f} (%) '.format(k, tiou_mRecallx[idx]*100)
                for idx, k in enumerate(self.top_k):
                    block += 'Recall@{:d} = {:>4.3f} (%) '.format(k, tiou_mRecall[idx]*100)
            print(block)
            print('Average mAP: {:>4.3f} (%)'.format(average_mAP*100))
            print(f"#pred: {len(preds)}")
            print(f"#gt: {num_gt_per_cls.sum()}")


        # return the results
        # return mAP, average_mAP, mRecall
        # return average_mAP*100, mRecallx*100, mRecall*100
        if eval_agnostic:
            return mAP*100, mRecallx*100, mRecall*100, prop_recallx*100, prop_recall*100
        else:
            return mAP*100, mRecallx*100, mRecall*100



def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def compute_topkx_recall_detection(
        ground_truth,
        prediction,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_kx_idx],
                                 this_gt[['t-start', 't-end']].values)

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def compute_topk_recall_detection(
        ground_truth,
        prediction,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(100, 1000, 10000),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-k results of a action category where x stands for the number of
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_k_idx = score_sort_idx[:max(top_k)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_k_idx],
                                 this_gt[['t-start', 't-end']].values)

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


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


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def run_mRec_eval(gt_filepath, pred_filepath, tiou_thresholds, thresh, dataset, num_workers=8, split='validation', verbose=True, get_csv=False):
    print(f'Evaluate split - {split}')
    evaluator = ANETdetection(
        gt_filepath,
        split,
        tiou_thresholds=tiou_thresholds,
        dataset_name=dataset, num_workers=num_workers,
        top_k=[50, 100, 300, 1000],
        top_kx=[1, 5],
    )

    if pred_filepath.endswith('.pkl'):
        with open(pred_filepath, 'rb') as fp:
            pred = pickle.load(fp)
        pred_df = pd.DataFrame.from_dict(pred)
    elif pred_filepath.endswith('.json'):
        pred_df = load_pred_seg_from_json(pred_filepath, label='label_id')

    print(f"Threshold {thresh}")
    pred_tgt = pred_df[pred_df['score'] >= thresh]

    prop_Rxs, prop_Rs = evaluator.evaluate_proposal(pred_tgt)
    prop_rec1x = prop_Rxs[0, 0]
    prop_rec5x = prop_Rxs[0, 1]
    prop_rec100 = prop_Rs[0, 1]
    prop_rec300 = prop_Rs[0, 2]
    prop_rec1000 = prop_Rs[0, 3]
    results_dict = {'R@1x': prop_rec1x, 'R@5x': prop_rec5x, 'R@100': prop_rec100, 'R@300':prop_rec300, 'R@1000': prop_rec1000}
    result_in_csv = ','.join([f'{f:.5f}' for f in list(results_dict.values())] + [str(len(pred_tgt))])
    if verbose:
        print(f'    pR@1x: {prop_rec1x:.3f} | pR@5x: {prop_rec5x:.3f} | pR@100: {prop_rec100:.3f}'
              f' | pR@300: {prop_rec300:.3f} | pR@1000: {prop_rec1000:.3f} | #preds: {len(pred_tgt)}')
        print()

    if get_csv:
        return prop_Rxs, prop_Rs, results_dict, result_in_csv
    else:
        return prop_Rxs, prop_Rs, results_dict

def run_mAP_eval(gt_filepath, pred_filepath, tiou_thresholds, thresh, dataset, num_workers=8, split='validation', verbose=True, get_csv=False):
    print(f'Evaluate split - {split}')
    evaluator = ANETdetection(
        gt_filepath,
        split,
        tiou_thresholds=tiou_thresholds,
        dataset_name=dataset, num_workers=num_workers,
        top_k=[100, 300],
        top_kx=[1, 5],
    )

    if pred_filepath.endswith('.pkl'):
        with open(pred_filepath, 'rb') as fp:
            pred = pickle.load(fp)
        pred_df = pd.DataFrame.from_dict(pred)
    elif pred_filepath.endswith('.json'):
        pred_df = load_pred_seg_from_json(pred_filepath, label='label_id')

    print(f"Threshold {thresh}")
    pred_tgt = pred_df[pred_df['score'] >= thresh]

    mAPs, mRxs, mRs = evaluator.evaluate_mAP(pred_tgt)
    print(mAPs)
    # mAP = mAPs[0]
    # print(mAP)

    return mAPs