import os
import os.path
import pandas as pd
import os.path as osp
from libs.utils import run_mRec_eval, run_mAP_eval

def eval_mAP(dataset_name, gt_filepath, pred_filepath, score_thresh=0.0, tgt_cls_arr=None, split='validation'):
    print(f'{pred_filepath} - {score_thresh}')

    mAPs = run_mAP_eval(gt_filepath, pred_filepath, tiou_thresholds, score_thresh, dataset_name,
                     num_workers=8, split=split, tgt_cls_arr=tgt_cls_arr)

    return mAPs

if __name__ == '__main__':
    # tgt_dataset_name = 'anet13'
    tgt_dataset_name = 'thumos14'
    # tgt_dataset_name = 'fineaction'
    # tiou_thresholds = [0.5]
    tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    gt_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_tal.json'
    gt_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_tal.json'
    gt_K400_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_K400_tal.json'
    gt_K400_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_K400_tal.json'
    gt_nonK400_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_nonK400_tal.json'
    gt_nonK400_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_nonK400_tal.json'
    k400_overlap_filepath = f'/root/datasets/{tgt_dataset_name}/annotations/{tgt_dataset_name}_labels_overlapK400.csv'

    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0'
    # pred_filepath = osp.join(pred_root_dir, 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_epoch_035.json')
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_all/thumos14_vifi_prop_all_0'
    # pred_filepath = osp.join(pred_root_dir, 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_all_0_epoch_035.json')
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_cls_all/thumos14_vifi_all_0'
    # pred_filepath = osp.join(pred_root_dir, 'proposal_cls_validation_tal_ema/thumos14_vifi_all_0_epoch_035.json')
    pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_PL_K400/thumos14_vifi_prop_nonK400_PL_1_load_as_ema'
    pred_filepath = osp.join(pred_root_dir, 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_nonK400_PL_1_load_as_ema_epoch_015.json')
    metric_filepath = pred_filepath.replace('proposal_', 'metric_').replace('.json', '.csv')

    eval_k400_overlap = True
    eval_k400_non_overlap = False
    tgt_cls_arr = None
    overlap_text = 'all'
    if eval_k400_overlap or eval_k400_non_overlap:
        k400_overlap_df = pd.read_csv(k400_overlap_filepath)
        tgt_cls_arr = k400_overlap_df['k400_overlap'].values
    if eval_k400_overlap:
        overlap_text = 'all_K400'
        metric_filepath = metric_filepath.replace('.csv', f'_{overlap_text}.csv')
    if eval_k400_non_overlap:
        tgt_cls_arr = ~tgt_cls_arr
        overlap_text = 'all_nonK400'
        metric_filepath = metric_filepath.replace('.csv', f'_{overlap_text}.csv')


    mAPs = eval_mAP(tgt_dataset_name, gt_val_filepath, pred_filepath, tgt_cls_arr=tgt_cls_arr)

    results_dict = {'split_name': overlap_text,
                **{f"mAP@{tiou}": mAPs[i] for i, tiou in enumerate(tiou_thresholds)},
                'mAP@avg': mAPs.mean()
                }

    print(f"Save --- {metric_filepath}")
    pd.DataFrame(results_dict, index=[0]).to_csv(metric_filepath, index=False)
