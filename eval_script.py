import os, argparse
import pandas as pd
import os.path as osp
from libs.utils import run_mRec_eval, run_mAP_eval

def eval_mAP(dataset_name, gt_filepath, pred_filepath, score_thresh=0.0, tgt_cls_arr=None, split='validation'):
    print(f'{pred_filepath} - {score_thresh}')
    mAPs = run_mAP_eval(gt_filepath, pred_filepath, tiou_thresholds, score_thresh, dataset_name,
                     num_workers=8, split=split, tgt_cls_arr=tgt_cls_arr)
    return mAPs

def eval_mRec(dataset_name, gt_filepath, pred_filepath, score_thresh=0.0, split='validation'):
    mRecs = run_mRec_eval(gt_filepath, pred_filepath, tiou_thresholds, score_thresh, dataset_name,
                    num_workers=8, split=split)
    return mRecs

def eval_and_save(dataset, gt_filepath, pred_filepath, tgt_cls_arr, tiou_thresholds):
    mAPs = eval_mAP(dataset, gt_filepath, pred_filepath, tgt_cls_arr=tgt_cls_arr)
    results_dict = {'split_name': overlap_text,
                **{f"mAP@{tiou}": mAPs[i] for i, tiou in enumerate(tiou_thresholds)},
                'mAP@avg': mAPs.mean()
                }
    print(results_dict)
    print(f"Save --- {metric_filepath}")
    os.makedirs(osp.dirname(metric_filepath), exist_ok=True)
    pd.DataFrame(results_dict, index=[0]).to_csv(metric_filepath, index=False)


if __name__ == '__main__':
    '''
        ##### Cross category evaluation example: trained on fineaction wo ST
        pred_dataset="fineaction"
        ckpt_dir="ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1/"
        pred_filename_all="proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
        pred_filename_k400="proposal_CLIP_cls_FA_validation_K400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
        pred_filename_nonk400="proposal_CLIP_cls_FA_validation_nonK400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
        python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset all
        python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset k400
        python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset nonk400
        python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_k400} --pred_subset k400 --eval_subset all
        python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_nonk400} --pred_subset nonk400 --eval_subset all
    '''
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pred_dataset", choices=["thumos14", "anet13", "fineaction", "uk600"], required=True)
    argparser.add_argument("--pred_subset", choices=["all", "k400", "nonk400"], required=True)
    argparser.add_argument("--eval_subset", choices=["all", "k400", "nonk400"], required=False, default="all")
    argparser.add_argument("--ckpt_dir", required=True, help="predicted model's ckpt dir")
    argparser.add_argument("--pred_filename", required=True, help="path relative ckpt_dir")
    args = argparser.parse_args()

    dataset = args.pred_dataset
    pred_subset = args.pred_subset
    eval_subset = args.eval_subset
    ckpt_dir = args.ckpt_dir
    pred_filename = args.pred_filename

    ## Set GT filepath based on the predefined pattern
    gt_all_train_filepath = f'./data/{dataset}/annotations/training_tal.json'
    gt_all_val_filepath = f'./data/{dataset}/annotations/validation_tal.json'
    gt_K400_train_filepath = f'./data/{dataset}/annotations/training_K400_tal.json'
    gt_K400_val_filepath = f'./data/{dataset}/annotations/validation_K400_tal.json'
    gt_nonK400_train_filepath = f'./data/{dataset}/annotations/training_nonK400_tal.json'
    gt_nonK400_val_filepath = f'./data/{dataset}/annotations/validation_nonK400_tal.json'
    k400_overlap_filepath = f'./data/{dataset}/annotations/{dataset}_labels_overlapK400.csv'

    if pred_subset == "all":
        gt_filepath = gt_all_val_filepath
    elif pred_subset == "k400":
        gt_filepath = gt_K400_val_filepath
    elif pred_subset == "nonk400":
        gt_filepath = gt_nonK400_val_filepath
    
    ## Set tiou threshold set
    tiou_thresholds_set_1 = [0.3, 0.4, 0.5, 0.6, 0.7]
    tiou_thresholds_set_2 = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    tiou_thresholds_set_mapper = {
        "thumos14": tiou_thresholds_set_1,
        "anet13": tiou_thresholds_set_2,
        "fineaction": tiou_thresholds_set_2,
        "uk600": tiou_thresholds_set_2,
    }
    tiou_thresholds = tiou_thresholds_set_mapper[dataset]
    # tiou_thresholds = [0.5]

    ## Set filepath
    pred_filepath = osp.join(ckpt_dir, pred_filename)
    metric_filename = pred_filename.replace('proposal_', 'metric_').replace('.json', '.csv')
    metric_filepath = osp.join(ckpt_dir, metric_filename)

    ## Set target eval subset metadata
    tgt_cls_arr = None
    overlap_text = 'all'
    if eval_subset != "all":
        k400_overlap_df = pd.read_csv(k400_overlap_filepath)
        tgt_cls_arr = k400_overlap_df['k400_overlap'].values
        overlap_text = 'all_K400'
        if eval_subset == "nonk400":
            tgt_cls_arr = ~tgt_cls_arr
            overlap_text = 'all_nonK400'
        metric_filepath = metric_filepath.replace('.csv', f'_{overlap_text}.csv')

    eval_and_save(dataset, gt_filepath, pred_filepath, tgt_cls_arr, tiou_thresholds)
