import os, argparse
import os.path
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

if __name__ == '__main__':
    '''
        python eval_script.py; python eval_script.py --eval_k400_overlap; python eval_script.py --eval_k400_non_overlap;
        python eval_script.py --eval_k400_overlap; python eval_script.py --eval_k400_non_overlap;
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eval_k400_overlap", action="store_true")
    argparser.add_argument("--eval_k400_non_overlap", action="store_true")
    args = argparser.parse_args()
    
    eval_k400_overlap = args.eval_k400_overlap
    eval_k400_non_overlap = args.eval_k400_non_overlap
    
    ## TODO. change dataset name & tiou for your use
    tgt_dataset_name = 'thumos14'
    tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    # tgt_dataset_name = 'anet13'
    # tiou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # tgt_dataset_name = 'fineaction'
    # tiou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # tgt_dataset_name = 'uk600'
    # tiou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    # tiou_thresholds = [0.5]
    
    gt_all_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_tal.json'
    gt_all_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_tal.json'
    gt_K400_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_K400_tal.json'
    gt_K400_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_K400_tal.json'
    gt_nonK400_train_filepath = f'./data/{tgt_dataset_name}/annotations/training_nonK400_tal.json'
    gt_nonK400_val_filepath = f'./data/{tgt_dataset_name}/annotations/validation_nonK400_tal.json'
    k400_overlap_filepath = f'/root/datasets/{tgt_dataset_name}/annotations/{tgt_dataset_name}_labels_overlapK400.csv'

    gt_filepath = gt_all_val_filepath
    # gt_filepath = gt_K400_val_filepath
    # gt_filepath = gt_nonK400_val_filepath
    
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1_effP_vis-1_text-16'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/fineaction_vifi_prop_K400_1_effP_vis-1_text-16_epoch_015.json'
    pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1'
    pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_8_th-0.40_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/FAk400_vifi_prop_k400_UKall-PL_8_th-0.40_load_as_ema_epoch_002.json'
    
    ## effprompt prediction
    # pred_root_dir = f"./ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0_effP_vis-1_text-0"
    # pred_filename = f"proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_effP_vis-1_text-0_epoch_035.json"
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_clip_prop_K400_0_effP_vis-1_text-0'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_clip_prop_K400_0_effP_vis-1_text-0_epoch_035.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_clip_prop_K400_0_effP_vis-1_text-16'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_clip_prop_K400_0_effP_vis-1_text-16_epoch_035.json'
    # # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0_effP_vis-1_text-16'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_effP_vis-1_text-16_epoch_035.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0_effP_vis-0_text-16'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_effP_vis-0_text-16_epoch_035.json'
    
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1_effP_vis-1_text-16'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_K400_1_effP_vis-1_text-16_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0_effP_vis-1_text-16'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/thumos14_vifi_prop_K400_0_effP_vis-1_text-16_epoch_035.json'
    
    
    ## gemini prediction
    # pred_root_dir = f"./gemini_output/{tgt_dataset_name}"
    # pred_filename = f"gemini-1.5-flash-001_time-inst-v0_{tgt_dataset_name}-gen-all_results.json"
    # pred_filename = f"gemini-1.5-flash-001_time-inst-v1_{tgt_dataset_name}-gen-all_results.json"
    # pred_filename = f"gemini-1.5-flash-001_time-inst-v1_{tgt_dataset_name}-con-base_results.json"
    # pred_filename = f"gemini-1.5-flash-001_time-inst-v1_{tgt_dataset_name}-con-novel_results.json"
    # pred_filename = f"gemini-1.5-pro-001_time-inst-v1_{tgt_dataset_name}-gen-all_results.json"
    # pred_filename = f"gemini-1.5-pro-001_time-inst-v1_{tgt_dataset_name}-con-base_results.json"
    # pred_filename = f"gemini-1.5-pro-001_time-inst-v1_{tgt_dataset_name}-con-novel_results.json"
    
    ## thumos14 prediction
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_clip_prop_K400_0'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_clip_prop_K400_0_epoch_035.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_epoch_035.json'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_epoch_035_gemini-vids.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_all/thumos14_vifi_prop_all_0'
    # pred_filepath = osp.join(pred_root_dir, 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_all_0_epoch_035.json')
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_all/thumos14_vifi_prop_all_0'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/thumos14_vifi_prop_all_0_epoch_035.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_cls_all/thumos14_vifi_all_0'
    # pred_filepath = osp.join(pred_root_dir, 'proposal_cls_validation_tal_ema/thumos14_vifi_all_0_epoch_035.json')
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR100k-PL_K400/THk400_vifi_prop_k400_UKR100k-PL_6_th-0.05_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/THk400_vifi_prop_k400_UKR100k-PL_6_th-0.05_load_as_ema_epoch_004.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_7_th-0.10_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/THk400_vifi_prop_k400_UKR10k-PL_7_th-0.10_load_as_ema_epoch_006.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_10_th-0.10_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/THk400_vifi_prop_k400_UKR10k-PL_10_th-0.10_load_as_ema_epoch_007.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_PL_K400/thumos14_vifi_prop_nonK400_PL_1_th-0.05_min-1_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/thumos14_vifi_prop_nonK400_PL_1_th-0.05_min-1_load_as_ema_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_PL_K400/TH_vifi_prop_K400_FA-PL_5_th-0.20_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal_ema/TH_vifi_prop_K400_FA-PL_5_th-0.20_load_as_ema_epoch_010.json'
    
    ## OPENTAL
    # pred_root_dir = '/root/code/actionformer-prop/opental_af_results/eval_cls_with_nms_a-only'
    # pred_filename = 'proposal_CLIP_cls_fusion-a_only_TH_validation_tal_ema/thumos14_vifi_prop_K400_0_ema_v2_epoch_055.json'
    
    ## fineaction prediction
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_all/fineaction_vifi_prop_all_1'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_all_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_fusion-c_only_FA_validation_tal_ema/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_nonK400_PL_5_th-0.30_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH2FA_agn_all/TH2FA_vifi_prop_all_PL_5_th-0.30_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/TH2FA_vifi_prop_all_PL_5_th-0.30_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/opental_eval'
    # pred_filename = 'proposal_CLIP_cls_TH_validation_tal/opental_final.json'
    # pred_root_dir = '/root/code/actionformer-prop/opental_eval'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/opental_epoch-19.json'
    # pred_root_dir = '/root/code/actionformer-prop/opental_TH2FA_eval'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/opental_final.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_viclip-l_prop_K400_1'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_viclip-l_prop_K400_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_clip_prop_K400_1'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/fineaction_clip_prop_K400_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1'
    # pred_filename = 'proposal_CLIP_cls_fusion-c_only_FA_validation_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_K400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015_gemini-vids.json'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_nonK400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015_gemini-vids.json'
    # pred_root_dir = './ckpt/TH_agn_PL_K400/thumos14_vifi_prop_nonK400_PL_1_th-0.05_min-1_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/thumos14_vifi_prop_nonK400_PL_1_th-0.05_min-1_load_as_ema_epoch_015.json'
    
    ## fineaction with random prediction
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKR5k-PL_K400/FAk400_vifi_prop_k400_UKR5k-PL_5_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKR5k-PL_5_th-0.40_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt//FA_agn_UKR10k-PL_K400/FAk400_vifi_prop_k400_UKR10k-PL_5_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKR10k-PL_5_th-0.40_load_as_ema_epoch_010.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.40_load_as_ema_epoch_006.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_6_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKR100k-PL_6_th-0.40_load_as_ema_epoch_004.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKR200k-PL_K400/FAk400_vifi_prop_k400_UKR200k-PL_8_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKR200k-PL_8_th-0.40_load_as_ema_epoch_002.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_8_th-0.40_load_as_ema/'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/FAk400_vifi_prop_k400_UKall-PL_8_th-0.40_load_as_ema_epoch_002.json'
    
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_7_th-0.05_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/THk400_vifi_prop_k400_UKR10k-PL_7_th-0.05_load_as_ema_epoch_006.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_10_th-0.10_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/THk400_vifi_prop_k400_UKR10k-PL_10_th-0.10_load_as_ema_epoch_007.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_UKR100k-PL_K400/THk400_vifi_prop_k400_UKR100k-PL_6_th-0.05_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/THk400_vifi_prop_k400_UKR100k-PL_6_th-0.05_load_as_ema_epoch_004.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/thumos14_vifi_prop_K400_0_epoch_035.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/TH_agn_PL_K400/TH_vifi_prop_K400_FA-PL_5_th-0.20_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_FA_validation_tal_ema/TH_vifi_prop_K400_FA-PL_5_th-0.20_load_as_ema_epoch_010.json'
    
    # ## fineaction pseudo labels
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema/'
    # pred_filename = 'pseudo_labels/training_T-FA-K400_E-FA-nonK400_th-0.10.json'
    
    
    ## anet13 prediction
    # pred_root_dir = '/root/code/actionformer-prop/opental_eval_768'
    # pred_filename = 'proposal_CLIP_cls_AN_validation_tal_ema/opental_final.json'
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/AN_agn_PL_K400/anet13_vifi_prop_nonK400_PL_6_th-0.20_load_as_ema'
    # pred_filename = 'proposal_CLIP_cls_AN_validation_tal_ema/anet13_vifi_prop_nonK400_PL_6_th-0.20_load_as_ema_epoch_010.json'
    
    # ## UK600 prediction
    # pred_root_dir = '/root/code/actionformer-prop/ckpt/AN_agn_all/anet13_vifi_prop_all_3'
    # pred_filename = 'proposal_CLIP_cls_UK600_validation_tal_ema/anet13_vifi_prop_all_3_epoch_015.json'
    
    pred_filepath = osp.join(pred_root_dir, pred_filename)
    metric_filename = pred_filename.replace('proposal_', 'metric_').replace('.json', '.csv')
    metric_filepath = osp.join(pred_root_dir, metric_filename)
    # metric_filepath = pred_filepath.replace('proposal_', 'metric_').replace('.json', '.csv')
    # metric_filepath = pred_filepath.replace('pseudo_labels', 'metric_PL').replace('.json', '.csv')

    tgt_cls_arr = None
    overlap_text = 'all'
    if eval_k400_overlap or eval_k400_non_overlap:
        k400_overlap_df = pd.read_csv(k400_overlap_filepath)
        tgt_cls_arr = k400_overlap_df['k400_overlap'].values
        overlap_text = 'all_K400'
        if eval_k400_non_overlap:
            tgt_cls_arr = ~tgt_cls_arr
            overlap_text = 'all_nonK400'
        metric_filepath = metric_filepath.replace('.csv', f'_{overlap_text}.csv')

    mAPs = eval_mAP(tgt_dataset_name, gt_filepath, pred_filepath, tgt_cls_arr=tgt_cls_arr)

    results_dict = {'split_name': overlap_text,
                **{f"mAP@{tiou}": mAPs[i] for i, tiou in enumerate(tiou_thresholds)},
                'mAP@avg': mAPs.mean()
                }
    print(results_dict)
    print(f"Save --- {metric_filepath}")
    pd.DataFrame(results_dict, index=[0]).to_csv(metric_filepath, index=False)
    print("Save", metric_filepath)
