import os
import os.path
from libs.utils import run_mRec_eval


def eval(gt_filepath, proposal_filepath, score_thresh=0.0, split='validation'):
    print(f'{proposal_filepath} - {score_thresh}')

    tiou_thresholds = [0.5]
    dataset_name = 'thumos14'
    _, _, results_dict = run_mRec_eval(gt_filepath, proposal_filepath, tiou_thresholds, score_thresh, dataset_name, num_workers=8, split=split)
    
    R1x = results_dict['R@1x']
    R5x = results_dict['R@5x']
    R100 = results_dict['R@100']
    R300 = results_dict['R@300']
    R1000 = results_dict['R@1000']
    
    # print(f"R@1x: {R1x:.3f}")
    # print(f"R@5x: {R5x:.3f}")
    # print(f"R@100: {R100:.3f}")
    # print(f"R@300: {R300:.3f}")
    # print(f"R@1000: {R1000:.3f}")

if __name__ == '__main__':
    gt_train_filepath = './data/thumos14/annotations/training_tal.json'
    gt_val_filepath = './data/thumos14/annotations/validation_tal.json'
    gt_K400_train_filepath = './data/thumos14/annotations/training_K400_tal.json'
    gt_K400_val_filepath = './data/thumos14/annotations/validation_K400_tal.json'
    gt_nonK400_train_filepath = './data/thumos14/annotations/training_nonK400_tal.json'
    gt_nonK400_val_filepath = './data/thumos14/annotations/validation_nonK400_tal.json'
    
    # gtad_clip_prop = '/root/code/gtad-zs/output/F16_w16F_s4F_clip_all_lr2e-5/prop_result_all.json'
    # gtad_vificlip_prop = '/root/code/gtad-zs/output/F16_w16F_s4F_ep10_all_lr2e-5/prop_result_all.json'
    # AF_clip_prop = './ckpt/cls_agnostic/thumos_CLIP_prop_all_1/proposal_validation_tal/thumos_CLIP_prop_all_1_epoch_035.json'
    # AF_vificlip_prop = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_all_1/proposal_validation_tal/thumos_ViFiCLIP_prop_all_1_epoch_035.json'
    
    # print("EVAL CLIP")
    # eval(gtad_clip_prop)
    # eval(AF_clip_prop)
    
    # print("EVAL VIFICLIP")
    # eval(gtad_vificlip_prop)
    # eval(AF_vificlip_prop)
    
    
    AF_K400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_K400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_nonK400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_nonK400_train_prop_ep30 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_030.json'
    split='training'
    tgt_gt_train_filepath = gt_K400_train_filepath
    tgt_AF_train_prop = AF_K400_train_prop_ep35
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.0)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.01)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.05)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.1)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.2)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.3)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.4)
    # eval(tgt_gt_train_filepath, tgt_AF_train_prop, split=split, score_thresh=0.5)
    
    # tgt_filepath1 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_th-0.0.json'
    # tgt_filepath2 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_th-0.01.json'
    # tgt_filepath3 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_th-0.05.json'
    # tgt_filepath4 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_th-0.1.json'
    # tgt_filepath5 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_th-0.2.json'
    # eval(gt_nonK400_train_filepath, tgt_filepath1, split=split)
    # eval(gt_nonK400_train_filepath, tgt_filepath2, split=split)
    # eval(gt_nonK400_train_filepath, tgt_filepath3, split=split)
    # eval(gt_nonK400_train_filepath, tgt_filepath4, split=split)
    # eval(gt_nonK400_train_filepath, tgt_filepath5, split=split)
    
    tgt_filepath1 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_top-1.json'
    tgt_filepath2 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_top-5.json'
    tgt_filepath3 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_top-10.json'
    tgt_filepath4 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_top-50.json'
    tgt_filepath5 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels/training_T-K400_E-nonK400_top-100.json'
    eval(gt_nonK400_train_filepath, tgt_filepath1, split=split)
    eval(gt_nonK400_train_filepath, tgt_filepath2, split=split)
    eval(gt_nonK400_train_filepath, tgt_filepath3, split=split)
    eval(gt_nonK400_train_filepath, tgt_filepath4, split=split)
    eval(gt_nonK400_train_filepath, tgt_filepath5, split=split)