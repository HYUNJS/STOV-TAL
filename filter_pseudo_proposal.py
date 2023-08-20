import os, json
import os.path as osp
from libs.utils import run_mRec_eval
import pandas as pd

def filter_props(split, tgt_prop_fliepath, save_path, model_cfg_name, score_thresh, topx, thresh_flag, topx_flag):
    if thresh_flag:
        filter_cfg_name = f'th-{score_thresh}'
    if topx_flag:
        filter_cfg_name = f'top-{topx}'
        
    with open(tgt_prop_fliepath, 'r') as fp:
        results = json.load(fp)['results']
        
    new_results = {}    
    vids = list(results.keys())
    new_annos_num, ori_annos_num = 0, 0
    for vid in vids:
        result_df = pd.DataFrame(results[vid])
        ori_annos_num += len(result_df)
        if thresh_flag:
            result_df = result_df[result_df['score'] >= score_thresh]
        if topx_flag:
            result_df = result_df[0:topx]
        new_result = result_df.to_dict('records')
        new_results[vid] = new_result
        new_annos_num += len(result_df)
    
    new_cfg = f'{split}_{model_cfg_name}_{filter_cfg_name}'
    print(f"{new_cfg} - #ori: {ori_annos_num} | #new: {new_annos_num}")
    save_filepath = osp.join(save_path, f'{new_cfg}.json')
    with open(save_filepath, 'w') as fp:
        json.dump({'results': new_results}, fp)

if __name__ == '__main__':
    gt_train_filepath = './data/thumos14/annotations/training_tal.json'
    gt_val_filepath = './data/thumos14/annotations/validation_tal.json'
    gt_K400_train_filepath = './data/thumos14/annotations/training_K400_tal.json'
    gt_K400_val_filepath = './data/thumos14/annotations/validation_K400_tal.json'
    gt_nonK400_train_filepath = './data/thumos14/annotations/training_nonK400_tal.json'
    gt_nonK400_val_filepath = './data/thumos14/annotations/validation_nonK400_tal.json'
    
    AF_K400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_K400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_nonK400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_nonK400_train_prop_ep30 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_030.json'
    split='training'
    tgt_gt_train_filepath = gt_nonK400_train_filepath
    tgt_AF_train_prop = AF_nonK400_train_prop_ep35
    
    with open(tgt_gt_train_filepath ,'r') as fp:
        tal_annos = json.load(fp)['database']
    
    save_path = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels'
    cfg_name = 'T-K400_E-nonK400'
    
    score_thresh = 0.01
    topx = 5
    thresh_flag = True
    topx_flag = False
    
    # filter_props(AF_nonK400_train_prop_ep35, save_path, cfg_name, score_thresh, topx, thresh_flag, topx_flag)
    # filter_props(AF_nonK400_train_prop_ep35, save_path, cfg_name, score_thresh, topx, thresh_flag, topx_flag)
    
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0.00, topx, True, False)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0.01, topx, True, False)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0.05, topx, True, False)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0.1, topx, True, False)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0.2, topx, True, False)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0, 1, False, True)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0, 5, False, True)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0, 10, False, True)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0, 50, False, True)
    # filter_props(split, AF_nonK400_train_prop_ep35, save_path, cfg_name, 0, 100, False, True)
    
    num_gt = 0
    for vid in tal_annos.keys():
        num_gt += len(tal_annos[vid]['annotations'])
    print(num_gt)