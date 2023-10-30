import os, json, argparse
import os.path as osp
from libs.utils import run_mRec_eval
import pandas as pd

def filter_props(tgt_prop_fliepath, save_path, model_cfg_name, score_thresh, topx, thresh_flag, topx_flag):
    if thresh_flag:
        filter_cfg_name = f'th-{score_thresh:.2f}'
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
    
    new_cfg = f'{subset}_{model_cfg_name}_{filter_cfg_name}'
    print(f"{new_cfg} - #ori: {ori_annos_num} | #new: {new_annos_num}")
    save_filepath = osp.join(save_path, f'{new_cfg}.json')
    with open(save_filepath, 'w') as fp:
        json.dump({'results': new_results}, fp)

    tiou_thresholds = [0.5]
    thresh = 0.0
    _, _, _, metric_in_csv = run_mRec_eval(tgt_gt_filepath, save_filepath, tiou_thresholds, thresh, dataset, split=subset, get_csv=True)

    return metric_in_csv

def filter_props_all_cfg(tgt_filepath, save_path):
    print(tgt_filepath)
    metric_in_csv1 = filter_props(tgt_filepath, save_path, cfg_name, 0.00, 1, True, False)
    metric_in_csv2 = filter_props(tgt_filepath, save_path, cfg_name, 0.01, 1, True, False)
    metric_in_csv3 = filter_props(tgt_filepath, save_path, cfg_name, 0.05, 1, True, False)
    metric_in_csv4 = filter_props(tgt_filepath, save_path, cfg_name, 0.1, 1, True, False)
    metric_in_csv5 = filter_props(tgt_filepath, save_path, cfg_name, 0.2, 1, True, False)
    metric_in_csv6 = filter_props(tgt_filepath, save_path, cfg_name, 0, 1, False, True)
    metric_in_csv7 = filter_props(tgt_filepath, save_path, cfg_name, 0, 5, False, True)
    metric_in_csv8 = filter_props(tgt_filepath, save_path, cfg_name, 0, 10, False, True)
    metric_in_csv9 = filter_props(tgt_filepath, save_path, cfg_name, 0, 50, False, True)
    metric_in_csv10 = filter_props(tgt_filepath, save_path, cfg_name, 0, 100, False, True)

    print(f'ori,{metric_in_csv1}')
    print(f'thresh 0.01,{metric_in_csv2}')
    print(f'thresh 0.05,{metric_in_csv3}')
    print(f'thresh 0.10,{metric_in_csv4}')
    print(f'thresh 0.20,{metric_in_csv5}')
    print(f'topx 1,{metric_in_csv6}')
    print(f'topx 5,{metric_in_csv7}')
    print(f'topx 10,{metric_in_csv8}')
    print(f'topx 50,{metric_in_csv9}')
    print(f'topx 100,{metric_in_csv10}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", default='fineaction')
    argparser.add_argument("--Esplit")
    argparser.add_argument("--Tsplit")
    argparser.add_argument("--subset")
    argparser.add_argument("--model")
    args = argparser.parse_args()

    dataset = args.dataset
    Tsplit = args.Tsplit
    Esplit = args.Esplit
    subset = args.subset
    model = args.model

    assert Tsplit in ['all', 'K400', 'nonK400']
    assert Esplit in ['all', 'K400', 'nonK400']
    assert subset in ['training', 'validation']
    assert model in ['clip', 'vificlip']

    gt_thumos14_filepath_dict = {
        'training': {
            'all': './data/thumos14/annotations/training_tal.json',
            'K400': './data/thumos14/annotations/training_K400_tal.json',
            'nonK400': './data/thumos14/annotations/training_nonK400_tal.json',
        },
        'validation': {
            'all': './data/thumos14/annotations/validation_tal.json',
            'K400': './data/thumos14/annotations/validation_K400_tal.json',
            'nonK400': './data/thumos14/annotations/validation_nonK400_tal.json',
        },
    }
    gt_fineaction_filepath_dict = {
        'training': {
            'all': './data/fineaction/annotations/training_tal.json',
            'K400': './data/fineaction/annotations/training_K400_tal.json',
            'nonK400': './data/fineaction/annotations/training_nonK400_tal.json',
        },
        'validation': {
            'all': './data/fineaction/annotations/validation_tal.json',
            'K400': './data/fineaction/annotations/validation_K400_tal.json',
            'nonK400': './data/fineaction/annotations/validation_nonK400_tal.json',
        },
    }
    gt_filepath_dict_all = {
        'thumos14': gt_thumos14_filepath_dict,
        'fineaction': gt_fineaction_filepath_dict,
    }

    AF_thumos14_K400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_K400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_thumos14_nonK400_train_prop_ep35 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_035.json'
    AF_thumos14_nonK400_train_prop_ep30 = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/proposal_training_nonK400_tal/thumos_ViFiCLIP_prop_K400_1_epoch_030.json'
    AF_thumos14_save_path = './ckpt/cls_agnostic/thumos_ViFiCLIP_prop_K400_1/pseudo_labels'

    pred_fineaction_vifi_filepath_dict_all = {
        'training': {
            'all': '',
            'K400': '',
            'nonK400': f'./ckpt/cls_agnostic/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall/proposal_training_nonK400_tal/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
        },
        'validation': {
            # 'all': f'./ckpt/cls_agnostic/fineaction_ViFiCLIP_prop_all_B2_ep3_WUep5_min1e-4_Eall/proposal_validation_tal/fineaction_ViFiCLIP_prop_all_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
            'all': f'./ckpt/cls_agnostic/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall/proposal_validation_tal/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
            'K400': '',
            'nonK400': f'./ckpt/cls_agnostic/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall/proposal_validation_nonK400_tal/fineaction_ViFiCLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
        }
    }
    pred_fineaction_clip_filepath_dict_all = {
        'training': {
            'all': f'./ckpt/cls_agnostic/fineaction_CLIP_prop_all_B2_ep3_WUep5_min1e-4_Eall/proposal_training_tal/fineaction_CLIP_prop_all_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
            'K400': '',
            'nonK400': f'./ckpt/cls_agnostic/fineaction_CLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall/proposal_training_nonK400_tal/fineaction_CLIP_prop_K400_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
        },
        'validation': {
            'all': f'./ckpt/cls_agnostic/fineaction_CLIP_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall/proposal_validation_tal/fineaction_CLIP_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
            'K400': '',
            'nonK400': f'./ckpt/cls_agnostic/fineaction_CLIP_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall/proposal_validation_nonK400_tal/fineaction_CLIP_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json',
        },
    }
    pred_fineaction_filepath_dict_all = {
        'vificlip': pred_fineaction_vifi_filepath_dict_all,
        'clip': pred_fineaction_clip_filepath_dict_all,
    }
    model_names = {'vificlip': 'ViFiCLIP', 'clip': 'CLIP'}
    pred_dirnames = {'all': f'proposal_{subset}_tal', 'K400': f'proposal_{subset}_K400_tal', 'nonK400':  f'proposal_{subset}_nonK400_tal'}
    # tgt_pred_filepath = AF_thumos14_nonK400_train_prop_ep35
    # save_path = AF_thumos14_save_path

    tgt_gt_filepath = gt_filepath_dict_all[dataset][subset][Esplit]
    # tgt_pred_filepath = f'./ckpt/cls_agnostic/fineaction_{model_names[model]}_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall/{pred_dirnames[Esplit]}/fineaction_{model_names[model]}_prop_{Tsplit}_B2_ep3_WUep5_min1e-4_Eall_epoch_008.json'
    # tgt_save_path = '/'.join(tgt_pred_filepath.split('/')[:-2] + ['pseudo_labels', subset])
    cfg_name = f'T-{Tsplit}_E-{Esplit}'
    tgt_pred_filepath = './ckpt/cls_agnostic_T-FA_E-TH/ViFiCLIP_prop_all/proposal_validation_tal/ViFiCLIP_prop_all_epoch_008.json'
    tgt_save_path = './ckpt/cls_agnostic_T-FA_E-TH/ViFiCLIP_prop_all/pseudo_labels'
    cfg_name = f'T-{Tsplit}-FA_E-{Esplit}'
    os.makedirs(tgt_save_path, exist_ok=True)
    
    with open(tgt_gt_filepath ,'r') as fp:
        tal_annos = json.load(fp)['database']
    
    
    # score_thresh = 0.01
    # topx = 5
    # thresh_flag = True
    # topx_flag = False
    # filter_props(AF_nonK400_train_prop_ep35, save_path, cfg_name, score_thresh, topx, thresh_flag, topx_flag)
    # filter_props(AF_nonK400_train_prop_ep35, save_path, cfg_name, score_thresh, topx, thresh_flag, topx_flag)

    filter_props_all_cfg(tgt_pred_filepath, tgt_save_path)

    
    num_gt = 0
    for vid in tal_annos.keys():
        num_gt += len(tal_annos[vid]['annotations'])
    print(num_gt)