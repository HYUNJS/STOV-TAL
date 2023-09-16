import os, json, argparse
import os.path as osp
from libs.utils import run_mRec_eval
import pandas as pd

split_50_list = [f'50-{i}' for i in range(10)]
split_non50_list = [f'non50-{i}' for i in range(10)]
split_75_list = [f'75-{i}' for i in range(10)]
split_non75_list = [f'non75-{i}' for i in range(10)]
    
gt_thumos14_filepath_dict = {
    'training': {
        'all': './data/thumos14/annotations/training_tal.json',
        ## 50 split
        **{s: f'./data/thumos14/annotations/train_50_test_50/training_{s}_tal.json' for s in split_50_list},
        **{s: f'./data/thumos14/annotations/train_50_test_50/training_{s}_tal.json' for s in split_non50_list},
        ## 75 split
        **{s: f'./data/thumos14/annotations/train_75_test_25/training_{s}_tal.json' for s in split_75_list},
        **{s: f'./data/thumos14/annotations/train_75_test_25/training_{s}_tal.json' for s in split_non75_list},
    },
    'validation': {
        'all': './data/thumos14/annotations/validation_tal.json',
        ## 50 split
        **{f'50-{i}': f'./data/thumos14/annotations/train_50_test_50/validation_50-{i}_tal.json' for i in range(10)},
        **{f'non50-{i}': f'./data/thumos14/annotations/train_50_test_50/validation_non50-{i}_tal.json' for i in range(10)},
        ## 75 split
        **{f'75-{i}': f'./data/thumos14/annotations/train_75_test_25/validation_75-{i}_tal.json' for i in range(10)},
        **{f'non75-{i}': f'./data/thumos14/annotations/train_75_test_25/validation_non75-{i}_tal.json' for i in range(10)},
    },
}

gt_filepath_dict_all = {
    'thumos14': gt_thumos14_filepath_dict,
}

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
    # metric_in_csv5 = filter_props(tgt_filepath, save_path, cfg_name, 0.2, 1, True, False)
    # metric_in_csv6 = filter_props(tgt_filepath, save_path, cfg_name, 0, 1, False, True)
    # metric_in_csv7 = filter_props(tgt_filepath, save_path, cfg_name, 0, 5, False, True)
    # metric_in_csv8 = filter_props(tgt_filepath, save_path, cfg_name, 0, 10, False, True)
    # metric_in_csv9 = filter_props(tgt_filepath, save_path, cfg_name, 0, 50, False, True)
    # metric_in_csv10 = filter_props(tgt_filepath, save_path, cfg_name, 0, 100, False, True)

    print(f'ori,{metric_in_csv1}')
    print(f'thresh 0.01,{metric_in_csv2}')
    print(f'thresh 0.05,{metric_in_csv3}')
    print(f'thresh 0.10,{metric_in_csv4}')
    # print(f'thresh 0.20,{metric_in_csv5}')
    # print(f'topx 1,{metric_in_csv6}')
    # print(f'topx 5,{metric_in_csv7}')
    # print(f'topx 10,{metric_in_csv8}')
    # print(f'topx 50,{metric_in_csv9}')
    # print(f'topx 100,{metric_in_csv10}')

if __name__ == '__main__':
    '''
    python filter_pseudo_proposal_split.py  --dataset thumos14 --Esplit non50-0 --Tsplit 50-0 --subset training --model vifi
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", default='thumos14')
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
    
    assert dataset in ['anet13', 'thumos14']
    assert Tsplit in ['all', 'K400', 'nonK400', *split_50_list, *split_non50_list, *split_75_list, *split_non75_list]
    assert Esplit in ['all', 'K400', 'nonK400', *split_50_list, *split_non50_list, *split_75_list, *split_non75_list]
    assert subset in ['training', 'validation']
    assert model in ['clip', 'vificlip', 'vifi']

    split_type = Tsplit.split('-')[0]
    
    ## thumos14 path
    th_ver='0'
    th_ep='035'
    th14_ckpt_cfg = f'thumos14_{model}_prop_{Tsplit}_{th_ver}'
    thumos14_train_prop = f'./ckpt/cls_agnostic_{split_type}/{th14_ckpt_cfg}/proposal_{subset}_{Esplit}_tal/{th14_ckpt_cfg}_epoch_{th_ep}.json'
    thumos14_save_path = f'./ckpt/cls_agnostic_{split_type}/{th14_ckpt_cfg}/pseudo_labels'
    
    ## anet13 path
    anet_ver='3'
    anet_ep='010'
    anet13_ckpt_cfg = f'anet13_{model}_prop_{Tsplit}_{anet_ver}'
    anet13_train_prop = f'./ckpt/cls_agnostic_{split_type}/{anet13_ckpt_cfg}/proposal_{subset}_{Esplit}_tal/{anet13_ckpt_cfg}_epoch_{anet_ep}.json'
    anet13_save_path = f'./ckpt/cls_agnostic_{split_type}/{anet13_ckpt_cfg}/pseudo_labels'
    
    ## target path
    save_paths = {'thumos14': thumos14_save_path, 'anet13': anet13_save_path}
    train_props = {'thumos14': thumos14_train_prop, 'anet13': anet13_train_prop}
    
    tgt_gt_filepath = gt_filepath_dict_all[dataset][subset][Esplit]
    tgt_pred_filepath = train_props[dataset]
    tgt_save_path = save_paths[dataset]
    cfg_name = f'T-{Tsplit}_E-{Esplit}'    
    os.makedirs(tgt_save_path, exist_ok=True)
    
    filter_props_all_cfg(tgt_pred_filepath, tgt_save_path)
    
    ## GT stat
    with open(tgt_gt_filepath ,'r') as fp:
        tal_annos = json.load(fp)['database']
    num_gt = 0
    for vid in tal_annos.keys():
        num_gt += len(tal_annos[vid]['annotations'])
    print(num_gt)