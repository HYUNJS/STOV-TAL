import os, json, argparse
import os.path as osp
from libs.utils import run_mRec_eval
import numpy as np
import pandas as pd
from tqdm import tqdm


split_50_list = [f'50-{i}' for i in range(10)]
split_non50_list = [f'non50-{i}' for i in range(10)]
split_75_list = [f'75-{i}' for i in range(10)]
split_non75_list = [f'non75-{i}' for i in range(10)]
split_random_list = [f'R{i}k' for i in [5, 10, 20, 40, 50, 80, 100, 150, 160, 200, 250, 300, 320]]

def get_gt_splits_filepath_dict(dataset_name):
    gt_filepath_dict = {
        'training': {
            'all': f'./data/{dataset_name}/annotations/training_tal.json',
            'K400': f'./data/{dataset_name}/annotations/training_K400_tal.json',
            'nonK400': f'./data/{dataset_name}/annotations/training_nonK400_tal.json',
            ## 50 split
            **{s: f'./data/{dataset_name}/annotations/train_50_test_50/training_{s}_tal.json' for s in split_50_list},
            **{s: f'./data/{dataset_name}/annotations/train_50_test_50/training_{s}_tal.json' for s in split_non50_list},
            ## 75 split
            **{s: f'./data/{dataset_name}/annotations/train_75_test_25/training_{s}_tal.json' for s in split_75_list},
            **{s: f'./data/{dataset_name}/annotations/train_75_test_25/training_{s}_tal.json' for s in split_non75_list},
            ## Random {...}k splits for UK600
            **{s: f'./data/{dataset_name}/annotations/training_{s}_tal.json' for s in split_random_list},
        },
        'validation': {
            'all': f'./data/{dataset_name}/annotations/validation_tal.json',
            'K400': f'./data/{dataset_name}/annotations/validation_K400_tal.json',
            'nonK400': f'./data/{dataset_name}/annotations/validation_nonK400_tal.json',
            ## 50 split
            **{f'50-{i}': f'./data/{dataset_name}/annotations/train_50_test_50/validation_50-{i}_tal.json' for i in range(10)},
            **{f'non50-{i}': f'./data/{dataset_name}/annotations/train_50_test_50/validation_non50-{i}_tal.json' for i in range(10)},
            ## 75 split
            **{f'75-{i}': f'./data/{dataset_name}/annotations/train_75_test_25/validation_75-{i}_tal.json' for i in range(10)},
            **{f'non75-{i}': f'./data/{dataset_name}/annotations/train_75_test_25/validation_non75-{i}_tal.json' for i in range(10)},
        },
    }
    
    return gt_filepath_dict



gt_filepath_dict_all = {
    'thumos14': get_gt_splits_filepath_dict('thumos14'),
    'anet13': get_gt_splits_filepath_dict('anet13'),
    'fineaction': get_gt_splits_filepath_dict('fineaction'),
    'uk600': get_gt_splits_filepath_dict('uk600'),
}

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

def remove_duplicate(result_df, tiou_thresh):
    pl_segment = np.stack(result_df['segment'].values)
    num_pl = len(pl_segment)

    ## filter out the highly overlapped predictions
    tiou_arr_pl2pl = k_segment_iou(pl_segment, pl_segment)  # [#pl, #pl]
    pl2pl_duplicate_mask = tiou_arr_pl2pl >= tiou_thresh
    pl2pl_duplicate_mask[np.arange(num_pl), np.arange(num_pl)] = False
    dup_idxs = np.stack(np.nonzero(pl2pl_duplicate_mask), axis=1)
    pl_wo_dup_mask = np.ones(num_pl).astype(bool)
    pl_wo_dup_mask[dup_idxs[dup_idxs[:, 1] > dup_idxs[:, 0], 1]] = False
    pl_segment = pl_segment[pl_wo_dup_mask]
    result_df = result_df[pl_wo_dup_mask].reset_index(drop=True)

    return result_df


def filter_props(tgt_prop_fliepath, save_path, model_cfg_name, score_thresh, topx, thresh_flag, topx_flag,
                 tiou_thresh=0.5, remove_duplicate_flag=False, min_num=1):
    assert thresh_flag or topx_flag
    if thresh_flag:
        # filter_cfg_name = f'th-{score_thresh:.2f}'
        filter_cfg_name = f'th-{score_thresh:.2f}_min-{min_num}'
    if topx_flag:
        filter_cfg_name = f'top-{topx}'
    if remove_duplicate_flag:
        filter_cfg_name = f"{filter_cfg_name}_duplicate-{tiou_thresh}"

    print("Filter: ", filter_cfg_name)
    
    with open(tgt_prop_fliepath, 'r') as fp:
        results = json.load(fp)['results']
        
    new_results = {}    
    vids = list(results.keys())
    new_annos_num, ori_annos_num = 0, 0
    for vid in tqdm(vids):
        result_df = pd.DataFrame(results[vid])
        result_df = result_df.sort_values('score', ascending=False)
        vinfo_ = vinfo[vinfo['video_id'] == vid]
        duration = vinfo_['duration'].iloc[0]
        ori_annos_num += len(result_df)
        if remove_duplicate_flag:
            result_df = remove_duplicate(result_df, tiou_thresh)

        if thresh_flag:
            mask = result_df['score'] >= score_thresh
            if mask.sum() < min_num:
                result_df = result_df[0:min_num]
            else:
                result_df = result_df[mask]

        if topx_flag:
            result_df = result_df[0:topx]

        ## clipping by 0 and duration
        # result_df.loc[:, 'ss'] = result_df['segment'].apply(lambda x: max(x[0], 0))
        # result_df.loc[:, 'es'] = result_df['segment'].apply(lambda x: min(x[1], duration))
        ss = result_df['segment'].apply(lambda x: max(x[0], 0))
        es = result_df['segment'].apply(lambda x: min(x[1], duration))
        result_df['segment'] = [[s, e] for s, e in zip(ss, es)]

        ## insert into {vid: annotation list}
        new_result = result_df.to_dict('records')
        new_results[vid] = new_result
        new_annos_num += len(result_df)
    
    cfg_full_name = f'{subset}_{model_cfg_name}_{filter_cfg_name}'
    print(f"{cfg_full_name} - #ori: {ori_annos_num} | #new: {new_annos_num}")
    save_filepath = osp.join(save_path, f'{cfg_full_name}.json')
    with open(save_filepath, 'w') as fp:
        json.dump({'results': new_results}, fp)

    return cfg_full_name
    # tiou_thresholds = [0.5]
    # thresh = 0.0
    # _, _, _, metric_in_csv = run_mRec_eval(tgt_gt_filepath, save_filepath, tiou_thresholds, thresh, Edataset, split=subset, get_csv=True)

    # return metric_in_csv

def filter_props_thresh(tgt_filepath, save_path, thresh, min_num):
    print("TGT filepath", tgt_filepath)
    cfg_full_name = filter_props(tgt_filepath, save_path, cfg_name, thresh, 1, True, False, min_num=min_num)
    return cfg_full_name

def filter_props_all_cfg(tgt_filepath, save_path):
    print("TGT filepath", tgt_filepath)
    remove_duplicate_flag = False
    tiou_thresh = 0.5
    duplicate_cfg = {'tiou_thresh': tiou_thresh, 'remove_duplicate_flag': remove_duplicate_flag}
    
    # metric_in_csv_th00 = filter_props(tgt_filepath, save_path, cfg_name, 0.00, 1, True, False)
    # metric_in_csv_th01 = filter_props(tgt_filepath, save_path, cfg_name, 0.01, 1, True, False)
    metric_in_csv_th05 = filter_props(tgt_filepath, save_path, cfg_name, 0.05, 1, True, False)
    # metric_in_csv_th10 = filter_props(tgt_filepath, save_path, cfg_name, 0.1, 1, True, False)
    # metric_in_csv_th20 = filter_props(tgt_filepath, save_path, cfg_name, 0.2, 1, True, False)
    # metric_in_csv_th30 = filter_props(tgt_filepath, save_path, cfg_name, 0.3, 1, True, False)
    # metric_in_csv_th40 = filter_props(tgt_filepath, save_path, cfg_name, 0.4, 1, True, False)
    # metric_in_csv_th50 = filter_props(tgt_filepath, save_path, cfg_name, 0.5, 1, True, False)
    # metric_in_csv_th60 = filter_props(tgt_filepath, save_path, cfg_name, 0.6, 1, True, False)
    # metric_in_csv_th70 = filter_props(tgt_filepath, save_path, cfg_name, 0.7, 1, True, False)
    # metric_in_csv_th80 = filter_props(tgt_filepath, save_path, cfg_name, 0.8, 1, True, False)
    
    # metric_in_csv_top1 = filter_props(tgt_filepath, save_path, cfg_name, 0, 1, False, True)
    # metric_in_csv_top5 = filter_props(tgt_filepath, save_path, cfg_name, 0, 5, False, True)
    # metric_in_csv_top10 = filter_props(tgt_filepath, save_path, cfg_name, 0, 10, False, True)
    # metric_in_csv_top50 = filter_props(tgt_filepath, save_path, cfg_name, 0, 50, False, True)
    # metric_in_csv_top100 = filter_props(tgt_filepath, save_path, cfg_name, 0, 100, False, True)
    
    # metric_in_csv1 = filter_props(tgt_filepath, save_path, cfg_name, 0.00, 1, True, False, **duplicate_cfg)
    # metric_in_csv2 = filter_props(tgt_filepath, save_path, cfg_name, 0.01, 1, True, False, **duplicate_cfg)
    # metric_in_csv3 = filter_props(tgt_filepath, save_path, cfg_name, 0.05, 1, True, False, **duplicate_cfg)
    # metric_in_csv4 = filter_props(tgt_filepath, save_path, cfg_name, 0.1, 1, True, False, **duplicate_cfg)
    # metric_in_csv6 = filter_props(tgt_filepath, save_path, cfg_name, 0, 1, False, True, **duplicate_cfg)
    # metric_in_csv7 = filter_props(tgt_filepath, save_path, cfg_name, 0, 5, False, True, **duplicate_cfg)
    # print(filter_props(tgt_filepath, save_path, cfg_name, 0.2, 1, True, False, **duplicate_cfg))
    # print(filter_props(tgt_filepath, save_path, cfg_name, 0.5, 1, True, False, **duplicate_cfg))
    # print(filter_props(tgt_filepath, save_path, cfg_name, 0.1, 1, False, True, **duplicate_cfg))

    # print(f'thresh 0.00,{metric_in_csv_th00}')
    # print(f'thresh 0.01,{metric_in_csv_th01}')
    # print(f'thresh 0.05,{metric_in_csv_th05}')
    # print(f'thresh 0.10,{metric_in_csv_th10}')
    # print(f'thresh 0.20,{metric_in_csv_th20}')
    # print(f'thresh 0.30,{metric_in_csv_th30}')
    # print(f'thresh 0.40,{metric_in_csv_th40}')
    # print(f'thresh 0.50,{metric_in_csv_th50}')
    # print(f'thresh 0.60,{metric_in_csv_th60}')
    # print(f'thresh 0.70,{metric_in_csv_th70}')
    # print(f'thresh 0.80,{metric_in_csv_th80}')
    # print(f'top 1,{metric_in_csv_top1}')
    # print(f'top 5,{metric_in_csv_top5}')
    # print(f'top 10,{metric_in_csv_top10}')
    # print(f'top 50,{metric_in_csv_top50}')
    # print(f'top 100,{metric_in_csv_top100}')

def merge_agn_anno(cfg_full_name):
    pl_filename = f"{cfg_full_name}.json"
    pl_filepath = osp.join(tgt_save_path, pl_filename)
    with open(pl_filepath, 'r') as fp:
        pl_annos = json.load(fp)['results']

    # train_gt_filepath = gt_filepath_dict_all[Tdataset]['training'][Tsplit.replace('non', '')]
    train_gt_filepath = gt_filepath_dict_all[Tdataset]['training'][Tsplit]
    with open(train_gt_filepath, 'r') as fp:
        gt_annos = json.load(fp)['database']

    tgt_gt_filepath = gt_filepath_dict_all[Edataset]['training'][Esplit]
    with open(tgt_gt_filepath, 'r') as fp:
        tgt_gt_annos = json.load(fp)['database']

    merged_annos = {}
    for vid in gt_annos.keys():
        annos = gt_annos[vid]['annotations'].copy()
        for a in annos:
            a['label_id'] = 0
            a.pop('label_name')
        merged_annos[vid] = gt_annos[vid]
        merged_annos[vid]['annotations'] = annos

    for vid in tqdm(pl_annos.keys()):
        merged_annos[vid] = tgt_gt_annos[vid]
        merged_annos[vid]['annotations'] = pl_annos[vid]

    merged_anno_filename = pl_filename.replace(f"{subset}_", "")
    with open(osp.join(pseudo_dirpath, merged_anno_filename), 'w') as fp:
        json.dump({'database': merged_annos}, fp)

    
def merge_agn_annos():
    pl_filenames = [f for f in sorted(os.listdir(tgt_save_path)) if f"{subset}_{cfg_name}" in f]
    
    print("Merge")
    for pl_filename in pl_filenames:
        pl_filepath = osp.join(tgt_save_path, pl_filename)
        with open(pl_filepath, 'r') as fp:
            pl_annos = json.load(fp)['results']

        # train_gt_filepath = gt_filepath_dict_all[Tdataset]['training'][Tsplit.replace('non', '')]
        train_gt_filepath = gt_filepath_dict_all[Tdataset]['training'][Tsplit]
        with open(train_gt_filepath, 'r') as fp:
            gt_annos = json.load(fp)['database']

        tgt_gt_filepath = gt_filepath_dict_all[Edataset]['training'][Esplit]
        with open(tgt_gt_filepath, 'r') as fp:
            tgt_gt_annos = json.load(fp)['database']

        merged_annos = {}
        for vid in gt_annos.keys():
            annos = gt_annos[vid]['annotations'].copy()
            for a in annos:
                a['label_id'] = 0
                a.pop('label_name')
            merged_annos[vid] = gt_annos[vid]
            merged_annos[vid]['annotations'] = annos

        for vid in tqdm(pl_annos.keys()):
            merged_annos[vid] = tgt_gt_annos[vid]
            merged_annos[vid]['annotations'] = pl_annos[vid]

        merged_anno_filename = pl_filename.replace(f"{subset}_", "")
        with open(osp.join(pseudo_dirpath, merged_anno_filename), 'w') as fp:
            json.dump({'database': merged_annos}, fp)

if __name__ == '__main__':
    '''
    python filter_PL_split.py --Tdataset anet13 --Tsplit K400 --Esplit nonK400 --subset training --model vifi 
    python filter_PL_split.py --Tdataset thumos14 --Tsplit 50-0 --Esplit non50-0 --subset training --model vifi 
    python filter_PL_split.py --Tdataset thumos14 --Edataset fineaction --Tsplit all --Esplit all --subset training --model vifi 
    python filter_PL_split.py --Tdataset thumos14 --Edataset fineaction --Tsplit K400 --Esplit all --subset training --model vifi 
    python filter_PL_split.py --Tdataset fineaction --Edataset uk600 --Tsplit K400 --Esplit R10k --subset training --model vifi --thresh 0.1
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--Tdataset")
    argparser.add_argument("--Edataset", default="")
    argparser.add_argument("--Tsplit", required=True)
    argparser.add_argument("--Esplit", required=True)
    argparser.add_argument("--subset", required=True)
    argparser.add_argument("--model", required=True)
    argparser.add_argument("--train-ver", default="", type=str)
    argparser.add_argument("--train-ep", required=True, type=int)
    argparser.add_argument("--noema", action='store_true')
    argparser.add_argument("--thresh", default=0.0, type=float)
    argparser.add_argument("--min-num", default=1, type=int)
    argparser.add_argument("--ckpt-cfg", default="")
    args = argparser.parse_args()

    Tdataset = args.Tdataset
    Edataset = args.Edataset
    Tsplit = args.Tsplit
    Esplit = args.Esplit
    subset = args.subset
    model = args.model
    train_ver = args.train_ver
    train_ep = args.train_ep
    ema_opt = '' if args.noema else '_ema'
    thresh = args.thresh
    min_num = args.min_num
    if Edataset == '':
        Edataset = Tdataset
    if args.ckpt_cfg != "":
        train_ver = args.ckpt_cfg
        
    print(f"Tsplit {Tsplit} | Esplit {Esplit} | Tdataset {Tdataset} | Edataset {Edataset} | subset {subset} | model {model}")
    assert Tdataset in ['anet13', 'thumos14', 'fineaction']
    assert Edataset in ['anet13', 'thumos14', 'fineaction', 'uk600']
    assert Tsplit in ['all', 'K400', 'nonK400', *split_50_list, *split_non50_list, *split_75_list, *split_non75_list, *split_random_list]
    assert Esplit in ['all', 'K400', 'nonK400', *split_50_list, *split_non50_list, *split_75_list, *split_non75_list, *split_random_list]
    assert subset in ['training', 'validation']
    assert model in ['clip', 'vifi']
    short_dataset_name_dict = {'thumos14': 'TH', 'anet13': 'AN', 'fineaction': 'FA', 'uk600': 'UK600'}
    short_Tdataset_name = short_dataset_name_dict[Tdataset]
    short_Edataset_name = short_dataset_name_dict[Edataset]

    ## TODO. config
    # train_ver = '0'
    # train_ver = '1'
    # train_ver = '0_ema'
    # train_ver = '3'
    # train_ep='035'
    # train_ep='015'
    train_ep = f"{train_ep:03d}"
    PL_flag = "PL_" in train_ver
    
    if args.ckpt_cfg != "":
        split_type = Tsplit
        ckpt_cfg = args.ckpt_cfg
    elif Tsplit not in ['all', 'K400', 'nonK400']:
        split_type = Tsplit.split('-')[0]
        split_id = Tsplit.split('-')[1]
        ## TODO. random split case.
        ckpt_cfg = f'{Tdataset}_{model}_prop_{split_type}_tmpl_split{split_id}_{train_ver}'
    else:
        split_type = Tsplit
        if PL_flag:
            ckpt_cfg = f'{Tdataset}_{model}_prop_non{split_type}_{train_ver}'
        else:
            ckpt_cfg = f'{Tdataset}_{model}_prop_{split_type}_{train_ver}'

    vinfo = pd.read_csv(f'./data/{Edataset}/vinfo.csv')
    # cfg_name = f'T-{Tsplit}_E-{Esplit}'
    cfg_name = f'T-{short_Tdataset_name}-{Tsplit}_E-{short_Edataset_name}-{Esplit}'
    tgt_gt_filepath = gt_filepath_dict_all[Edataset][subset][Esplit]

    ## generalized version
    if PL_flag:
        pred_root_path = f"./ckpt/{short_Tdataset_name}_agn_PL_{split_type}/{ckpt_cfg}"
    else:
        pred_root_path = f"./ckpt/{short_Tdataset_name}_agn_{split_type}/{ckpt_cfg}"
    # pred_root_path = f"./ckpt/FA_agn_UKR100k-PL_K400/{ckpt_cfg}"
    # pred_root_path = f"./ckpt/TH_agn_K400/{ckpt_cfg}"
    
    if Esplit == 'all':
        pred_filepath = f'proposal_{short_Edataset_name}_{subset}_tal_ema/{ckpt_cfg}_epoch_{train_ep}.json'
    else:
        pred_filepath = f'proposal_{short_Edataset_name}_{subset}_{Esplit}_tal_ema/{ckpt_cfg}_epoch_{train_ep}.json'
    # pred_filepath = f'proposal_{subset}_{Esplit}_tal_ema/{ckpt_cfg}_epoch_{train_ep}.json'
    tgt_pred_filepath = osp.join(pred_root_path, pred_filepath)
    tgt_save_path = osp.join(pred_root_path, 'pseudo_labels')
    
    if Tdataset == Edataset:
        pseudo_dirname = f"PL_non{split_type}_{model}_ver_{train_ver}"
        pseudo_dirpath = f'./data/{Edataset}/{pseudo_dirname}'
    else:
        if args.ckpt_cfg != "":
            pseudo_dirname = f"PL_{args.ckpt_cfg}_to_{short_Edataset_name}-{Esplit}"
            pseudo_dirpath = f'./data/joint/{pseudo_dirname}'
        else:
            pseudo_dirname = f"PL_{short_Tdataset_name}-{Tsplit}_to_{short_Edataset_name}-{Esplit}_{model}_ver_{train_ver}"
            pseudo_dirpath = f'./data/joint/{pseudo_dirname}'
        
    print("pseudo_dirpath:", pseudo_dirpath)

    os.makedirs(tgt_save_path, exist_ok=True)
    os.makedirs(pseudo_dirpath, exist_ok=True)
    
    # filter_props_all_cfg(tgt_pred_filepath, tgt_save_path)

    # ## GT stat
    # with open(tgt_gt_filepath ,'r') as fp:
    #     tal_annos = json.load(fp)['database']
    # num_gt = 0
    # for vid in tal_annos.keys():
    #     num_gt += len(tal_annos[vid]['annotations'])
    # print(num_gt)

    ## merge into agnostic anno
    # merge_agn_annos()
    
    # filter_cfg_name = f'th-{thresh:.2f}'
    # cfg_full_name = f'{subset}_{cfg_name}_{filter_cfg_name}'
    cfg_full_name = filter_props_thresh(tgt_pred_filepath, tgt_save_path, thresh, min_num)
    merge_agn_anno(cfg_full_name)
    