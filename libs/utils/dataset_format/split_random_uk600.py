import json, os, pickle
import os.path as osp
import pandas as pd
import numpy as np

def save_vid_list():
    for i in range(len(cfg_name_list)):
        cfg_name = cfg_name_list[i]
        vids = vids_list[i]
        with open(osp.join(data_dir, 'random_split_vids', f'vids_{cfg_name}.pkl'), 'wb') as fp:
            pickle.dump(vids, fp)

def save_annotation():
    ## save split GT annotations
    anno_filepath = osp.join(data_dir, 'annotations', 'training_tal.json')
    with open(anno_filepath, 'r') as fp:
        annos = json.load(fp)['database']

    for i in range(len(cfg_name_list)):
        cfg_name = cfg_name_list[i]
        tgt_vids = vids_list[i]
        tgt_annos = {}
        [tgt_annos.update({vid: annos[vid]}) for vid in tgt_vids]
        print(f"Save annotation {cfg_name}")
        with open(osp.join(data_dir, 'annotations', f'training_R{cfg_name}_tal.json'), 'w') as fp:
            json.dump({'database': tgt_annos}, fp)

def save_prediction(pred_rootdir, pred_filename):
    pred_filepath = osp.join(pred_rootdir, 'proposal_UK600_training_tal_ema', pred_filename)
    # with open(pred_filepath, 'r') as fp:
    #     preds = json.load(fp)['results']
    with open(pred_filepath, 'r') as fp:
        preds = json.load(fp)
    print("Load raw prediction")
    
    for i in range(len(cfg_name_list)):
        cfg_name = cfg_name_list[i]
        tgt_vids = vids_list[i]
        tgt_preds = {}
        [tgt_preds.update({vid: preds[vid]}) for vid in tgt_vids]
        print(f"Save prediction {cfg_name}")
        tgt_pred_rootdir = osp.join(pred_rootdir, f'proposal_UK600_training_R{cfg_name}_tal_ema')
        os.makedirs(tgt_pred_rootdir, exist_ok=True)
        tgt_pred_filepath = osp.join(tgt_pred_rootdir, pred_filename)
        with open(tgt_pred_filepath, 'w') as fp:
            json.dump({'results': tgt_preds}, fp)

def save_or_read_vids():
    if osp.isfile(vids_filepath):
        print(f"Read {vids_filepath}")
        with open(vids_filepath, 'rb') as fp:
            vids_all = pickle.load(fp)
    else:
        print(f"Save {vids_filepath}")
        vinfo = pd.read_csv(osp.join(data_dir, 'vinfo.csv'))
        vids_all = list(vinfo[vinfo['subset'] == 'training']['video_id'])
        np.random.shuffle(vids_all)
        with open(vids_filepath, 'wb') as fp:
            pickle.dump(vids_all, fp)

    return vids_all

if __name__ == "__main__":
    data_dir = f"{osp.dirname(__file__)}/../../../data/uk600"
    vids_filepath = osp.join(data_dir, 'random_vids.pkl')
    vids_all = save_or_read_vids()

        
    num_vid_per_group = 10000
    vids_5k = vids_all[0:round(0.5*num_vid_per_group)]
    vids_10k = vids_all[0:1*num_vid_per_group]
    vids_20k = vids_all[0:2*num_vid_per_group]
    vids_40k = vids_all[0:4*num_vid_per_group]
    vids_50k = vids_all[0:5*num_vid_per_group]
    vids_80k = vids_all[0:8*num_vid_per_group]
    vids_100k = vids_all[0:10*num_vid_per_group]
    vids_150k = vids_all[0:15*num_vid_per_group]
    vids_160k = vids_all[0:16*num_vid_per_group]
    vids_200k = vids_all[0:20*num_vid_per_group]
    vids_250k = vids_all[0:25*num_vid_per_group]
    vids_300k = vids_all[0:30*num_vid_per_group]
    vids_320k = vids_all[0:32*num_vid_per_group]

    # cfg_name_list = ['5k', '10k', '20k', '40k', '50k', '80k', '100k', '150k', '160k', '200k', '250k', '300k', '320k']
    # vids_list = [vids_5k, vids_10k, vids_20k, vids_40k, vids_50k, vids_80k, vids_100k, vids_150k, vids_160k, vids_200k,
    #             vids_250k, vids_300k, vids_320k]
    cfg_name_list = ['5k', '10k', '20k', '40k', '50k', '80k', '150k', '160k', '200k', '250k', '300k', '320k']
    vids_list = [vids_5k, vids_10k, vids_20k, vids_40k, vids_50k, vids_80k, vids_150k, vids_160k, vids_200k,
                vids_250k, vids_300k, vids_320k]
    # cfg_name_list = ['100k']
    # vids_list = [vids_100k]

    # save_vid_list()
    # save_annotation()
    
    # pred_rootdir = 'ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1'
    # pred_filename = 'fineaction_vifi_prop_K400_1_epoch_015.json'
    pred_rootdir = 'ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0'
    pred_filename = 'thumos14_vifi_prop_K400_0_epoch_035.json'
    save_prediction(pred_rootdir, pred_filename)