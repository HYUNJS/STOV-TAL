import json, os
import os.path as osp


root_dir = "/root/code/actionformer-prop/ckpt/TH_agn_K400/thumos14_vifi_prop_K400_0"
pred_filename = "thumos14_vifi_prop_K400_0_epoch_035.json"

anno_merged = {}
for i in range(4):
    pred_dirname = f"proposal_UK600_training_tal_{i:02d}_ema"
    pred_filepath = osp.join(root_dir, pred_dirname, pred_filename)
    with open(pred_filepath, 'r') as fp:
        anno = json.load(fp)['results']
    anno_merged.update(anno)
anno_merged = {'results': anno_merged}

save_dirname = "proposal_UK600_training_tal_ema"
save_filepath = osp.join(root_dir, save_dirname, pred_filename)
os.makedirs(osp.join(root_dir, save_dirname), exist_ok=True)
with open(save_filepath, 'w') as fp:
    json.dump(anno_merged, fp)