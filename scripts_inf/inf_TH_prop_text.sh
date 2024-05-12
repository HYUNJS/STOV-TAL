#!/bin/bash

cfg_all="configs/thumos14_split_inf_val/thumos14_vifi_prop_text_all.yaml"
cfg_k400="configs/thumos14_split_inf_val/thumos14_vifi_prop_text_K400.yaml"
cfg_nonk400="configs/thumos14_split_inf_val/thumos14_vifi_prop_text_nonK400.yaml"

# epoch="--epoch 10"
# epoch="--epoch 6"
# epoch="--epoch 4"
# epoch="--epoch 2"
# epoch="--epoch 1"
epoch=""

# ckpt="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_vifi_prop_nonK400_PL_1_th-0.20_min-1_load_as_ema/"
ckpt="ckpt_folder ./ckpt/TH_agn_PL_K400/TH_vifi_prop_K400_FA-PL_5_th-0.20_load_as_ema/"
CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_6_th-0.05_load_as_ema/"
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_6_th-0.05_load_as_ema/"
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/TH_agn_UKR100k-PL_K400/THk400_vifi_prop_k400_UKR100k-PL_7_th-0.20_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/TH_agn_UKR100k-PL_K400/THk400_vifi_prop_k400_UKR100k-PL_6_th-0.30_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &