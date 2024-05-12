#!/bin/bash

split="R100k"
ver=7
ckpt="-c 1"
cfg_name="configs/joint/THk400_vifi_prop_k400_UK${split}-PL.yaml"

anno_005="./data/joint/PL_TH-K400_to_UK600-${split}_vifi_ver_0/T-TH-K400_E-UK600-${split}_th-0.05_min-1.json"
anno_010="./data/joint/PL_TH-K400_to_UK600-${split}_vifi_ver_0/T-TH-K400_E-UK600-${split}_th-0.10_min-1.json"
anno_020="./data/joint/PL_TH-K400_to_UK600-${split}_vifi_ver_0/T-TH-K400_E-UK600-${split}_th-0.20_min-1.json"
anno_030="./data/joint/PL_TH-K400_to_UK600-${split}_vifi_ver_0/T-TH-K400_E-UK600-${split}_th-0.30_min-1.json"
anno_040="./data/joint/PL_TH-K400_to_UK600-${split}_vifi_ver_0/T-TH-K400_E-UK600-${split}_th-0.40_min-1.json"
v1_train_cfg="opt.epochs 10 opt.warmup_epochs 5"
v3_train_cfg="opt.epochs 3 opt.warmup_epochs 2"
v5_train_cfg="opt.epochs 5 opt.warmup_epochs 5"
v6_train_cfg="opt.epochs 2 opt.warmup_epochs 2"
v7_train_cfg="opt.epochs 4 opt.warmup_epochs 2"
v8_train_cfg="opt.epochs 1 opt.warmup_epochs 1"

train_cfg_var="v${ver}_train_cfg"
train_cfg=$(eval echo \$$train_cfg_var)

ckpt_folder_th005="ckpt_folder ./ckpt/TH_agn_UK${split}-PL_K400/THk400_vifi_prop_k400_UK${split}-PL_${ver}_th-0.05_load_as_ema/"
ckpt_folder_th010="ckpt_folder ./ckpt/TH_agn_UK${split}-PL_K400/THk400_vifi_prop_k400_UK${split}-PL_${ver}_th-0.10_load_as_ema/"
ckpt_folder_th020="ckpt_folder ./ckpt/TH_agn_UK${split}-PL_K400/THk400_vifi_prop_k400_UK${split}-PL_${ver}_th-0.20_load_as_ema/"
ckpt_folder_th040="ckpt_folder ./ckpt/TH_agn_UK${split}-PL_K400/THk400_vifi_prop_k400_UK${split}-PL_${ver}_th-0.40_load_as_ema/"

CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} ${ckpt} --output ${ver}_th-0.05_load_as_ema --opts dataset.train_json_file ${anno_005} ${ckpt_folder_th005} ${train_cfg} &
CUDA_VISIBLE_DEVICES=1 python train.py ${cfg_name} ${ckpt} --output ${ver}_th-0.10_load_as_ema --opts dataset.train_json_file ${anno_010} ${ckpt_folder_th010} ${train_cfg} &
CUDA_VISIBLE_DEVICES=2 python train.py ${cfg_name} ${ckpt} --output ${ver}_th-0.20_load_as_ema --opts dataset.train_json_file ${anno_020} ${ckpt_folder_th020} ${train_cfg} &
CUDA_VISIBLE_DEVICES=3 python train.py ${cfg_name} ${ckpt} --output ${ver}_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} ${ckpt_folder_th040} ${train_cfg} &