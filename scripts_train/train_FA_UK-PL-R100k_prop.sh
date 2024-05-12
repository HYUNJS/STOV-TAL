split="R100k"
# ver=5
# ver=3
# ver=6
ver=8

# cfg_name="configs/joint/FAk400_vifi_prop_k400_UKR10k-PL.yaml"
cfg_name="configs/joint/FAk400_vifi_prop_k400_UK${split}-PL.yaml"
ckpt=1

echo $ver $cfg_name
# anno_005="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.05.json"
# anno_010="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.10.json"
# anno_020="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.20.json"
# anno_030="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.30.json"
anno_040="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.40.json"

ckpt_folder_v3_th040="ckpt_folder ./ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_3_th-0.40_load_as_ema/"
ckpt_folder_v6_th040="ckpt_folder ./ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_6_th-0.40_load_as_ema/"
ckpt_folder_v7_th040="ckpt_folder ./ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_7_th-0.40_load_as_ema/"
ckpt_folder_v8_th040="ckpt_folder ./ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_8_th-0.40_load_as_ema/"
v3_train_cfg="opt.epochs 3 opt.warmup_epochs 2"
v6_train_cfg="opt.epochs 2 opt.warmup_epochs 2"
v7_train_cfg="opt.epochs 4 opt.warmup_epochs 2"
v8_train_cfg="opt.epochs 1 opt.warmup_epochs 1"

# CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} --output ${ver}_th-0.05_load_as_ema --opts dataset.train_json_file ${anno_005} &
# CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.10_load_as_ema --opts dataset.train_json_file ${anno_010} &
# CUDA_VISIBLE_DEVICES=1 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.20_load_as_ema --opts dataset.train_json_file ${anno_020} &
# CUDA_VISIBLE_DEVICES=2 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.30_load_as_ema --opts dataset.train_json_file ${anno_030} &
CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} -c ${ckpt} --output 3_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} ${ckpt_folder_v3_th040} ${v3_train_cfg} &
CUDA_VISIBLE_DEVICES=1 python train.py ${cfg_name} -c ${ckpt} --output 6_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} ${ckpt_folder_v6_th040} ${v6_train_cfg} &
CUDA_VISIBLE_DEVICES=2 python train.py ${cfg_name} -c ${ckpt} --output 7_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} ${ckpt_folder_v7_th040} ${v7_train_cfg} &
CUDA_VISIBLE_DEVICES=3 python train.py ${cfg_name} -c ${ckpt} --output 8_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} ${ckpt_folder_v8_th040} ${v8_train_cfg} &