# split="R100k"
split="all"
# ver=5
# ver=3
ver=6

# cfg_name="configs/joint/FAk400_vifi_prop_k400_UKR10k-PL.yaml"
cfg_name="configs/joint/FAk400_vifi_prop_k400_UK${split}-PL.yaml"
th="0.40"
# ckpt="-c 1"

echo ver $ver th $th $cfg_name 

anno="dataset.train_json_file ./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-${th}.json"

ckpt_folder_v3="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_3_th-${th}_load_as_ema/"
ckpt_folder_v6="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_6_th-${th}_load_as_ema/"
ckpt_folder_v7="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_7_th-${th}_load_as_ema/"
ckpt_folder_v8="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_8_th-${th}_load_as_ema/"

v3_train_cfg="opt.epochs 3 opt.warmup_epochs 2"
v6_train_cfg="opt.epochs 2 opt.warmup_epochs 2"
v7_train_cfg="opt.epochs 4 opt.warmup_epochs 2"
v8_train_cfg="opt.epochs 1 opt.warmup_epochs 1"

v3_output_dir="--output 3_th-${th}_load_as_ema"
v6_output_dir="--output 6_th-${th}_load_as_ema"
v7_output_dir="--output 7_th-${th}_load_as_ema"
v8_output_dir="--output 8_th-${th}_load_as_ema"

CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} ${ckpt} ${v3_output_dir} --opts ${anno} ${ckpt_folder_v3} ${v3_train_cfg} &
sleep 3
CUDA_VISIBLE_DEVICES=1 python train.py ${cfg_name} ${ckpt} ${v6_output_dir} --opts ${anno} ${ckpt_folder_v6} ${v6_train_cfg} &
sleep 3
CUDA_VISIBLE_DEVICES=2 python train.py ${cfg_name} ${ckpt} ${v7_output_dir} --opts ${anno} ${ckpt_folder_v7} ${v7_train_cfg} &
sleep 3
CUDA_VISIBLE_DEVICES=3 python train.py ${cfg_name} ${ckpt} ${v8_output_dir} --opts ${anno} ${ckpt_folder_v8} ${v8_train_cfg} &
sleep 3