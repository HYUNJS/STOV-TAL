split="R50k"
# ver=5
# ver=3
ver=7

# cfg_name="configs/joint/FAk400_vifi_prop_k400_UKR10k-PL.yaml"
cfg_name="configs/joint/FAk400_vifi_prop_k400_UK${split}-PL.yaml"
ckpt=1

echo $ver $cfg_name
anno_005="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.05.json"
anno_010="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.10.json"
anno_020="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.20.json"
anno_030="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.30.json"
anno_040="./data/joint/PL_FA-K400_to_UK600-${split}_vifi/T-FA-K400_E-UK600-${split}_th-0.40.json"


# CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} --output ${ver}_th-0.05_load_as_ema --opts dataset.train_json_file ${anno_005} &
CUDA_VISIBLE_DEVICES=0 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.10_load_as_ema --opts dataset.train_json_file ${anno_010} &
CUDA_VISIBLE_DEVICES=1 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.20_load_as_ema --opts dataset.train_json_file ${anno_020} &
CUDA_VISIBLE_DEVICES=2 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.30_load_as_ema --opts dataset.train_json_file ${anno_030} &
CUDA_VISIBLE_DEVICES=3 python train.py ${cfg_name} -c ${ckpt} --output ${ver}_th-0.40_load_as_ema --opts dataset.train_json_file ${anno_040} &