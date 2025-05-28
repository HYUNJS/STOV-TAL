
dataset='thumos14'
model='viclip-b'
split="K400"
VER="0"
# model=$1
ckpt_freq="-c 5"

min_num="1"
th="0.03"
output="split${i}_${VER}_th-${th}_min-${min_num}_load_as_ema"
train_json="dataset.train_json_file ./data/${dataset}/PL_non${split}_${model}_ver_${VER}/T-TH-${split}_E-TH-non${split}_th-${th}_min-${min_num}.json"
CUDA_VISIBLE_DEVICES=0 python train.py configs/${dataset}_split/${dataset}_${model}_prop_non${split}_PL.yaml ${ckpt_freq} --output ${output} --opts ${train_json} &

th="0.01"
output="split${i}_${VER}_th-${th}_min-${min_num}_load_as_ema"
train_json="dataset.train_json_file ./data/${dataset}/PL_non${split}_${model}_ver_${VER}/T-TH-${split}_E-TH-non${split}_th-${th}_min-${min_num}.json"
CUDA_VISIBLE_DEVICES=0 python train.py configs/${dataset}_split/${dataset}_${model}_prop_non${split}_PL.yaml ${ckpt_freq} --output ${output} --opts ${train_json} &
