#!/bin/bash

VER=1

dataset='thumos14'
# model='clip'
# model='vifi'
model=$1
th=$2 # 0.00 | 0.01 | 0.05
# th="0.05"
split="75"

echo $dataset $th $model $split
sleep 3;

# ckpt_freq="-c 2"
ckpt_freq="-c 5"
# ckpt_freq="-c 10"

# devices=(0 1 2 3 0 1 2 3 0 1)
# devices=(0 1 0 1 0 1 0 1 0 1)
# devices=(0 0 0 0 0 1 1 1 1 1)
# devices=(2 2 2 2 2 3 3 3 3 3)
# devices=(1 1 1 1 1 1 1 1 1 1)
devices=(2 3 2 3 2 3 2 3 2 3)
for ((i=0;i<=9;i++))
do
    # train_json="dataset.train_json_file ./data/${dataset}/PL_non${split}_${model}/T-${split}-${i}_E-non${split}-${i}_th-${th}.json"
    # train_json="dataset.train_json_file ./data/${dataset}/PL_non${split}_${model}_ver_0_ema/T-TH-${split}-${i}_E-TH-non${split}-${i}_th-${th}.json"
    split_id="dataset.split_id ${i}"
    # output="split${i}_${VER}_th-${th}"
    # output="split${i}_${VER}_th-${th}_min-5_load_as_ema"
    output="split${i}_${VER}_th-${th}_load_as_ema_PL-vifi"

    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_non${split}_PL_tmpl.yaml ${ckpt_freq} --output ${output} --opts ${train_json} ${split_id}
    else
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_non${split}_PL_tmpl.yaml ${ckpt_freq} --output ${output} --opts ${train_json} ${split_id} &
    fi
    sleep 3;
done
