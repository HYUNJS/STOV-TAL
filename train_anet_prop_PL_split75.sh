#!/bin/bash

th=$1 # 0.00 | 0.01 | 0.05
VER=0

dataset='anet13'
echo threshold $th $dataset
sleep 5;

# devices=(0 1 2 3 0 1 2 3 0 1)
devices=(1 1 1 1 1 1 1 1 1 1)
# devices=(2 2 2 2 2 2 2 2 2 2)
# devices=(2 2 2 2 2 3 3 3 3 3)
for ((i=0;i<=9;i++))
do
    train_json="dataset.train_json_file ./data/${dataset}/pseudo_annos_non75_all/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
    split_id="dataset.split_id ${i}"
    CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_vifi_prop_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${split_id}
    sleep 3;
done
