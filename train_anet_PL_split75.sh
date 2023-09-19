#!/bin/bash

th=$1
# th=0.00
# th=0.01
# th=0.05
# VER=0 # ep3 | dim=256
# VER=1 # ep3 | dim=512 | no_pos
VER=2 # ep10 | dim=512 | no_pos

dataset='anet13'
echo threshold $th $dataset
sleep 5

# i=0
# train_json="dataset.train_json_file ./data/${dataset}/pseudo_annos_non75/T-75-${i}_E-non75-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/${dataset}/annotations/train_75_test_25/validation_non75-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=0 python train.py configs/${dataset}_split/${dataset}_vifi_non75_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# devices=(0 1 2 3 0 1 2 3 0 1)
devices=(0 0 0 0 0 1 1 1 1 1)
# devices=(2 2 2 2 2 3 3 3 3 3)
for ((i=0;i<=9;i++))
do
    train_json="dataset.train_json_file ./data/${dataset}/pseudo_annos_non75/T-75-${i}_E-non75-${i}_th-${th}_CLS-ep10.json"
    val_json="dataset.val_json_file ./data/${dataset}/annotations/train_75_test_25/validation_non75-${i}_tal.json"
    CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_vifi_non75_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
    sleep 3;
done
