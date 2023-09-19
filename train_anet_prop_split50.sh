#!/bin/bash

th=$1
# th=0.00
# th=0.01
# th=0.05
VER=2

dataset='anet13'
echo threshold $th $dataset
sleep 5;

# devices=(0 1 2 3 0 1 2 3 0 1)
# devices=(0 0 0 0 0 1 1 1 1 1)
devices=(2 2 2 2 2 3 3 3 3 3)
for ((i=0;i<=9;i++))
do
    split_id="dataset.split_id ${i}"
    CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_vifi_prop_50_tmpl.yaml --output split${i}_${VER} --opts ${split_id}&
    sleep 3;
done
