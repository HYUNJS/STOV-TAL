#!/bin/bash

VER=3

dataset='anet13'
model='clip'
# model='vifi'

echo $dataset
sleep 5
# ckpt_freq="-c 2"
# devices=(0 1 2 3 0 1 2 3 0 1)
# devices=(0 0 1 1 0 0 1 1 0 1)
# devices=(0 0 0 0 0 1 1 1 1 1)
# devices=(2 2 2 2 2 3 3 3 3 3)
# devices=(1 1 1 1 1 1 1 1 1 1)
# devices=(1 2 3 1 2 3 1 2 3 0)
devices=(2 3 2 3 2 3 2 3 2 3)
for ((i=0;i<=9;i++))
do
    split_id="dataset.split_id ${i}"
    if [ $i -eq 3 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    # if [ $i -eq 5 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_50_tmpl.yaml ${ckpt_freq} --output split${i}_${VER} --opts ${split_id}&
    else
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_50_tmpl.yaml ${ckpt_freq} --output split${i}_${VER} --opts ${split_id}&
    fi
    sleep 1;
done
