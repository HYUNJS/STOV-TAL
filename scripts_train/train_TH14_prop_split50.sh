#!/bin/bash

#VER=1
#VER="1_ema"
# VER="0_ema"
VER="0"

## split option
split="50"

## dataset option
dataset='thumos14'

## backbone option
model='clip'
# model='vifi'

echo $dataset $split $model
sleep 3

devices=(0 1 0 1 0 1 0 1 0 1)
for ((i=0;i<=9;i++))
do
    cfg_file="configs/${dataset}_split/${dataset}_${model}_prop_${split}_tmpl.yaml"
    split_id="dataset.split_id ${i}"
    output="--output split${i}_${VER}"

    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
      CUDA_VISIBLE_DEVICES=${devices[i]} python train.py ${cfg_file} ${output} --opts ${split_id}
    else
      CUDA_VISIBLE_DEVICES=${devices[i]} python train.py ${cfg_file} ${output} --opts ${split_id}&
    fi
    sleep 1;
done
