#!/bin/bash

#VER=1
VER="0_ema"

## split option
#split="50"
split="75"

## dataset option
#dataset='anet13'
dataset='thumos14'

## backbone option
#model='clip'
model='vifi'

echo $dataset $split $model
sleep 5;

# devices=(0 1 2 3 0 1 2 3 0 1)
# devices=(0 0 0 0 0 0 0 0 0 0)
devices=(3 3 3 3 3 3 3 3 3 3)
#devices=(1 1 1 1 1 1 1 1 1 1)
# devices=(2 2 2 2 2 3 3 3 3 3)
for ((i=0;i<=9;i++))
do
    split_id="dataset.split_id ${i}"
    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
      CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_${split}_tmpl.yaml --output split${i}_${VER} --opts ${split_id}
    else
      CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/${dataset}_split/${dataset}_${model}_prop_${split}_tmpl.yaml --output split${i}_${VER} --opts ${split_id}&
    fi
    sleep 1;
done
