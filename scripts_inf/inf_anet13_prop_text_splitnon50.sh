#!/bin/bash

epoch=$1
# subset="train"
# subset="val"
subset=$2

## split option
split="non50"

## dataset option
dataset='anet13'
# dataset='thumos14'

## backbone option
#model='clip'
#model='vifi'
model=$3

echo $dataset $split $model
sleep 3;

#devices=(1 1 1 1 1 1 1 1 1 1)
# devices=(0 1 0 1 0 1 0 1 0 1)
devices=(2 3 2 3 2 3 2 3 2 3)
# devices=(0 1 2 3 0 1 2 3 0 1)
for ((i=0;i<=9;i++))
do
    echo $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}
#    if [ $i -eq 4 ] || [ $i -eq 9 ]
    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    # if [ $i -eq 3 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_text_${split}_tmpl.yaml -e ${epoch} --opts ${split_id};
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_text_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
    fi
    sleep 1
done