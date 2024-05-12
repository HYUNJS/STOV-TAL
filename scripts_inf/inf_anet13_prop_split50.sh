#!/bin/bash

## epoch
epoch=$1

## subset
# subset="train"
# subset="val"
subset=$2

## dataset option
dataset='anet13'
# dataset='thumos14'

## backbone option
#model='clip'
#model='vifi'
model=$3

## split option
split="50"
#split="75"

devices=(0 0 0 0 0 0 0 0 0 0)
for ((i=0;i<=9;i++))
do
    echo $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}
    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
    fi
done
