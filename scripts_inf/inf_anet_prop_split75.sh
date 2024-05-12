#!/bin/bash

dataset="anet13"
# dataset="thumos14"

# subset="train"
# subset="val"
subset=$2

epoch=$1
split=75

devices=(1 1 1 1 1 1 1 1 1 1)
for ((i=0;i<=9;i++))
do
    echo $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}
    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_vifi_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_vifi_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
    fi
done
