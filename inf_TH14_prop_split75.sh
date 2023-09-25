#!/bin/bash

# dataset="anet13"
dataset="thumos14"
subset="train"
# subset="val"
epoch=$1
split=75
devices=(0 0 0 0 0 0 0 0 0 0)
for ((i=0;i<=9;i++))
do
    echo $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}
    if [ $i -eq 4 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_vifi_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py configs${dataset}_split_inf_${subset}/${dataset}_vifi_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
    fi
done
