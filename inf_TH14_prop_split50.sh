#!/bin/bash

epoch=$1
# subset="train"
# subset="val"
subset=$2

## split option
split="50"
#split="75"

## dataset option
#dataset='anet13'
dataset='thumos14'

## backbone option
model='clip'
#model='vifi'

echo $dataset $split $model
sleep 5;

devices=(0 0 0 0 0 0 0 0 0 0)
for ((i=0;i<=9;i++))
do
    echo $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}
    if [ $i -eq 4 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
    fi
    sleep 1
done

#devices=(0 0 0 0 0 1 1 1 1 1)
#for ((i=0;i<=9;i++))
#do
#    echo $i
#    split_id="dataset.split_id ${i}"
#    D=${devices[i]}
#    if [ $i -eq 4 ] || [ $i -eq 9 ]
#    then
#        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
#    else
#        CUDA_VISIBLE_DEVICES=$D python inference.py configs/${dataset}_split_inf_${subset}/${dataset}_${model}_prop_${split}_tmpl.yaml -e ${epoch} --opts ${split_id}&
#    fi
#
#    sleep 1
#done
