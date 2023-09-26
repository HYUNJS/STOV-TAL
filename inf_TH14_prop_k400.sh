#!/bin/bash

SUBSET_TRAIN="train"
SUBSET_VAL="val"
#subset=$1

## dataset option
dataset='thumos14'
#dataset='fineaction'

## backbone option
MODEL_TYPE1='clip'
MODEL_TYPE2='vifi'
#model=$MODEL_TYPE1

k400_val_anno_opt="--opts dataset.json_file ./data/thumos14/annotations/validation_K400_tal.json"
nk400_val_anno_opt="--opts dataset.json_file ./data/thumos14/annotations/validation_nonK400_tal.json"

echo $dataset
sleep 5;

CUDA_VISIBLE_DEVICES=0 python inference.py configs/${dataset}_split_inf_${SUBSET_VAL}/${dataset}_${MODEL_TYPE1}_prop_k400.yaml ${k400_val_anno_opt}&
CUDA_VISIBLE_DEVICES=0 python inference.py configs/${dataset}_split_inf_${SUBSET_VAL}/${dataset}_${MODEL_TYPE1}_prop_k400.yaml ${nk400_val_anno_opt}&
CUDA_VISIBLE_DEVICES=0 python inference.py configs/${dataset}_split_inf_${SUBSET_TRAIN}/${dataset}_${MODEL_TYPE1}_prop_k400.yaml&

CUDA_VISIBLE_DEVICES=1 python inference.py configs/${dataset}_split_inf_${SUBSET_VAL}/${dataset}_${MODEL_TYPE2}_prop_k400.yaml ${k400_val_anno_opt}&
CUDA_VISIBLE_DEVICES=1 python inference.py configs/${dataset}_split_inf_${SUBSET_VAL}/${dataset}_${MODEL_TYPE2}_prop_k400.yaml ${nk400_val_anno_opt}&
CUDA_VISIBLE_DEVICES=1 python inference.py configs/${dataset}_split_inf_${SUBSET_TRAIN}/${dataset}_${MODEL_TYPE2}_prop_k400.yaml&
