#!/bin/bash

# dataset="anet13"
# dataset="thumos14"
dataset = $1

# model="clip"
# model="vifi"
model=$2

subset="training"

echo $dataset $model $subset
for ((i=0;i<=9;i++))
do
    echo $i
    python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non50-$i --Tsplit 50-$i --subset ${subset} --model ${model}
    python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non75-$i --Tsplit 75-$i --subset ${subset} --model ${model}
done
