#!/bin/bash

#dataset="anet13"
dataset="thumos14"

echo "training"
for ((i=0;i<=9;i++))
do
    echo $i
    python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non50-$i --Tsplit 50-$i --subset training --model vifi
#    python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non75-$i --Tsplit 75-$i --subset training --model vifi
done
