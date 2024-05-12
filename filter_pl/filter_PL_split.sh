#!/bin/bash

# # dataset="anet13"
# # dataset="thumos14"
# dataset=$1

# # model="clip"
# # model="vifi"
# model=$2

# subset="training"

# echo $dataset $model $subset
# for ((i=0;i<=9;i++))
# do
#     echo $i
#     python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non50-$i --Tsplit 50-$i --subset ${subset} --model ${model}
#     python filter_pseudo_proposal_split.py  --dataset ${dataset} --Esplit non75-$i --Tsplit 75-$i --subset ${subset} --model ${model}
# done

# for ((i=0;i<=9;i++))
# do
#     python filter_PL_split.py --Tdataset anet13 --Esplit non50-$i --Tsplit 50-$i --subset training --model clip 
# done

# for ((i=0;i<=9;i++))
# do
#     echo "split ${i}"
#     python filter_PL_split.py --Tdataset anet13 --Esplit non75-$i --Tsplit 75-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.2 &
#     python filter_PL_split.py --Tdataset anet13 --Esplit non75-$i --Tsplit 75-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.3 &
#     python filter_PL_split.py --Tdataset anet13 --Esplit non75-$i --Tsplit 75-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.4 &
# done

# for ((i=0;i<=9;i++))
# do
#     echo "split ${i}"
#     python filter_PL_split.py --Tdataset anet13 --Esplit non50-$i --Tsplit 50-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.2 &
#     python filter_PL_split.py --Tdataset anet13 --Esplit non50-$i --Tsplit 50-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.3 &
#     python filter_PL_split.py --Tdataset anet13 --Esplit non50-$i --Tsplit 50-$i --subset training --model clip --train-ver 3 --train-ep 15 --thresh 0.4 &
# done

# for ((i=0;i<=9;i++))
# do
#     echo "split ${i}"
#     python filter_PL_split.py --Tdataset thumos14 --Esplit non75-$i --Tsplit 75-$i --subset training --model clip --train-ver 0 --train-ep 35 --thresh 0.01 --min-num 1 &
#     python filter_PL_split.py --Tdataset thumos14 --Esplit non50-$i --Tsplit 50-$i --subset training --model clip --train-ver 0 --train-ep 35 --thresh 0.01 --min-num 1 &
# done

for ((i=0;i<=9;i++))
do

    echo "split ${i}"
    # python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 50-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.05 --min-num 1 &
    python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 75-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.05 --min-num 1 &
    
    # if [ $i -eq 3 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    # then
    #     python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 75-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.10 --min-num 1 &
    #     python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 75-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.20 --min-num 1 
    # else
    #     python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 75-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.10 --min-num 1 &
    #     python filter_PL_split.py --Tdataset thumos14 --Edataset uk600 --Tsplit 75-$i --Esplit R100k --subset training --model vifi --train-ver 0 --train-ep 35 --thresh 0.20 --min-num 1 &
    # fi
done