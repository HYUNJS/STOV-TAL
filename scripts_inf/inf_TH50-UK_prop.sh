#!/bin/bash

cfg_file="configs/uk600_split_inf_train/TH502uk600_vifi_prop_all.yaml"
json_file0="dataset.json_file ./data/uk600/annotations/training_tal_00.json"
json_file1="dataset.json_file ./data/uk600/annotations/training_tal_01.json"
json_file2="dataset.json_file ./data/uk600/annotations/training_tal_02.json"
json_file3="dataset.json_file ./data/uk600/annotations/training_tal_03.json"
json_file_UKR5k="dataset.json_file ./data/uk600/annotations/training_R5k_tal.json"
json_file_UKR10k="dataset.json_file ./data/uk600/annotations/training_R10k_tal.json"
json_file_UKR100k="dataset.json_file ./data/uk600/annotations/training_R100k_tal.json"

# devices=(0 1 2 3 0 1 2 3 2 3)
devices=(0 1 0 1 0 1 0 1 0 1)
for ((i=0;i<=9;i++))
do
    echo Split $i
    split_id="dataset.split_id ${i}"
    D=${devices[i]}

    # if [ $i -eq 4 ] || [ $i -eq 9 ]
    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=$D python inference.py ${cfg_file} --opts ${json_file_UKR100k} ${split_id}
    else
        CUDA_VISIBLE_DEVICES=$D python inference.py ${cfg_file} --opts ${json_file_UKR100k} ${split_id} &
    fi
    sleep 1
done

# for ((i=0;i<=9;i++))
# do
#     echo Split $i
#     split_id="dataset.split_id ${i}"
#     CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_file} --opts ${json_file0} ${split_id} &
#     CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_file} --opts ${json_file1} ${split_id} & 
#     CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_file} --opts ${json_file2} ${split_id} &
#     CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_file} --opts ${json_file3} ${split_id} ;
# done

