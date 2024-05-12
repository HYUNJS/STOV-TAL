#!/bin/bash

cfg_file="configs/uk600_split_inf_train/THk4002uk600_vifi_prop_all.yaml"
json_file0="dataset.json_file ./data/uk600/annotations/training_tal_00.json"
json_file1="dataset.json_file ./data/uk600/annotations/training_tal_01.json"
json_file2="dataset.json_file ./data/uk600/annotations/training_tal_02.json"
json_file3="dataset.json_file ./data/uk600/annotations/training_tal_03.json"

CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_file} --opts ${json_file0} &
CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_file} --opts ${json_file1} & 
CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_file} --opts ${json_file2} &
CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_file} --opts ${json_file3} &