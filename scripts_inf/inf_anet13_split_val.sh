#!/bin/bash

# i=0
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# i=1
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# sleep 120;

# i=2
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# i=3
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# sleep 120;

# i=4
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# i=5
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# sleep 120;

# i=6
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# i=7
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
# sleep 120;

i=8
CUDA_VISIBLE_DEVICES=0 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
i=9
CUDA_VISIBLE_DEVICES=2 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_50-${i}.yaml&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/anet13_split_inf_val/anet13_vifi_prop_75-${i}.yaml&
sleep 120;