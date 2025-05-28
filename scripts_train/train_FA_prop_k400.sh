#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train.py configs/fineaction_split/fineaction_viclip-b_prop_K400.yaml --output 1&
# CUDA_VISIBLE_DEVICES=1 python train.py configs/fineaction_split/fineaction_viclip-l_prop_K400.yaml --output 1&
CUDA_VISIBLE_DEVICES=2 python train.py configs/fineaction_split/fineaction_clip_prop_K400.yaml --output "1-1"&
CUDA_VISIBLE_DEVICES=3 python train.py configs/fineaction_split/fineaction_vifi_prop_K400.yaml --output "1-1"&

#CUDA_VISIBLE_DEVICES=0 python train.py configs/fineaction_split/fineaction_clip_prop_k400.yaml --output 1 --opts opt.epochs 30&
#CUDA_VISIBLE_DEVICES=1 python train.py configs/fineaction_split/fineaction_vifi_prop_k400.yaml --output 1 --opts opt.epochs 30&
