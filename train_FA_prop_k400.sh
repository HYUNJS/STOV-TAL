#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py configs/fineaction_split/fineaction_clip_prop_k400.yaml --output 2&
CUDA_VISIBLE_DEVICES=1 python train.py configs/fineaction_split/fineaction_vifi_prop_k400.yaml --output 2&

#CUDA_VISIBLE_DEVICES=0 python train.py configs/fineaction_split/fineaction_clip_prop_k400.yaml --output 1 --opts opt.epochs 30&
#CUDA_VISIBLE_DEVICES=1 python train.py configs/fineaction_split/fineaction_vifi_prop_k400.yaml --output 1 --opts opt.epochs 30&
