#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python train.py configs/thumos14_split/thumos14_clip_prop_k400.yaml --output 0&
#CUDA_VISIBLE_DEVICES=1 python train.py configs/thumos14_split/thumos14_clip_prop_k400.yaml --output 1 --opts opt.epochs 10&

#CUDA_VISIBLE_DEVICES=1 python train.py configs/thumos14_split/thumos14_clip_prop_k400.yaml --output 1_ema&
#CUDA_VISIBLE_DEVICES=2 python train.py configs/thumos14_split/thumos14_vifi_prop_k400.yaml --output 1_ema&

CUDA_VISIBLE_DEVICES=0 python train.py configs/thumos14_split/thumos14_clip_prop_k400.yaml --output 0_ema&
CUDA_VISIBLE_DEVICES=1 python train.py configs/thumos14_split/thumos14_vifi_prop_k400.yaml --output 0_ema&
