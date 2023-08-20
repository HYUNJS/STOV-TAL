#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_CLIP_prop_all.yaml -e 35
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_CLIP_prop_all.yaml -e 30
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_CLIP_prop_all.yaml -e 25
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_CLIP_prop_all.yaml -e 20
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_CLIP_prop_all.yaml -e 15