#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos_ViFiCLIP_prop_all.yaml -e 35
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos_ViFiCLIP_prop_all.yaml -e 30
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos_ViFiCLIP_prop_all.yaml -e 25
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos_ViFiCLIP_prop_all.yaml -e 20
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos_ViFiCLIP_prop_all.yaml -e 15