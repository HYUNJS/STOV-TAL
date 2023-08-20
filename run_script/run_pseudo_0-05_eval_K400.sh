#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_ViFiCLIP_prop_pseudo_0-05_K400.yaml -e 35
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_ViFiCLIP_prop_pseudo_0-05_K400.yaml -e 30
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_ViFiCLIP_prop_pseudo_0-05_K400.yaml -e 25
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_ViFiCLIP_prop_pseudo_0-05_K400.yaml -e 20
CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos_ViFiCLIP_prop_pseudo_0-05_K400.yaml -e 15