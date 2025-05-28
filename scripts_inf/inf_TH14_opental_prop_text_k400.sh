#!/bin/bash

th_prop_opt_1="test_cfg.proposal_filepath ./data/opental_output/thumos14/eval_ca_with_nms_fusion/proposal_TH_validation_K400_tal_ema/thumos14_vifi_prop_K400_0_ema_epoch_035.json"
out_dir_opt_1="--out_dir opental_af_results/eval_ca_with_nms_fusion"
th_prop_opt_2="test_cfg.proposal_filepath ./data/opental_output/thumos14/eval_ca_with_nms_no_fusion/proposal_TH_validation_K400_tal_ema/thumos14_vifi_prop_K400_0_ema_epoch_035.json"
out_dir_opt_2="--out_dir opental_af_results/eval_ca_with_nms_no_fusion"


CUDA_VISIBLE_DEVICES=0 python inference_opental.py configs/opental/thumos14_opental_prop_text_all.yaml ${out_dir_opt_1} --opts ${th_prop_opt_1} &
CUDA_VISIBLE_DEVICES=0 python inference_opental.py configs/opental/thumos14_opental_prop_text_K400.yaml ${out_dir_opt_1} --opts ${th_prop_opt_1} &
CUDA_VISIBLE_DEVICES=0 python inference_opental.py configs/opental/thumos14_opental_prop_text_nonK400.yaml ${out_dir_opt_1} --opts ${th_prop_opt_1} &

CUDA_VISIBLE_DEVICES=1 python inference_opental.py configs/opental/thumos14_opental_prop_text_all.yaml ${out_dir_opt_2} --opts ${th_prop_opt_2} &
CUDA_VISIBLE_DEVICES=1 python inference_opental.py configs/opental/thumos14_opental_prop_text_K400.yaml ${out_dir_opt_2} --opts ${th_prop_opt_2} &
CUDA_VISIBLE_DEVICES=1 python inference_opental.py configs/opental/thumos14_opental_prop_text_nonK400.yaml ${out_dir_opt_2} --opts ${th_prop_opt_2} &

