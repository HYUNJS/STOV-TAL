
############### thumos14

th_ckpt_opt_1="ckpt_folder ./ckpt/TH_agn_K400/thumos14_viclip-b_prop_K400_0"
th_ckpt_opt_2="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.05_min-1_load_as_ema/"
th_ckpt_opt_3="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.10_min-1_load_as_ema/"
th_ckpt_opt_4="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.05_min-5_load_as_ema/"
th_ckpt_opt_5="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.10_min-5_load_as_ema/"
th_ckpt_opt_6="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.01_min-1_load_as_ema/"
th_ckpt_opt_7="ckpt_folder ./ckpt/TH_agn_PL_K400/thumos14_viclip-b_prop_nonK400_PL_split_0_th-0.03_min-1_load_as_ema/"
th_ckpt_opt_8="ckpt_folder ./ckpt/TH_agn_K400/thumos14_clip_prop_K400_0/"
th_ckpt_opt_viclip_b_0="ckpt_folder ./ckpt/TH_agn_K400/thumos14_viclip-b_prop_K400_0/"
th_ckpt_opt_viclip_l_0="ckpt_folder ./ckpt/TH_agn_K400/thumos14_viclip-l_prop_K400_0/"

# CUDA_VISIBLE_DEVICES=2 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_all.yaml --opts ${th_ckpt_opt_1} &
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_K400.yaml --opts ${th_ckpt_opt_1} &
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_nonK400.yaml --opts ${th_ckpt_opt_1} &

# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_clip_prop_text_all.yaml --opts ${th_ckpt_opt_8} &
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_clip_prop_text_K400.yaml --opts ${th_ckpt_opt_8} &
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_clip_prop_text_nonK400.yaml --opts ${th_ckpt_opt_8} &

# CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_all.yaml --opts ${th_ckpt_opt_7} &
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_K400.yaml --opts ${th_ckpt_opt_7} &
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_nonK400.yaml --opts ${th_ckpt_opt_7} &

CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_all.yaml --opts ${th_ckpt_opt_viclip_b_0} &
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_K400.yaml --opts ${th_ckpt_opt_viclip_b_0} &
CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-b_prop_text_nonK400.yaml --opts ${th_ckpt_opt_viclip_b_0} &

# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-l_prop_text_all.yaml --opts ${th_ckpt_opt_viclip_l_0} &
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-l_prop_text_K400.yaml --opts ${th_ckpt_opt_viclip_l_0} &
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14_split_inf_val/thumos14_viclip-l_prop_text_nonK400.yaml --opts ${th_ckpt_opt_viclip_l_0} &


############### fineaction

# fa_ckpt_opt_1="ckpt_folder ./ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1"
# fa_ckpt_opt_2="ckpt_folder ./ckpt/FA_agn_PL_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema"
# fa_ckpt_opt_3="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_8_th-0.40_load_as_ema"

# CUDA_VISIBLE_DEVICES=2 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_all.yaml --opts ${fa_ckpt_opt_1} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_K400.yaml --opts ${fa_ckpt_opt_1} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_nonK400.yaml --opts ${fa_ckpt_opt_1};

# CUDA_VISIBLE_DEVICES=2 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_all.yaml --opts ${fa_ckpt_opt_2} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_K400.yaml --opts ${fa_ckpt_opt_2} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_nonK400.yaml --opts ${fa_ckpt_opt_2};

# CUDA_VISIBLE_DEVICES=2 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_all.yaml --opts ${fa_ckpt_opt_3} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_K400.yaml --opts ${fa_ckpt_opt_3} &
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/fineaction_split_inf_val/fineaction_vifi_prop_nonK400.yaml --opts ${fa_ckpt_opt_3};