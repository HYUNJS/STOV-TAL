#!/bin/bash

cfg_all="configs/fineaction_split_inf_val/fineaction_vifi_prop_text_all.yaml"
cfg_k400="configs/fineaction_split_inf_val/fineaction_vifi_prop_text_K400.yaml"
cfg_nonk400="configs/fineaction_split_inf_val/fineaction_vifi_prop_text_nonK400.yaml"

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR100k-PL_K400/FAk400_vifi_prop_k400_UKR100k-PL_7_th-0.40_load_as_ema/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_6_th-0.30_load_as_ema/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKall-PL_K400/FAk400_vifi_prop_k400_UKall-PL_7_th-0.40_load_as_ema/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKR200k-PL_K400/FAk400_vifi_prop_k400_UKR200k-PL_3_th-0.40_load_as_ema/"

# epoch="--epoch 6"
# epoch="--epoch 4"
# epoch="--epoch 1"
epoch=" "

# ckpt="ckpt_folder ./ckpt/TH_agn_PL_K400/TH_vifi_prop_K400_FA-PL_5_th-0.05_load_as_ema/"
CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_k400} &
CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_nonk400} &
CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_all} &

# ckpt="ckpt_folder ./ckpt/TH_agn_UKR10k-PL_K400/THk400_vifi_prop_k400_UKR10k-PL_3_th-0.10_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/TH_agn_PL_K400/TH_vifi_prop_K400_FA-PL_5_th-0.05_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &


# ckpt="ckpt_folder ./ckpt/TH_agn_UKR100k-PL_K400/THk400_vifi_prop_k400_UKR100k-PL_7_th-0.10_load_as_ema/"
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1_load_as_ema_UKR100k-PL_6_th-0.40_ep4/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} ${epoch} --opts ${ckpt};

# ckpt="ckpt_folder ./ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1_load_as_ema_nonK400_PL_5_th-0.30_ep10/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} ${epoch} --opts ${ckpt};

# ckpt="ckpt_folder ./ckpt/FA_agn_K400/fineaction_vifi_prop_K400_6_load_as_ema_nonK400_PL_5_th-0.30_ep10/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} ${epoch} --opts ${ckpt};

# ckpt="ckpt_folder ./ckpt/FA_agn_K400/fineaction_vifi_prop_K400_6_load_as_ema_UKR100k-PL_6_th-0.40_ep4/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt};
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} ${epoch} --opts ${ckpt};

# # ckpt="ckpt_folder ./ckpt/FA_agn_PL-ST2_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.10_load_as_ema/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKR100k-PL-ST2_K400/FAk400_vifi_prop_k400_UKR100k-PL-ST2_8_th-0.40_load_as_ema/"
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_PL-ST2_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.20_load_as_ema/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKR100k-PL-ST2_K400/FAk400_vifi_prop_k400_UKR100k-PL-ST2_8_th-0.40_load_as_ema_stage0/"
# ckpt="ckpt_folder ./ckpt/FA_agn_UKR100k-PL-ST2_K400/FAk400_vifi_prop_k400_UKR100k-PL-ST2_6_th-0.40_load_as_ema_stage0/"
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_PL-ST2_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.30_load_as_ema/"
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=2 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_PL-ST2_K400/fineaction_vifi_prop_nonK400_PL_5_th-0.40_load_as_ema/"
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_all} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_k400} ${epoch} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=3 python inference.py ${cfg_nonk400} ${epoch} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_6_th-0.40_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.40_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} --opts ${ckpt} &
# wait

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_6_th-0.30_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.30_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} --opts ${ckpt} &
# wait

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_6_th-0.20_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.20_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} --opts ${ckpt} &
# wait

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_6_th-0.10_load_as_ema/"
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=0 python inference.py ${cfg_nonk400} --opts ${ckpt} &

# ckpt="ckpt_folder ./ckpt/FA_agn_UKR50k-PL_K400/FAk400_vifi_prop_k400_UKR50k-PL_7_th-0.10_load_as_ema/"
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_all} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_k400} --opts ${ckpt} &
# CUDA_VISIBLE_DEVICES=1 python inference.py ${cfg_nonk400} --opts ${ckpt} &
# wait

