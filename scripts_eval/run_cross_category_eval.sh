##### Cross category evaluation example: trained on fineaction wo ST
pred_dataset="fineaction"
ckpt_dir="ckpt/FA_agn_K400/fineaction_vifi_prop_K400_1/"
pred_filename_all="proposal_CLIP_cls_FA_validation_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
pred_filename_k400="proposal_CLIP_cls_FA_validation_K400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
pred_filename_nonk400="proposal_CLIP_cls_FA_validation_nonK400_tal_ema/fineaction_vifi_prop_K400_1_epoch_015.json"
python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset all
python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset k400
python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_all} --pred_subset all --eval_subset nonk400
python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_k400} --pred_subset k400 --eval_subset all
python eval_script.py --pred_dataset ${pred_dataset} --ckpt_dir ${ckpt_dir} --pred_filename ${pred_filename_nonk400} --pred_subset nonk400 --eval_subset all
