#!/bin/bash

th=$1
# th=0.00
# th=0.01
# th=0.05
VER=0
D1=0
D2=1

echo threshold $th
sleep 5;

# i=0
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D1 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=1
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D1 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=2
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D1 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=3
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D1 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=4
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D1 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=5
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D2 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=6
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D2 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=7
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D2 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=8
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D2 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
# sleep 5;

# i=9
# train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
# val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
# CUDA_VISIBLE_DEVICES=$D2 python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json};
# sleep 60;


# for ((i=0;i<=9;i++))
# do
#     train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_nonsplit/T-50-${i}_E-non50-$i_th-${th}_CLS-ep10.json"
#     val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_50-${i}_tal.json"
#     python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output 1 ${train_json} ${val_json}
# done
# devices=(0 1 2 3 0 1 2 3 0 1)
# devices=(0 0 0 0 0 1 1 1 1 1)
devices=(2 2 2 2 2 3 3 3 3 3)
for ((i=0;i<=9;i++))
do
    train_json="dataset.train_json_file ./data/thumos14/pseudo_annos_non50/T-50-${i}_E-non50-${i}_th-${th}_CLS-ep10.json"
    val_json="dataset.val_json_file ./data/thumos14/annotations/train_50_test_50/validation_non50-${i}_tal.json"
    CUDA_VISIBLE_DEVICES=${devices[i]} python train.py configs/thumos14_split/thumos14_vifi_non50_pseudo_tmpl.yaml --output split${i}_th${th}_${VER} --opts ${train_json} ${val_json}&
    sleep 3;
done