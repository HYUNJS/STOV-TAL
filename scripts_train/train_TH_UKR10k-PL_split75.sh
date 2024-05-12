#!/bin/bash

split="R10k"
# ver=10
ver=$1
thresh=$2
echo $ver $thresh

ckpt="-c 2"
cfg_name="configs/joint/TH_vifi_prop_75_UK${split}-PL.yaml"

v1_train_cfg="opt.epochs 10 opt.warmup_epochs 5"
v3_train_cfg="opt.epochs 3 opt.warmup_epochs 2"
v5_train_cfg="opt.epochs 5 opt.warmup_epochs 5"
v6_train_cfg="opt.epochs 2 opt.warmup_epochs 2"
v7_train_cfg="opt.epochs 4 opt.warmup_epochs 2"
v8_train_cfg="opt.epochs 1 opt.warmup_epochs 1"
v10_train_cfg="opt.epochs 5 opt.warmup_epochs 2"

train_cfg_var="v${ver}_train_cfg"
train_cfg=$(eval echo \$$train_cfg_var)
ckpt="-c 2"

devices=(2 3 2 3 2 3 2 3 2 3)
for ((i=0;i<=9;i++))
do
    split_id="dataset.split_id ${i}"
    output="split${i}_${ver}_th-${thresh}_load_as_ema"
    train_json="dataset.train_json_file ./data/joint/PL_TH-75-${i}_to_UK600-R10k_vifi_ver_0/T-TH-75-${i}_E-UK600-R10k_th-${thresh}_min-1.json"

    if [ $i -eq 1 ] || [ $i -eq 3 ] || [ $i -eq 5 ] || [ $i -eq 7 ] || [ $i -eq 9 ]
    then
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py ${cfg_name} ${ckpt} --output ${output} --opts ${train_json} ${train_cfg} ${split_id} 
    else
        CUDA_VISIBLE_DEVICES=${devices[i]} python train.py ${cfg_name} ${ckpt} --output ${output} --opts ${train_json} ${train_cfg} ${split_id} &
    fi
    sleep 1;
done
