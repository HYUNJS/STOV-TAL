# Exploring Scalability of Self-Training for Open-Vocabulary Temporal Action Localization (STOV-TAL)

The code is currently not cleaned. We will clean it in the future. If you need any clarification, please contact `js-hyun@yonsei.ac.kr`.

## Installation

We use packages from [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP). Our experiments were conducted with `cuda==11.3`, `torch==1.11.0`, `torchvision==0.12.0`, and `numpy==1.24.4`.
Alternatively, you can use the following Docker image:
```
docker pull jshyunaa/vificlip_tal:v2
```

## Dataset
Below are Google Drive links to the datasets, including features and annotations used in our experiments. For each dataset, we provide CLIP-B (clip) and ViFi-CLIP-B (ep10) features.

[[FineAction]](https://drive.google.com/drive/folders/1JXh-ZSUcs-8y1gdtyAzV0GTnoSVyQnmc)
[[THUMOS14]](https://drive.google.com/drive/folders/1gqWMTrPZSUY6A5uO580fkOiITONMKhaI)
[[ANET13]](https://drive.google.com/drive/folders/1jiFY3DuU2pI5VcjFPzwP8mUZOA-Xumjy)

Due to the large volume, we do not release other features: ViCLIP-B/L features and ViFi-CLIP-B features for untrimmed YouTube videos used for scaling up. If you need these features, please contact us.

## Directory Structure Preparation
After downloading the datasets, please organize them and the checkpoints according to the following project structure:

```
{project_root}/
│
├── data/
│   ├── anet13/
│   │   ├── annotations/
│   │   ├── vificlip_feats/
│   │   │   └── F16_w16F_s4F_ep10_trainval/
│   │   └── vinfo.csv
│   ├── thumos14/
│   │   ├── annotations/
│   │   ├── vificlip_feats/
│   │   └── vinfo.csv
│   ├── fineaction/
│   │   ├── annotations/
│   │   ├── vificlip_feats/
│   │   └── vinfo.csv
│   └── joint/
│       ├── annotations/
│       ├── vificlip_feats/
│       └── vinfo.csv
│
├── checkpoints/ (Stores pretrained backbones)
│   ├── vifi_clip/
│   │   └── vifi_clip_10_epochs_k400_full_finetuned.pth
│   └── viclip/
│       ├── ViCLIP-B_InternVid-FLT-10M.pth
│       └── ViCLIP-L_InternVid-FLT-10M.pth
├── ckpt/ (Stores training outputs)
│   ├── {config of training data}/
│   └── TH_agn_K400/
│       └── {config of model}/
│           └── thumos14_vifi_prop_K400_0/
│               ├── epoch_xxx.pth.tar
│               └── logs/
│                   └── ...
...
```

## How to Extract Video Features
Please refer to this repository for feature extraction. Note: Before extracting features, resize the shortest side of the videos to 256 pixels and set the frame rate to 30 FPS.
https://github.com/HYUNJS/vifi-clip-tal

## LMM-based OV-TAL
Please refer to this repository for OV-TAL with Gemini.
https://github.com/HYUNJS/LMM-TAL



## Citation
```BibTeX
@inproceedings{hyun2025exploring,
  title={Exploring Scalability of Self-Training for Open-Vocabulary Temporal Action Localization},
  author={Hyun, Jeongseok and Han, Su Ho and Kang, Hyolim and Lee, Joon-Young and Kim, Seon Joo},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  year={2025},
  pages={9388-9397}
}
```

## Acknowledgement
Our code is based on [ActionFormer](https://github.com/happyharrycn/actionformer_release), [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP), [InternVideo](https://github.com/OpenGVLab/InternVideo). We sincerely thank the authors for releasing their code.