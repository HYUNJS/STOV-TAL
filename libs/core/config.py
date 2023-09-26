import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "epic",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    'split_name': 'all',
    "model_name": "LocPointTransformer",
    "dataset": {
        'split_id': -1,
        'root_dir': './data',
        'json_file': '',
        'train_json_file': '',
        'label_filepaths': {
            'thumos14': {
                'all': './data/thumos14/annotations/thumos14_labels.csv',
                'K400': './data/thumos14/annotations/thumos14_K400_overlap_labels.csv',
                'nonK400': './data/thumos14/annotations/thumos14_K400_nonoverlap_labels.csv',
            },
            'fineaction': {
                'all': './data/fineaction/annotations/fineaction_labels.csv',
                'K400': './data/fineaction/annotations/fineaction_K400_overlap_labels.csv',
                'nonK400': './data/fineaction/annotations/fineaction_K400_nonoverlap_labels.csv',
            },
            'anet13': {
                'all': './data/anet13/annotations/anet13_labels.csv',
                'K400': './data/anet13/annotations/anet13_K400_overlap_labels.csv',
                'nonK400': './data/anet13/annotations/anet13_K400_nonoverlap_labels.csv',
            }
        },
        'val_json_file': '',
        'val_file_dir': '',
        'val_file_list': [],
        'class_agnostic': False,
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # number of classes
        "num_classes": 97,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # if true, force upsampling of the input features into a fixed size
        # only used for ActivityNet
        "force_upsampling": False,
        "tiou_thresholds": [0.5]
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (convTransformer | conv)
        "backbone_type": 'convTransformer',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_mha_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # starting level for fpn
        "fpn_start_level": 0,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # use rel position encoding (added to self-attention)
        "use_rel_pe": False,
        # classifier with CLIP
        "CLIP_cls": False,
        # "CLIP_inf_only": False,
        "CLIP_dim": 512,
        "CLIP_backbone_name": "ViT-B/16",
        "prompt_n_ctx": 0,
        "CLIP_softmax": False,
        "CLIP_weight": 'clip',
    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0, # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
        "run_val": False
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        "eta_min": 1e-8,
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    },
    "ckpt_folder": ''
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["label_filepaths"] = config["dataset"]["label_filepaths"]
    config["model"]["dataset_name"] = config["dataset_name"]
    config["model"]["split_name"] = config["split_name"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def _update_split_id(config):
    split_id = config['dataset']['split_id']
    if split_id != -1:
        config['dataset']['json_file'] = config['dataset']['json_file'].format(split_id=split_id)
        config['dataset']['train_json_file'] = config['dataset']['train_json_file'].format(split_id=split_id)
        config['dataset']['val_json_file'] = config['dataset']['val_json_file'].format(split_id=split_id)
        config['dataset']['val_file_list'] = [s.format(split_id=split_id) for s in config['dataset']['val_file_list']]
        config['ckpt_folder'] = config['ckpt_folder'].format(split_id=split_id)
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    config = _update_split_id(config)
    
    return config

def merge_args(config, cfg_list):
    assert len(cfg_list) % 2 == 0, "Override list has odd length: {}; it must be a list of pairs".format(cfg_list)
    
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split(".")
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, "Non-existent key: {}".format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, "Non-existent key: {}".format(full_key)
        assert isinstance(v , str), "currently only support string input"

        if v.isdigit():
            v = int(v)
        d[subkey] = v
        
        if subkey == 'split_id':
            config = _update_split_id(config)