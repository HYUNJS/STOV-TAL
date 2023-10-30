# python imports
import argparse
import os
import glob
import time
import json
from pprint import pprint
import pandas as pd

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config, merge_args
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed, run_mRec_eval, run_mAP_eval



################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    if args.opts is not None:
        merge_args(cfg, args.opts)
        
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else: # ckpt in format of directory
        ckpt_folder = cfg['ckpt_folder'] if args.ckpt == '' else args.ckpt
        assert os.path.isdir(ckpt_folder), f"CKPT file folder does not exist! - {ckpt_folder}"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                ckpt_folder, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(ckpt_folder, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
        print("Load ", ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )

    load_ema = cfg['test_cfg']['load_ema']
    if load_ema:
        print("Loading from EMA model ...")
        _load_state_dict = checkpoint['state_dict_ema']
    else:
        print("Loading from final model ...")
        _load_state_dict = checkpoint['state_dict']

    load_state_dict = {}
    for k in _load_state_dict.keys():
        if 'prompt_learner.token' in k:
            continue
        load_state_dict[k] = _load_state_dict[k]

    msg = model.load_state_dict(load_state_dict, strict=False)
    print(msg)
    del checkpoint
    CLIP_inf_only = cfg['CLIP']['inf_only']

    ##### set dirpath for saving output and metric
    output_dirpath = args.out_dir
    model_cfg = ckpt_file.split('/')[-2]
    ckpt_cfg = ckpt_file.split('/')[-1].replace('.pth', '').replace('.tar', '')
    proposal_filename = f'{model_cfg}_{ckpt_cfg}.json'
    tgt_eval_file = os.path.split(val_dataset.json_file)[-1].replace('.json', '')
    tgt_eval_file = f"{val_dataset.dataset_shortname}_{tgt_eval_file}"
    if output_dirpath == '':
        output_dirpath = os.path.split(ckpt_file)[0]

    if CLIP_inf_only:
        clip_topk = cfg['CLIP']['topk']
        clip_cos = not cfg['CLIP']['softmax']
        if clip_topk > 1:
            proposal_dirname = f'proposal_CLIP_clsK{clip_topk}'
            metric_dirname = f'metric_CLIP_clsK{clip_topk}'
        else:
            proposal_dirname = f'proposal_CLIP_cls'
            metric_dirname = f'metric_CLIP_cls'
        if clip_cos:
            proposal_dirname += '_cos'
            metric_dirname += '_cos'
        proposal_dirname += f"_{tgt_eval_file}"
        metric_dirname += f"_{tgt_eval_file}"
    elif cfg['dataset']['class_agnostic']:
        proposal_dirname = f'proposal_{tgt_eval_file}'
        metric_dirname = f'metric_{tgt_eval_file}'
    else:
        proposal_dirname = f'proposal_cls_{tgt_eval_file}'
        metric_dirname = f'metric_cls_{tgt_eval_file}'

    if load_ema:
        proposal_dirname += '_ema'
        metric_dirname += '_ema'

    proposal_filepath = os.path.join(output_dirpath, proposal_dirname, proposal_filename)
    metric_filepath = os.path.join(output_dirpath, metric_dirname, proposal_filename.replace('.json', '.csv'))
    os.makedirs(os.path.dirname(proposal_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(metric_filepath), exist_ok=True)

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    results = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=None,
        output_file=None,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        return_output=True,
        verbose=False,
        CLIP_inf_only=CLIP_inf_only,
    )
    
    """6. evaluate proposal recall"""
    with open(proposal_filepath, 'w') as fp:
        json.dump(results, fp)
    
    # tiou_thresholds = [0.5]
    dataset_name = cfg['dataset_name']
    if dataset_name == 'anet13':
        # tiou_thresholds = [0.5, 0.75, 0.95]
        tiou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    elif dataset_name == 'thumos14':
        tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    elif dataset_name == 'fineaction':
        # tiou_thresholds = [0.5, 0.75, 0.95]
        tiou_thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    else:
        raise NotImplemented(f"{dataset_name} evaluation is not implemented")
        
    score_thresh = 0.0
    dataset_name = cfg['dataset_name']
    if cfg['dataset']['class_agnostic']:
        _, _, results_dict = run_mRec_eval(val_dataset.json_file, proposal_filepath, tiou_thresholds, score_thresh,
                                           dataset_name, num_workers=cfg['loader']['num_workers'], split=cfg['val_split'][0])

        df = pd.DataFrame(results_dict, index=[0])
        df.insert(0, 'split_name', val_dataset.split_name)
        df.to_csv(metric_filepath, index=False)

        R1x, R5x = results_dict['R@1x'], results_dict['R@5x']
        R100, R300, R1000 = results_dict['R@100'], results_dict['R@300'], results_dict['R@1000']
        print(f"R@1x: {R1x:.3f}")
        print(f"R@5x: {R5x:.3f}")
        print(f"R@100: {R100:.3f}")
        print(f"R@300: {R300:.3f}")
        print(f"R@1000: {R1000:.3f}")
    else:
        mAPs = run_mAP_eval(val_dataset.json_file, proposal_filepath, tiou_thresholds, score_thresh,
                                           dataset_name, num_workers=cfg['loader']['num_workers'], split=cfg['val_split'][0])

        results_dict = {'split_name': val_dataset.split_name,
                        **{f"mAP@{tiou}": mAPs[i] for i, tiou in enumerate(tiou_thresholds)},
                        'mAP@avg': mAPs.mean()
                        }
        pd.DataFrame(results_dict, index=[0]).to_csv(metric_filepath, index=False)

    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    # parser.add_argument('ckpt', type=str, metavar='DIR',
    #                     help='path to a checkpoint')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to a checkpoint')
    parser.add_argument('--out_dir', type=str, default='',
                        help='path to a checkpoint')
    parser.add_argument('-e', '--epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)
