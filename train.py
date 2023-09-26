# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import os.path as osp

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config, merge_args
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection, valid_one_epoch, valid_proposal_all_splits,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, ANETdetectionProp)
from tqdm import tqdm

def parse_split_name(val_filename, cfg):
    if val_filename == 'validation_tal.json':
        split_name = 'val_all'
    elif val_filename == 'validation_K400_tal.json':
        split_name = 'val_K400'
    elif val_filename == 'validation_nonK400_tal.json':
        split_name = 'val_nonK400'
    elif '50-' in val_filename:
        split_cfg = val_filename.split('_')[1]
        split_name = f'val_{split_cfg}'
    elif '75-' in val_filename:
        split_cfg = val_filename.split('_')[1]
        split_name = f'val_{split_cfg}'
    elif cfg['split_name'] != '':
        split_name = cfg['split_name']
    else:
        raise NotImplementedError(f"{val_filename} is not the case")
    print("split", split_name)
    
    return split_name
    
################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    if args.opts is not None:
        merge_args(cfg, args.opts)
    # pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    run_val = cfg['train_cfg']['run_val']
    if run_val:
        val_bs = 1
        val_file_list = cfg['dataset']['val_file_list']
        if len(val_file_list) != 0:
            val_loader_list, det_eval_list, split_name_list = [], [], []
            for val_filename in val_file_list:
                split_name = parse_split_name(val_filename, cfg)
                split_name_list.append(split_name)
                cfg['dataset']['val_json_file'] = osp.join(cfg['dataset']['val_file_dir'], val_filename)
                val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
                val_loader = make_data_loader(val_dataset, False, None, val_bs, cfg['loader']['num_workers'])
                det_eval = ANETdetectionProp(
                    val_dataset.json_file,
                    val_dataset.split[0],
                    tiou_thresholds = val_dataset.tiou_thresholds,
                    dataset_name=cfg['dataset_name'], num_workers=cfg['loader']['num_workers'],
                    top_k=[10, 100, 300, 1000], top_kx=[1, 5]
                )
                val_loader_list.append(val_loader)
                det_eval_list.append(det_eval)
        else:
            val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
            split_name = parse_split_name(val_dataset.json_file, cfg)
            val_loader = make_data_loader(val_dataset, False, None, val_bs, cfg['loader']['num_workers'])
            det_eval = ANETdetectionProp(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_dataset.tiou_thresholds,
                dataset_name=cfg['dataset_name'], num_workers=cfg['loader']['num_workers'],
                top_k=[10, 100, 300, 1000], top_kx=[1, 5]
                )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in tqdm(range(args.start_epoch, max_epochs), desc='outer loop', leave=True):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        if run_val:
            if len(val_file_list) != 0:
                _ = valid_proposal_all_splits(
                    val_loader_list,
                    # model,
                    model_ema.module,
                    epoch,
                    evaluator_list=det_eval_list,
                    output_file=None,
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    split_name_list=split_name_list,
                )                
            else:
                _ = valid_one_epoch(
                    val_loader,
                    # model,
                    model_ema.module,
                    epoch,
                    evaluator=det_eval,
                    output_file=None,
                    ext_score_file=cfg['test_cfg']['ext_score_file'],
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                    verbose=False,
                    cls_agnostic=cfg['dataset']['class_agnostic'],
                    split_name=split_name
                )
        
        # save ckpt once in a while
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    # parser.add_argument('-c', '--ckpt-freq', default=1, type=int,
    #                     help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)
    main(args)
