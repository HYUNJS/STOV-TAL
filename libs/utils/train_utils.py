import os
import shutil
import time
import pickle

from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config, freeze_CLIP=True):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)
    CLIP_Frz_weights = ['logit_scale', 'text_encoder']
    CLIP_noFrz_weights = ['prompt_learner']
    freezed_weight_names = set()
    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if any([t in fpn for t in CLIP_Frz_weights]):
                p.requires_grad_(False)
                freezed_weight_names.add(fpn)
                continue

            if any([t in fpn for t in CLIP_noFrz_weights]):
                decay.add(fpn)

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    # union_params = decay | no_decay
    union_params = decay | no_decay | freezed_weight_names
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    print(freezed_weight_names)
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    eta_min = optimizer_config['eta_min']
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch,
                eta_min=eta_min
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch,
                eta_min=eta_min
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch,
                eta_min=eta_min
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch,
                eta_min=eta_min
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(tqdm(train_loader), 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # # print to terminal
            # block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
            #     curr_epoch, iter_idx, num_iters
            # )
            # block2 = 'Time {:.2f} ({:.2f})'.format(
            #     batch_time.val, batch_time.avg
            # )
            # block3 = 'Loss {:.2f} ({:.2f})\n'.format(
            #     losses_tracker['final_loss'].val,
            #     losses_tracker['final_loss'].avg
            # )
            # block4 = ''
            # for key, value in losses_tracker.items():
            #     if key != "final_loss":
            #         block4  += '\t{:s} {:.2f} ({:.2f})'.format(
            #             key, value.val, value.avg
            #         )

            # print('\t'.join([block1, block2, block3, block4]))

    # # finish up and print
    # lr = scheduler.get_last_lr()[0]
    # print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

def forward_validation(model, val_loader):
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    for iter_idx, video_list in enumerate(tqdm(val_loader), 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])
    
    return results

def evaluator_validation(results, evaluator, ext_score_file, cls_agnostic):
    if evaluator is not None:
        if ext_score_file is not None and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        if cls_agnostic:
            prop_Rxs, prop_Rs = evaluator.evaluate_proposal(results, verbose=True)
            eval_metrics = (prop_Rxs, prop_Rs)
        else:
            mAPs, mRecallxs, mRecalls = evaluator.evaluate_mAP(results, verbose=True)
            eval_metrics = (mAPs, mRecallxs, mRecalls)
    else:
        eval_metrics = 0.0
    
    return eval_metrics

def log_eval_results(tb_writer, evaluator, eval_metrics, cls_agnostic, curr_epoch, split_name):
    tiou_idx = -1
    for idx, tiou in enumerate(evaluator.tiou_thresholds):
        if tiou == 0.5:
            tiou_idx = idx
            break
    assert tiou_idx != -1
    
    if cls_agnostic:
        prop_Rxs, prop_Rs = eval_metrics
        pR1x, pR5x = prop_Rxs[tiou_idx, 0], prop_Rxs[tiou_idx, 1]
        pR10, pR100, pR300, pR1000 = prop_Rs[tiou_idx, 0], prop_Rs[tiou_idx, 1], prop_Rs[tiou_idx, 2], prop_Rs[tiou_idx, 3]
        tb_writer.add_scalar(f'{split_name}/pR@1x', pR1x, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/pR@5x', pR5x, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/pR@10', pR10, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/pR@100', pR100, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/pR@300', pR300, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/pR@1000', pR1000, curr_epoch)
    else:
        mAPs, mRecallxs, mRecalls = eval_metrics
        mAP = mAPs[tiou_idx]
        mR1x, mR5x = mRecallxs[tiou_idx, 0], mRecallxs[tiou_idx, 1]
        mR10, mR100, mR300, mR1000 = mRecalls[tiou_idx, 0], mRecalls[tiou_idx, 1], mRecalls[tiou_idx, 2], mRecalls[tiou_idx, 3]
        tb_writer.add_scalar(f'{split_name}/mAP', mAP, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@1x', mR1x, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@5x', mR5x, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@10', mR10, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@100', mR100, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@300', mR300, curr_epoch)
        tb_writer.add_scalar(f'{split_name}/mR@1000', mR1000, curr_epoch)

def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    return_output = False,
    verbose=True,
    cls_agnostic=False,
    split_name='validation',
    CLIP_inf_only=False,
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None) or return_output

    # switch to evaluate mode
    model.eval()

    ## init CLIP classifier
    if CLIP_inf_only:
        model(None, val_loader.dataset.cls_name_list)

    # gather all stats and evaluate
    results = forward_validation(model, val_loader)
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    # run evaluator
    eval_metrics = evaluator_validation(results, evaluator, ext_score_file, cls_agnostic)

    # dump to a pickle file that can be directly used for evaluation
    if output_file is not None:
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
            
    # log mAP to tb_writer
    if tb_writer is not None:
        log_eval_results(tb_writer, evaluator, eval_metrics, cls_agnostic, curr_epoch, split_name)
    
    # output tal results in json format
    if return_output:
        results = format_pkl2json(results, 0.0)
        return results

    return eval_metrics

def valid_proposal_all_splits(
    val_loader_list,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator_list = None,
    output_file = None,
    tb_writer = None,
    return_output = False,
    split_name_list=None
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator_list is not None) or (output_file is not None) or return_output

    # switch to evaluate mode
    model.eval()

    eval_metrics_list, results_list = [], []
    for i in range(len(val_loader_list)):
        val_loader = val_loader_list[i]
        evaluator = evaluator_list[i]
        split_name = split_name_list[i]
        
        # gather all stats and evaluate
        results = forward_validation(model, val_loader)
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()

        # run evaluator
        eval_metrics = evaluator_validation(results, evaluator, ext_score_file, True)
        eval_metrics_list.append(eval_metrics)
        
        # dump to a pickle file that can be directly used for evaluation
        if output_file is not None:
            with open(f'{output_file}.{split_name}', "wb") as f:
                pickle.dump(results, f)
                
        # log mAP to tb_writer
        if tb_writer is not None:
            log_eval_results(tb_writer, evaluator, eval_metrics, True, curr_epoch, split_name)
        
        # output tal results in json format
        if return_output:
            results = format_pkl2json(results, 0.0)
            results_list.append(results)
    
    if return_output:
        return results_list
    return eval_metrics_list

def format_pkl2json(results, thresh):
    ## read output
    vids = results['video-id']
    starts = results['t-start']
    ends = results['t-end']
    labels = results['label']
    scores = results['score']

    ## convert format
    conv_results = {}
    for i in range(len(vids)):
        if vids[i] not in conv_results:
            conv_results[vids[i]] = []
        if scores[i] >= thresh:
            conv_results[vids[i]].append({'label_id': int(labels[i]),
                                          'segment': [float(starts[i]), float(ends[i])],
                                          'score': float(scores[i])
                                          })
    pred_results_format = {'results': conv_results}
    
    return pred_results_format
