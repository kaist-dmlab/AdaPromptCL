# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# Added code for adapromptcl
# -- Doyoung, dodokim@kaist.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import clip
from torchvision.transforms import ToPILImage, Normalize
import utils
from copy import deepcopy
from collections import defaultdict
from random import random as rand
import torch.distributed as dist

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from kornia.augmentation.auto import RandAugment
import kornia.augmentation as K
from sklearn.metrics import silhouette_score # silhouette_samples
from sklearn.cluster import KMeans
import networkx as net



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None,
                    anytime_inference_dict=None, loss_noise_pairs=None, val_data_loader=None, 
                    ema_model=None, args=None,):
    
    model.train(set_training_mode)
    original_model.eval()

    if (args.freezeE_zerograd) and (task_id >= args.freezeE_t):
        zerogradE = True
    else:
        zerogradE = False

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    if (args.noisy_task_boundary) and (args.noisy_type=='start'):
        n_batches = len(data_loader)
        noisy_batches = int(n_batches * args.noisy_pct)
        print("noisy_batches: ", noisy_batches)

    if args.wo_sep_mask: # no use sep_mask: with sep_mask -> without sep_mask 
        sep_mask = True
    else: # with sep_mask: without -> with
        sep_mask = False
        
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        for _ in range(args.num_updates):
                   
            if args.noisy_labels:
                input, target, noise = batch_data
            else:
                input, target = batch_data
                
                            
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                    if args.cl_prompts:
                        transform = K.AugmentationSequential(RandAugment(n=2, m=10)) if args.cl_randaug else \
                                        K.AugmentationSequential(ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                                                                RandomGrayscale(p=0.2))
                        aug_input = transform(input)
                        aug_cls_features = original_model(aug_input)['pre_logits']
                    else:
                        aug_cls_features = None
                else:
                    cls_features = None

            clip_emb = None
            with torch.cuda.amp.autocast():
                output = model(input, target=target, task_id=task_id, cls_features=cls_features,
                    aug_cls_features=aug_cls_features, clip_emb=clip_emb, train=set_training_mode,
                    )
            logits = output['logits'] 
            # (bs,D) D=512 if clip-text-encoder head; D=#classes otherwise
            
            with torch.cuda.amp.autocast():
                if (args.train_mask) and (class_mask is not None) and \
                    (not args.clip_text_head) and (not args.blurryCL) and (not args.online_cls_mask): 
                    mask = class_mask[task_id] # why disj setup is currently not working for sep_specialization
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                if args.blurryCL or args.online_cls_mask : # or args.milder_vtab:
                    exp_classes = torch.unique(target, sorted=True)
                    nonexp_classes = np.setdiff1d(np.arange(args.nb_classes), exp_classes.cpu().numpy() )
                    nonexp_classes = torch.tensor(nonexp_classes, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=nonexp_classes, value=float('-inf'))
                    
                loss = criterion(logits, target) # base criterion (CrossEntropyLoss) or BCE
                # print('logits', logits[torch.arange(target.size(0)), target.squeeze()])
                
                    
                if args.pull_constraint and 'reduce_sim' in output: # if not (feat_prompt or coda_prompt)
                    loss = loss - args.pull_constraint_coeff * output['reduce_sim']
                
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            if args.grad_accum:
                loss = loss/args.accum_step
                loss.backward()
                if (i+1)%args.accum_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            else:
                optimizer.zero_grad()
                loss.backward() 
                if zerogradE:
                    for n, p in model.named_parameters():
                        if n.startswith(tuple(['e_prompt', 'module.e_prompt'])): 
                            # print(n, p.grad)
                            p.grad = p.grad * 0
                            # print(n, p.grad)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                

            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
            # In train_one_epoch: evaluate when anytime_inference == True
            if (args.anytime_inference) and (i%args.anytime_inference_period==0):
                _, _ = evaluate_till_now(model=model, original_model=original_model, data_loader=val_data_loader,
                            device=device, task_id=task_id, class_mask=class_mask, 
                            acc_matrix=None, anytime_inference_dict=anytime_inference_dict,
                            only_anytime_inference=True, ema_model=ema_model, args=args)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, ema_model=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    ind2cnt = defaultdict(int)
    classid2acc, classid2cnt = defaultdict(int), defaultdict(int)
    target_set = set()

    taskid_correct, total = 0, 0
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            
            clip_emb = None
            output = model(input, task_id=task_id, cls_features=cls_features, clip_emb=clip_emb,)
            logits = output['logits']


            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            # to compute recency 
            cls_inds, cls_cnts = torch.unique(torch.argmax(logits, dim=1), return_counts=True) 
            for ind, cnt in zip(cls_inds, cls_cnts):
                ind, cnt = ind.cpu().item(), cnt.cpu().item()
                ind2cnt[ind] += cnt
            target_set = target_set.union(set(target.cpu().numpy()))

            # to compute acc per class
            correct = torch.argmax(logits, dim=1) == target
            for uniq_trg in torch.unique(target):
                classid2acc[uniq_trg.item()] += (correct[target==uniq_trg]).sum() # correct_cnt
                classid2cnt[uniq_trg.item()] += (target==uniq_trg).sum() # total_cnt
                
            # to compute task id prediction accuracy when using E prompts
            if args.use_e_prompt:
                prompt_idx = output['prompt_idx'] # (B, topk)
                taskid_correct += torch.sum(prompt_idx == task_id).item()

            total += target.size(0)
            # print('taskid_correct: ', taskid_correct, total)
            # task_id


            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    in_task_pred, out_task_pred = 0, 0
    for cls_ind,cls_cnt in ind2cnt.items():
        if cls_ind in target_set:
            in_task_pred += cls_cnt
        else:
            out_task_pred += cls_cnt
    in_task_ratio = in_task_pred*100/(in_task_pred+out_task_pred)
    print("Task {}, In-Task ratio: {}".format(task_id+1, in_task_ratio))

    taskid_pred_acc = taskid_correct/total
    print("Task {}, TaskId pred Acc: {}".format(task_id+1, taskid_pred_acc))
    
    for cls_id, correct_cnt in classid2acc.items():
        classid2acc[cls_id] = (correct_cnt/classid2cnt[cls_id]).item()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stats['in_task_ratio'] = in_task_ratio
    test_stats['taskid_pred_acc'] = taskid_pred_acc
    test_stats['classid2acc'] = dict(classid2acc)
    test_stats['task2acc'] = np.array(list(classid2acc.values())).mean()


    return test_stats


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, device,
                task_id=-1, class_mask=None, acc_matrix=None,
                anytime_inference_dict=None, only_anytime_inference=False, ema_model=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    recency_list = [] # (#tasks,)
    taskid_pred_acc_list = [] # (#tasks,)
    tasks_acc_list = [] # (#tasks,)
    clsid_acc_dict = dict() # (#tasks,)
    
    # print("args.num_tasks, task_id: ", args.num_tasks, task_id)
    for i in range(task_id+1):
        if args.eval_only_acc and i<task_id and ((task_id+1)<args.num_tasks):
            continue
        
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask,ema_model=ema_model, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        recency_list += [test_stats['in_task_ratio']]
        taskid_pred_acc_list += [test_stats['taskid_pred_acc']]
        tasks_acc_list += [test_stats['task2acc'],]
        clsid_acc_dict.update(test_stats['classid2acc'])

        if args.anytime_inference:
            anytime_inference_dict['task{}'.format(i+1)] += [ test_stats['Acc@1'], ] # i is task_id
            if only_anytime_inference == False: # end of task -> update anytime_inference_dict & acc_matrix
                acc_matrix[i, task_id] = test_stats['Acc@1']    
            # else: # only_anytime_inference == True -> update anytime_inference_dict: already done

        elif acc_matrix is not None: # no anytime_inference -> update only acc_matrix
            acc_matrix[i, task_id] = test_stats['Acc@1']

    if args.dataset in ['overlapping-vtab-1k',]:
        overlap_tasks = np.array([ data_loader[i]['task'] for i in range(task_id+1) ])
        print('overlap_tasks', overlap_tasks)
        print('stat_matrix', stat_matrix)

        uniq, inv, cnts = np.unique(overlap_tasks, return_inverse=True, return_counts=True)
        for task_i, cnt in enumerate(cnts):
            if cnt > 1: # overlapping_tasks 
                # real_task_i = uniq[task_i]
                overlapping_tasks = np.arange(len(inv))[inv==task_i]
                stat_matrix[:, overlapping_tasks] = stat_matrix[:, overlapping_tasks]/cnt
        print('stat_matrix', stat_matrix)
        avg_stat = list(np.divide(np.sum(stat_matrix, axis=1), len(uniq)))
        
    else:
        avg_stat = list(np.divide(np.sum(stat_matrix, axis=1), task_id+1))

    if acc_matrix is None:
        return None, None
        
    diagonal = np.diag(acc_matrix)
    clsid_acc_dict = sorted(clsid_acc_dict.items(), key=lambda kv: kv[0])

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
        avg_stat = list(avg_stat) + [forgetting, backward]
        print(avg_stat)
    print(result_str)
    

    return test_stats, avg_stat+[recency_list, taskid_pred_acc_list, tasks_acc_list, clsid_acc_dict]

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, ema_model=None, args = None,):
    
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    if args.anytime_inference:
        anytime_inference_dict = defaultdict(list)
    else:
        anytime_inference_dict = None
    
    converge_hist = None
    
    cls_prototypes = None
    # use if uni_or_specific and comp_warmups
    warmup_prompts = [] # (#tasks, ...); 
    birch, postmerge_birch = None, None
    mergable_prompts_info, reduce_mergable_info = None, None
    uni_or_specific_epoch = None
    uni_or_spcf = 'ND'
    args.uni_or_spcf_gt, exist = [], []
    if args.uni_or_spcf_gt:
        args.task_id_list = [data_loader[i]['task'] for i in range(len(data_loader))] 
        for t_id in args.task_id_list:
            if t_id in exist:
                args.uni_or_spcf_gt += ['uni'] 
            else:
                exist += [t_id]
                args.uni_or_spcf_gt += ['spcf'] 
        print('task_id_list', args.task_id_list)
        print('uni_or_spcf_gt', args.uni_or_spcf_gt)

    args.exposed_classes = set()
    for task_id in range(args.num_tasks):
        loss_noise_pairs = None
        

        # if (task_id==args.freezeG_t) and (args.clip_emb) and (args.freezeG): # When using clip-model
        if (task_id==args.freezeG_t) and (args.freezeG): 
            for n, p in model_without_ddp.named_parameters():
                if n in ['g_prompt', 'module.g_prompt']: 
                    print('p.requires_grad', p.requires_grad)
                    p.requires_grad = False
                    
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                            find_unused_parameters=False,
                                            # True if not args.clip_text_head else False
                                            )

            for n, p in model.named_parameters():    
                if n in ['g_prompt', 'module.g_prompt']: 
                    print('p.requires_grad', p.requires_grad)
                    
        if (task_id==args.freezeE_t) and (args.freezeE): 
            print("Freeze E")
            for n, p in model_without_ddp.named_parameters():
                # print(n, 'p.requires_grad', p.requires_grad)
                if n.startswith(tuple(['e_prompt', 'module.e_prompt'])): 
                    print(n, 'p.requires_grad', p.requires_grad)
                    p.requires_grad = False
                    
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                            find_unused_parameters=False,
                                            # True if not args.clip_text_head else False
                                            )

            for n, p in model.named_parameters():    
                if n.startswith(tuple(['e_prompt', 'module.e_prompt'])): 
                    print('p.requires_grad', p.requires_grad)

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k if not args.shared_mean else None
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    if not (args.clip_emb or args.only_G): # if not using clip_emb, no use of task-id and E-prompt
                        with torch.no_grad():
                            if args.distributed:
                                model.module.e_prompt.prompt.grad.zero_()
                                if args.shared_mean:
                                    model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx].mean(dim=2, keepdim=True)
                                else:
                                    model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        ##### START: if freezeG by # of epoch, unfreeze G at beginning of task ##############
        if (args.freezeG) and (args.freezeG_e > -1): # freezeG by the # of epoch
            for n, p in model_without_ddp.named_parameters():
                if n in ['g_prompt', 'module.g_prompt']: 
                    print('p.requires_grad', p.requires_grad)
                    p.requires_grad = True
                    
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                            find_unused_parameters=False,
                                            # True if not args.clip_text_head else False
                                            )

            for n, p in model.named_parameters():    
                if n in ['g_prompt', 'module.g_prompt']: 
                    print('p.requires_grad', p.requires_grad)
        ##### END: if freezeG by # of epoch, unfreeze G at beginning of task ##############
    
        ##### if uni_or_specific: temporarily save state_dict of clf-head #####
        if args.uni_or_specific:
            temp_head_params = dict()
            for n, p in model.module.named_parameters():
                if n.startswith('head'): 
                    temp_head_params[n] = deepcopy(p.detach()) # (#classes,D=768)
        ##### if uni_or_specific: temporarily save state_dict of clf-head #####
        
        num_of_epochs = args.epochs+args.evolve_epoch+1 if args.uni_or_specific else args.epochs

        if (args.eval_prototype_clf) and (task_id>0):
            num_of_epochs = 0 # thus, pass the training after 1st task

        args.use_film = False
        args.mergable_cands = None
        
        # for epoch in range(args.epochs):        
        for epoch in range(num_of_epochs):       
            
            # ### after deciding uni or specific, if epoch_cnt > left_epochs: continue
            if args.uni_or_specific and (uni_or_specific_epoch is not None):
                uni_or_specific_epoch += 1
                if uni_or_specific_epoch > left_epochs: 
                    # print("pass iter: uni_or_specific_epoch > left_epochs")
                    continue
            

            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, 
                                        anytime_inference_dict=anytime_inference_dict,
                                        val_data_loader=data_loader if args.anytime_inference else None,
                                        loss_noise_pairs=loss_noise_pairs, ema_model=ema_model,
                                        args=args,)

            if lr_scheduler:
                lr_scheduler.step(epoch)
                
                
            if args.data_driven_evolve and args.evolve_epoch == epoch:
                print("Epoch: {} (warmup), Check how to evolve prompts".format(epoch))
                if args.uni_or_specific:
                    converge_hist, warmup_prompts, birch, left_epochs, uni_or_spcf, postmerge_birch, mergable_cands, mergable_prompts_info, reduce_mergable_info = \
                        uni_or_specific_prompts(model=model, optimizer=optimizer, birch=birch, postmerge_birch=postmerge_birch, task_id=task_id, converge_hist=converge_hist,
                            warmup_prompts=warmup_prompts, mergable_prompts_info=mergable_prompts_info, reduce_mergable_info=reduce_mergable_info, args=args)
                    uni_or_specific_epoch = 0
                    args.mergable_cands = mergable_cands
                    
                    #### recover clf-head's param after warmup ####
                    if task_id >0:
                        with torch.no_grad():
                            # num_same = torch.all(model.module.head.weight==temp_head_params['head.weight'], dim=1).sum()
                            # print('before recover, num_same', num_same)
                            model.module.head.weight.data = temp_head_params['head.weight']
                            model.module.head.bias.data = temp_head_params['head.bias']
                            # num_same = torch.all(model.module.head.weight==temp_head_params['head.weight'], dim=1).sum()
                            # print('after recover, num_same', num_same)
                                                    
                        optimizer = create_optimizer(args, model)
                        
                        ##### reinit temp_head_params #####
                        temp_head_params = dict()
                        for n, p in model.module.named_parameters():
                            if n.startswith('head'): 
                                temp_head_params[n] = deepcopy(p.detach()) # (#classes,D=768)
                        
                        
            ### recover head and reinit optim
            if args.uni_or_specific and args.mergable_prompts and (uni_or_specific_epoch is not None):
                if args.reinit_clf and (epoch==(left_epochs+args.evolve_epoch-10) or epoch==(num_of_epochs-10-1)) : # 99 <- 100-1
                    
                    
                    model.module.head.weight.data = temp_head_params['head.weight']
                    model.module.head.bias.data = temp_head_params['head.bias']
                   
                    # reinit optim
                    optimizer = create_optimizer(args, model)
                                
            
            if (args.save_sdict_by_epochs) and (epoch in args.save_sdict_epochs) :
                # e.g., args.save_sdict_epochs = [0,4,29]
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, 
                                               'checkpoint/task{}_checkpoint_e{}.pth'.format(task_id+1, epoch))
                state_dict = {
                        'model': model_without_ddp.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                utils.save_on_master(state_dict, checkpoint_path)
           
        # after post-training, reinit uni_or_specific_epoch to None
        if args.uni_or_specific:
            uni_or_specific_epoch = None
            uni_or_spcf = 'ND'
            
        # convergent or divergent evolution
        if args.data_driven_evolve and args.evolve_epoch == -1:
            print("Check how to evolve prompts")
            converge_hist = evolve_prompts(model=model, original_model=original_model,task_id=task_id, 
                           converge_hist=converge_hist, args=args)
            
        
        test_stats, _avg_stat = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, anytime_inference_dict=anytime_inference_dict,
                  only_anytime_inference=False, ema_model=ema_model, args=args)
        avg_stat = {'Acc@1':_avg_stat[0], 'Acc@5':_avg_stat[1],'Loss':_avg_stat[2]}
        if task_id > 0:
            avg_stat['forgetting'] = _avg_stat[3]
            avg_stat['backward'] = _avg_stat[4]
        avg_stat['in_task_ratio'] = _avg_stat[-4] # 'in_task_ratio'
        avg_stat['taskid_pred_acc'] = _avg_stat[-3] # 'taskid_pred_acc'
        avg_stat['tasks_acc_list'] = _avg_stat[-2] # 'tasks_acc_list'
        # avg_stat['classid_acc'] = _avg_stat[-1] # 'classid_acc'

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            ### save model's state_dict #### Uncomment this to save model's state_dict
            # if task_id == (args.num_tasks-1): # save the last state dict
                # utils.save_on_master(state_dict, checkpoint_path)
            if args.save_sdict:
                utils.save_on_master(state_dict, checkpoint_path)

            if args.save_warmup_prompts and task_id == (args.num_tasks-1): # save the last state dict
                if args.data_driven_evolve and args.uni_or_specific:
                    utils.save_on_master(warmup_prompts, checkpoint_path)

            ### save prep-groups history ###
            if args.data_driven_evolve and  args.mergable_prompts and args.save_prepgroup_info:
                Path(os.path.join(args.output_dir, 'prepgroup_info')).mkdir(parents=True, exist_ok=True)
                prepgroup_path = os.path.join(args.output_dir, 'prepgroup_info/task{}_ckpt.pth'.format(task_id+1))
                utils.save_on_master(reduce_mergable_info, prepgroup_path)

            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in avg_stat.items()},
            # **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}
        
        if converge_hist is not None:
            print('converge_hist: ', list(converge_hist.items()))
            log_stats['converge_hist'] = list(converge_hist.items())

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    # save evaluation results of anytime inference
    if args.anytime_inference and args.output_dir and utils.is_main_process():
        import csv
        Path(os.path.join(args.output_dir, 'anytime_inference_dict')).mkdir(parents=True, exist_ok=True)
        anytime_inference_dict = dict(anytime_inference_dict) # {'task{id}':[acc1,acc2,...], ...}
        anytime_inference_path = os.path.join(args.output_dir,'anytime_inference_dict/anytime_inference.csv')
        w = csv.writer(open(anytime_inference_path, "w"))
        for key, val in anytime_inference_dict.items():
            w.writerow([key, val])
        
def evolve_prompts(model, original_model, task_id, converge_hist, args):
    """
    1. get similarity between prompts
    2. decision: by threshold
    3. if merge, update map_funtion
    4. merge
    """
    from torchmetrics.functional import pairwise_cosine_similarity

    if task_id > 0:
        with torch.no_grad():    
            # 1. get similarity between prompts
            prompts = model.module.e_prompt.prompt if args.distributed else model.e_prompt.prompt
            _, _, num_tasks, _, _, _ = prompts.size() # num_layer, dual, num_tasks, length, heads, emb
            prompts = prompts.mean(0).mean(0).mean(1).view(num_tasks, -1)[:task_id+1] # including itself
            sim = pairwise_cosine_similarity( prompts ) # (task_id+1, task_id+1)
            sim = sim[-1,:] # (task_id+1,)
            print('similarity', sim)
            
            # 2. decision: by threshold
            converged_p = model.module.e_prompt.converged_p if args.distributed else model.e_prompt.converged_p
            
            # if directional, ignore the converged prompts
            if (not args.bidirectional) and (converged_p is not None):
                sim[converged_p] = -1
            
            # 3. if merge, update map_funtion
            similar_tf = sim > args.converge_thrh
            similar_prompt_ids = torch.arange(task_id+1)[similar_tf]
            similar_prompt_sim = sim[similar_tf]

            converge_hist[task_id] = [float(v) for v in list(similar_prompt_ids.float().numpy())]
            print( 'task {} coverged with '.format(task_id), list(similar_prompt_ids.numpy()) )
            
            if args.mix_with_sim and (similar_tf.sum() > 0):
                bwd_ratio = similar_prompt_sim
                fwd_ratio = similar_prompt_sim.mean()
            else:
                bwd_ratio = [args.bwd_ratio for _ in range(similar_tf.sum())]
                fwd_ratio = args.fwd_ratio
            
            print('bwd&fwd', bwd_ratio, fwd_ratio)
            
            # 4. merge
            if (args.bidirectional) and (similar_tf.sum() > 0): # Bi-Directional
                # compress all similar prompts
                prompt_p_ids = (slice(None),slice(None),similar_prompt_ids)
                cprs_prev_prompts = model.module.e_prompt.prompt[prompt_p_ids]
                cprs_prev_keys = model.module.e_prompt.prompt_key[similar_prompt_ids]
                print('cprs_prevs', cprs_prev_prompts.size(), cprs_prev_keys.size())
                
                if args.normalize_by_sim: # compressing based on similarity
                    norm_sim = similar_prompt_sim/similar_prompt_sim.sum()
                    cprs_prev_prompts = [norm_sim[i]*cprs_prev_prompts[:,:,i] for i in range(len(similar_prompt_ids))]
                    cprs_prev_prompts = cprs_prev_prompts.sum(dim=2, keepdim=True)
                    cprs_prev_keys = [norm_sim[i]*cprs_prev_keys[i] for i in range(len(similar_prompt_ids))]
                    cprs_prev_keys = cprs_prev_keys.sum(dim=2, keepdim=True)
                else: # simply averaging
                    cprs_prev_prompts = cprs_prev_prompts.mean(dim=2, keepdim=True)
                    cprs_prev_keys = cprs_prev_keys.mean(dim=0, keepdim=True)
                    
                prompt_curr_ind = (slice(None),slice(None),[task_id])
                
                print('cprs_prevs', cprs_prev_prompts.size(), cprs_prev_keys.size())
                print('curr', model.module.e_prompt.prompt[prompt_curr_ind].size(), model.module.e_prompt.prompt_key[[task_id]].size())

                # mix prompts: curr -> prev
                for i, p_id in enumerate(similar_prompt_ids):
                    prompt_p_id = (slice(None),slice(None),[p_id])
                    model.module.e_prompt.prompt.grad.zero_()
                    model.module.e_prompt.prompt[prompt_p_id] = (bwd_ratio[i])*model.module.e_prompt.prompt[prompt_curr_ind] +\
                                        (1-bwd_ratio[i])*model.module.e_prompt.prompt[prompt_p_id]
                    # mix keys
                    model.module.e_prompt.prompt_key.grad.zero_()
                    model.module.e_prompt.prompt_key[[p_id]] = (bwd_ratio[i])*model.module.e_prompt.prompt_key[[task_id]] +\
                                        (1-bwd_ratio[i])*model.module.e_prompt.prompt_key[[p_id]]

                # mix prompts: cprs -> curr
                model.module.e_prompt.prompt.grad.zero_()
                model.module.e_prompt.prompt[prompt_curr_ind] = (fwd_ratio)*cprs_prev_prompts +\
                                    (1-fwd_ratio)*model.module.e_prompt.prompt[prompt_curr_ind]
                # mix keys
                model.module.e_prompt.prompt_key.grad.zero_()
                model.module.e_prompt.prompt_key[[task_id]] = (fwd_ratio)*cprs_prev_keys +\
                                    (1-fwd_ratio)*model.module.e_prompt.prompt_key[[task_id]]
                
            else: # Directional: Converge current_prompt to similar_prompts 
                # add current_prompt to converged_p
                if (similar_tf).sum() > 0 :
                    if converged_p is None:
                        if args.distributed:
                            model.module.e_prompt.converged_p = torch.tensor([task_id])
                        else:
                            model.e_prompt.converged_p = torch.tensor([task_id])
                    else:
                        if args.distributed:
                            model.module.e_prompt.converged_p = torch.cat([converged_p, torch.tensor([task_id])])
                        else:
                            model.e_prompt.converged_p = torch.cat([converged_p, torch.tensor([task_id])])
                        
                for i, p_id in enumerate(similar_prompt_ids):
                    prompt_curr_ind = (slice(None),slice(None),[task_id])
                    prompt_p_id = (slice(None),slice(None),[p_id])
                    
                    # mix prompts
                    model.module.e_prompt.prompt.grad.zero_()
                    model.module.e_prompt.prompt[prompt_p_id] = (bwd_ratio[i])*model.module.e_prompt.prompt[prompt_curr_ind] +\
                                        (1-bwd_ratio[i])*model.module.e_prompt.prompt[prompt_p_id]
                    # mix keys
                    model.module.e_prompt.prompt_key.grad.zero_()
                    model.module.e_prompt.prompt_key[[p_id]] = (bwd_ratio[i])*model.module.e_prompt.prompt_key[[task_id]] +\
                                        (1-bwd_ratio[i])*model.module.e_prompt.prompt_key[[p_id]]
    
    else:
        converge_hist = dict()
        
    return converge_hist                

        
def uni_or_specific_prompts(model, optimizer, birch, postmerge_birch, task_id,
         converge_hist, warmup_prompts, mergable_prompts_info=None, reduce_mergable_info=None, args=None):
    """
    1. get prompt embedding
    2. BIRCH decision: by threshold
    3. update map_funtion
    
    initial warmup_prompts <- []
    """
    # from torchmetrics.functional import pairwise_cosine_similarity
    from sklearn.cluster import Birch

    if birch is None:
        birch = Birch(n_clusters=None, threshold=args.converge_thrh)
    if postmerge_birch is None and args.mergable_prompts :
        postmerge_birch = Birch(n_clusters=None, threshold=args.postmerge_thrh)
        mergable_prompts_info = ( dict(), dict(), dict(), [0] ) # mergable_hist, cluster_hist, mergableC2cands, seen
        reduce_mergable_info = ([], dict(), dict(), dict()) # blocks, block2pid, reduce_hist, task2learningblocks
        
    if converge_hist is None:
        converge_hist = dict()
        
    uni_or_spcf = None
    # at warm-up epochs!
    with torch.no_grad():    
        # 1. get similarity between prompts
        prompts = model.module.e_prompt.prompt if args.distributed else model.e_prompt.prompt
        _, _, num_tasks, _, _, _ = prompts.size() # num_layer, dual, num_tasks, length, heads, emb
        prompt = prompts.mean(0).mean(0).mean(1).view(num_tasks, -1)[[task_id],:] # prmp at task_id:(1,D) 
        
        if args.wo_normalizing_prompts: # let's be euclidean
            norm_prompt = deepcopy(prompt.detach().cpu().numpy()) # (1,D)
        else:
            norm_prompt = deepcopy(torch.nn.functional.normalize(prompt.detach().cpu(), dim=1).numpy()) # (1,D)

        ## UB of using GT of "uni_or_spcf"
        if args.uni_or_spcf_gt and args.uni_or_spcf_gt[task_id]=='uni': 
            # make all prompts specific by very low threshold & make all the same tasks universal
            real_task_id = args.task_id_list[task_id]
            overlapping_task_id = np.arange(len(args.task_id_list))[np.array(args.task_id_list) == real_task_id]
            overlapping_task_id = overlapping_task_id[0]
            print('get prompt number', overlapping_task_id)
            norm_prompt = deepcopy(warmup_prompts[overlapping_task_id])


        # 2. decision: by threshold
        birch.partial_fit(norm_prompt)
        warmup_prompts += [ norm_prompt ] # (#tasks,D)
        cluster_ids = birch.predict(np.concatenate(warmup_prompts, axis=0)) # taskID2clusterID
        
        # clusterID2promptID
        clusterID2promptID = dict()
        for prompt_id, c_id in enumerate(cluster_ids):
            if c_id not in clusterID2promptID: # newly-created cluster
                clusterID2promptID[c_id] = prompt_id
                
        # promptID_of_taskID <= (taskID2clusterID -> clusterID2promptID)
        promptID_of_taskID = clusterID2promptID[cluster_ids[-1]] 
        
        mergable_cands = None
        if args.mergable_prompts: 
            mergable_hist, cluster_hist, mergableC2cands, seen = mergable_prompts_info
            postmerge_birch.partial_fit(norm_prompt)
            postmerge_birch_cluster_ids = postmerge_birch.predict(np.concatenate(warmup_prompts, axis=0))
            num_of_tasks = len(postmerge_birch_cluster_ids)
            mergable_cands = list(np.arange(num_of_tasks)[postmerge_birch_cluster_ids==postmerge_birch_cluster_ids[-1]])
            merged_cands = list(np.arange(num_of_tasks)[cluster_ids==cluster_ids[-1]])
            
            mergable_cands = np.array(list(set(mergable_cands) - set(merged_cands))+merged_cands)  # mergable-merged
            if len(list(mergable_hist.keys())) == 0: # task_id = 0
                mergable_cands = np.array([0])
                
            if len(mergable_cands) > len(merged_cands): # insert new prompts to mergable cands
                mergable_cands = np.sort(mergable_cands)
                seen += [mergable_cands[-1]] # ignore splitting case (from uni to spcf)
                mergable_cands = [elem for elem in mergable_cands if elem in seen]
                print('mergable_cands for task {}: '.format(task_id), mergable_cands)
                mergable_hist[task_id] = (mergable_cands, 
                                    len(np.unique(cluster_ids[mergable_cands])),
                                    cluster_ids[mergable_cands])
                mergableC = postmerge_birch_cluster_ids[mergable_cands][0] 
                mergableC2cands[mergableC] = (mergable_cands, task_id)
            if task_id == 0:
                mergable_hist[task_id] = (mergable_cands, None, None)
            cluster_hist[task_id] = cluster_ids
            mergable_prompts_info = (mergable_hist, cluster_hist, mergableC2cands, seen)
            # no mergable_hist when it is not mergable

            # reference current state
            reclustered_mergable_pids, zero_covered = [], False
            blocks, block2pid, reduce_hist, task2learningblocks = reduce_mergable_info

        ####### if exists mergable cand for current task, reduce mergable
        if (args.mergable_prompts) and (task_id in mergable_hist): 
            # if not mergable (either uni or spcf), pass the below
            uni_or_spcf = None if task_id==0 else 'mergable'
            cands, num_clusters, cluster_res = mergable_hist[task_id]
            
            if task_id==0: 
                blocks += [ {0} ]
                block2pid[frozenset({0})] = 0
            else:
                cands = np.sort(cands)
                spcf_ps = np.concatenate(warmup_prompts, axis=0)[cands]
                
                min_cls_perm, min_cls, min_multi_init_res = None, 100, None
                min_multi_init_res_list = []
                min_p1_cls_perm, min_p1_cls, min_p1_multi_init_res = None, 100, None
                min_p1_multi_init_res_list = []
                cluster2silhoutte = defaultdict(list)

                for _ in range(args.sampling_size_for_remerge): # default:1000 
                    test_brc = Birch(n_clusters=None, threshold=args.converge_thrh)
                    rand_perm = np.random.permutation(spcf_ps.shape[0])
                    test_brc.fit(spcf_ps[rand_perm])
                    multi_init_res = test_brc.predict(spcf_ps)
                    pred = multi_init_res.max()

                    if pred < min_cls:
                        min_cls = pred
                        min_cls_perm = rand_perm
                        min_multi_init_res = multi_init_res
                        min_multi_init_res_list = [min_multi_init_res,]
                        min_p1_cls = min_cls+1
                        min_p1_multi_init_res_list = []

                    if pred == min_cls:
                        min_multi_init_res_list += [multi_init_res,]

                    if pred == min_p1_cls:
                        min_p1_multi_init_res_list += [multi_init_res,]

                    # get_silhoutte
                    c_num = len(np.unique(multi_init_res))
                    if c_num>1 and len(spcf_ps)>2 and len(spcf_ps)>c_num: # condition of silhouette_score
                        silhoutte_avg = silhouette_score(spcf_ps, multi_init_res)
                        cluster2silhoutte[c_num] += [silhoutte_avg]
                    
                if len(cluster2silhoutte)>0:
                    cluster2silhoutte = {c_num:sum(sil_avg)/len(sil_avg) for c_num, sil_avg in cluster2silhoutte.items()}
                    cluster2silhoutte = sorted(cluster2silhoutte.items(), key=lambda x: -x[1])
                    num_cls_with_highest_silhoutte = cluster2silhoutte[0][0]
                    # print(cluster2silhoutte, num_cls_with_highest_silhoutte)
                else:
                    num_cls_with_highest_silhoutte = None
                

                if num_cls_with_highest_silhoutte is not None:
                    if len(spcf_ps) > num_cls_with_highest_silhoutte:
                        trg_clusters = num_cls_with_highest_silhoutte+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]
                        
                    if args.kmeans_top_n>1 and (len(spcf_ps) > num_cls_with_highest_silhoutte+1):
                        trg_clusters = num_cls_with_highest_silhoutte+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]
                    
                    if args.kmeans_top_n>2 and (len(spcf_ps) > num_cls_with_highest_silhoutte+1+1):
                        trg_clusters = num_cls_with_highest_silhoutte+1+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]
                    
                    if args.kmeans_top_n>3 and (len(spcf_ps) > num_cls_with_highest_silhoutte+1+1+1):
                        trg_clusters = num_cls_with_highest_silhoutte+1+1+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]

                else:
                    if len(spcf_ps) > min_cls+1:
                        trg_clusters = min_cls+1+1
                        
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]
                        
                    if args.kmeans_top_n>1 and (len(spcf_ps) > min_cls+1+1):
                        trg_clusters = min_cls+1+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]

                    if args.kmeans_top_n>2 and (len(spcf_ps) > min_cls+1+1+1):
                        trg_clusters = min_cls+1+1+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]

                    if args.kmeans_top_n>3 and (len(spcf_ps) > min_cls+1+1+1+1):
                        trg_clusters = min_cls+1+1+1+1+1
                        for _ in range(args.kmeans_run): 
                            test_kmeans = KMeans(n_clusters=trg_clusters, n_init=5)
                            rand_perm = np.random.permutation(spcf_ps.shape[0])
                            test_kmeans.fit(spcf_ps[rand_perm])
                            multi_init_res = test_kmeans.predict(spcf_ps)

                            min_p1_multi_init_res_list += [multi_init_res,]

                # manage blocks for new clustering lists with new task: check possiblity & if possbile, insert
                num_past_blocks = len(blocks)
                perm_clusters = min_cls+1
                if perm_clusters < num_clusters: # reducable by new input order
                    min_multi_init_res_list, cnts = np.unique(np.stack(min_multi_init_res_list, axis=0), 
                                                        axis=0, return_counts=True)
                    min_multi_init_res_list, uniq_pred_ids, cnts = remove_duplicates(min_multi_init_res_list, cnts)
                    

                    # check possiblity & if possbile, insert
                    blocks, reduce_hist = manage_blocks(min_multi_init_res_list, cands, blocks, reduce_hist, task_i=task_id)

                    if len(min_p1_multi_init_res_list)>0:
                        min_p1_multi_init_res_list, cnts = np.unique(np.stack(min_p1_multi_init_res_list, axis=0),
                                                        axis=0, return_counts=True)
                        min_p1_multi_init_res_list,uniq_pred_ids, cnts = remove_duplicates(min_p1_multi_init_res_list, cnts)
                        blocks, reduce_hist = manage_blocks(min_p1_multi_init_res_list, cands, blocks, reduce_hist, task_i=task_id)
                        
                
                else:
                    print('task {}: cands {} not reduced -- from {}:{} to {}'.format(task_id, cands, num_clusters, cluster_res, perm_clusters))
                    min_multi_init_res_list, cnts = np.unique(np.stack(min_multi_init_res_list, axis=0), axis=0, 
                                                        return_counts=True)
                    min_multi_init_res_list,uniq_pred_ids, cnts = remove_duplicates(min_multi_init_res_list, cnts)
                    
                    # check possiblity & if possbile, insert
                    blocks, reduce_hist = manage_blocks(min_multi_init_res_list, cands, blocks, reduce_hist, task_i=task_id)

                    if len(min_p1_multi_init_res_list)>0:
                        min_p1_multi_init_res_list, cnts = np.unique(np.stack(min_p1_multi_init_res_list, axis=0), axis=0, 
                                                        return_counts=True)
                        min_p1_multi_init_res_list,uniq_pred_ids, cnts = remove_duplicates(min_p1_multi_init_res_list, cnts)
                        blocks, reduce_hist = manage_blocks(min_p1_multi_init_res_list, cands, blocks, reduce_hist, task_i=task_id)

                num_new_blocks = len(blocks)
                task2learningblocks[task_id] = num_new_blocks-num_past_blocks
                
                # update block2pid
                if task2learningblocks[task_id]>0:
                    new_blocks = blocks[-task2learningblocks[task_id]:]
                    for blck in new_blocks:
                        if len(blck) == 1: # spcf pid
                            block2pid[frozenset(blck)] = task_id # i <- task_id
                        else: # mergable pid
                            block2pid[frozenset(blck)] = model.module.e_prompt.mergable_prompts_start_idx if args.distributed else \
                                            model.e_prompt.mergable_prompts_start_idx
                            if args.distributed:
                                model.module.e_prompt.mergable_prompts_start_idx += 1 
                            else:
                                model.e_prompt.mergable_prompts_start_idx += 1 

                print('block2pid', block2pid)
                training_mergable_pids = [block2pid[frozenset(blck)] for blck in new_blocks]
                print('training_pid',training_mergable_pids)
                
                training_pid2base_pid = dict()
                for (blck, pid) in zip(new_blocks, training_mergable_pids):
                    if len(blck)>1:
                        _blck = deepcopy(blck)
                        _blck.remove(task_id)
                        training_pid2base_pid[pid] = block2pid[frozenset(_blck)]
                    else:
                        training_pid2base_pid[pid] = None
                print('training_pid2base_pid',training_pid2base_pid)
                
                # copy base-prompts to new prompts
                copy_base_prompts(model, optimizer, training_pid2base_pid, args)

                for c_id, cand in mergableC2cands.items():
                    cand, task_i = cand
                    origin_birch_result = len(np.unique(cluster_ids[cand]))
                    new_num_clusters = reduce_hist[task_i][0]

                    # check if new_num_clusters is reduced
                    # if origin_birch_result > new_num_clusters: # reduced by merge 
                    min_subgraph = reduce_hist[task_i][1] # subgraph of minimum cluster result
                    reclustered_mergable_pids += [block2pid[frozenset(subg)] for subg in min_subgraph]
                    zero_covered = np.array([0 in subg for subg in min_subgraph if len(subg)>1]).any()
                print('reclustered_pid', reclustered_mergable_pids)

            # update reduce_mergable_info when task_id=0 as well
            reduce_mergable_info = (blocks, block2pid, reduce_hist, task2learningblocks)
            
    ### adjust converge_hist ###
        
    # 3. if merge, update map_funtion
    converge_hist[task_id] = [promptID_of_taskID]
    print( 'task {} coverged with '.format(task_id), promptID_of_taskID )
    print('without mergable, birch clusters: ', cluster_ids)
    if args.mergable_prompts and task_id>0: # report reclustering results 
        rest_tasks = np.arange(task_id+1)
        reducing_cands = []
        remerging_cluster_id = args.num_tasks
        for c_id, cand in mergableC2cands.items():
            cand, task_i = cand
            origin_birch_result = len(np.unique(cluster_ids[cand]))
            new_num_clusters = reduce_hist[task_i][0]
            
            # check if new_num_clusters is reduced
            if origin_birch_result > new_num_clusters: # reduced by merge
                rest_tasks = np.setdiff1d(rest_tasks, cand)
                reducing_cands += [new_num_clusters]
                min_subgraph = reduce_hist[task_i][1] # e.g.,  [{0, 3, 4, 7, 8}, {1, 2, 5, 6}]
                print('min_subgraph', min_subgraph)
                for subg in min_subgraph: 
                    for t in subg:
                        converge_hist[t] = [remerging_cluster_id] # assign new cluster id
                    remerging_cluster_id += 1

        print('original #p to reduced #p: from {} to {}'.format(len(np.unique(cluster_ids)),
                                        len(np.unique(cluster_ids[rest_tasks])) + sum(reducing_cands)  ) )
        
        

    # converged_p={taskid-promptid}: add promptID_of_taskID 
    if uni_or_spcf is None: # => uni_or_spcf is not mergable 
        left_epochs = args.spcf_epochs if promptID_of_taskID == task_id else args.uni_epochs
        uni_or_spcf = 'spcf' if promptID_of_taskID == task_id else 'uni'
        model.module.e_prompt.mergable_task = False
            
        
        if args.distributed:
            model.module.e_prompt.converged_p = torch.cat([model.module.e_prompt.converged_p,
                                                        torch.tensor([promptID_of_taskID])])
            reclustered_mergable_pids = model.module.e_prompt.reclustered_mergable_pids if args.mergable_prompts \
                                            else None
        else:
            model.e_prompt.converged_p = torch.cat([model.e_prompt.converged_p, 
                                                    torch.tensor([promptID_of_taskID])])
            reclustered_mergable_pids = model.e_prompt.reclustered_mergable_pids if args.mergable_prompts else None
        
            
    else: # => uni_or_spcf = 'mergable'
        model.module.e_prompt.mergable_task = True 
        if len(training_mergable_pids) > 1:
            left_epochs = min(args.mergable_epochs*len(training_mergable_pids), args.max_mergable_epochs) 
        else:
            left_epochs = args.max_mergable_epochs
        

        # converged_p <- insert -1; send training_pid & reclustered_pid to e_prompt 
        if args.distributed:
            model.module.e_prompt.converged_p = torch.cat([model.module.e_prompt.converged_p,
                                                                torch.tensor([-1])])
            model.module.e_prompt.training_mergable_pids = training_mergable_pids
            model.module.e_prompt.reclustered_mergable_pids = reclustered_mergable_pids
        else:
            model.e_prompt.converged_p = torch.cat([model.e_prompt.converged_p, 
                                                            torch.tensor([-1])])
            model.e_prompt.training_mergable_pids = training_mergable_pids
            model.e_prompt.reclustered_mergable_pids = reclustered_mergable_pids

    if task_id == 0:
        left_epochs = args.left_1Tepochs # 95 
        

    if args.mergable_prompts: # converged_prompts for inference
        conv_p = torch.cat( (model.module.e_prompt.converged_p[model.module.e_prompt.converged_p!=-1], 
                                torch.tensor(reclustered_mergable_pids)) )
        conv_p = torch.unique(conv_p)
        if zero_covered: # remove 0 when 0 is already covered by `reclustered_mergable_pids`
            conv_p = conv_p[conv_p!=0]
        if task_id>0:
            if len(rest_tasks) > 0: # `rest_tasks`: uni or un-reduced mergable cands 
                # include uni prompts (not mergable cands)
                conv_p = torch.unique(torch.cat((conv_p,model.module.e_prompt.converged_p[rest_tasks])))
                # remove mergable cands b/c they are already considered
                conv_p = conv_p[conv_p!=-1]
        if args.distributed:
            model.module.e_prompt.pids4inf = conv_p
        else:
            model.e_prompt.pids4inf = conv_p
           
    return converge_hist, warmup_prompts, birch, left_epochs, uni_or_spcf, postmerge_birch, mergable_cands, \
            mergable_prompts_info, reduce_mergable_info

def copy_base_prompts(model, optimizer, training_pid2base_pid, args):
    for dest_pid, src_pid in training_pid2base_pid.items():
        if src_pid is not None: # not specific one, but mergable one
            print('copy from prompt {} to prompt {}'.format(src_pid, dest_pid))
            # Transfer previous learned prompt base to the new prompt
            src_start, src_end, dest_start, dest_end = src_pid, src_pid + 1, dest_pid, dest_pid + 1
            src_idx = (slice(None), slice(None), slice(src_start, src_end)) 
            dest_idx = (slice(None), slice(None), slice(dest_start, dest_end)) 

            with torch.no_grad():
                if args.distributed:
                    model.module.e_prompt.prompt.grad.zero_()
                    model.module.e_prompt.prompt[dest_idx] = model.module.e_prompt.prompt[src_idx]
                    optimizer.param_groups[0]['params'] = model.module.parameters()
                else:
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[dest_idx] = model.e_prompt.prompt[src_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
            
            # Transfer previous learned prompt base keys to the new prompt
            src_idx = (slice(src_start, src_end)) 
            dest_idx = (slice(dest_start, dest_end)) 
            with torch.no_grad():
                if args.distributed:
                    model.module.e_prompt.prompt_key.grad.zero_()
                    model.module.e_prompt.prompt_key[dest_idx] = model.module.e_prompt.prompt_key[src_idx]
                    optimizer.param_groups[0]['params'] = model.module.parameters()
                else:
                    model.e_prompt.prompt_key.grad.zero_()
                    model.e_prompt.prompt_key[dest_idx] = model.e_prompt.prompt_key[src_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()      
              
def manage_blocks(res_list, cands, blocks, reduce_hist, task_i):
    adding_blocks = []
    for res in res_list:
        print('res', res)
        
        ### check if min_res_list is possible?
        check_res = res[:-1]

        g = net.Graph()
        g.add_nodes_from(cands[:-1])

        for uniq in np.unique(check_res):
            path = cands[:-1][check_res==uniq]
            if len(path)>1:
                net.add_path(g, list(path))

        all_nodes = list(cands[:-1])
        subgraph = []
        while all_nodes:
            node = all_nodes[0]
            descendants = net.descendants(g, node)
            if len(descendants)>0:
                for descend in descendants:
                    all_nodes.remove(descend)
            all_nodes.remove(node)

            descendants.add(node)
            subgraph += [descendants]

        # print('subgraph', subgraph)
        tf = True
        for subg in subgraph:
            tf = subg in blocks
            # print(subg, blocks, tf)
            if tf == False:
                break
        if tf == False:
            print('no match in ', subg)
            continue

        ### if min_res_list is possible, insert to list
        path = cands[res==res[-1]]
        g.add_nodes_from(cands[[-1]])
        net.add_path(g, list(path))
        descendants = net.descendants(g, cands[-1])
        descendants.add(cands[-1])
        if descendants not in blocks:
            blocks += [descendants]
            adding_blocks += [descendants]
        print('==> blocks', blocks, len(blocks))

        # get subgraph based on the minimum possible cluster results
        all_nodes, subgraph = list(cands), []
        while all_nodes:
            node = all_nodes[0]
            descendants = net.descendants(g, node)
            if len(descendants)>0:
                for descend in descendants:
                    all_nodes.remove(descend)
            all_nodes.remove(node)

            descendants.add(node)
            subgraph += [descendants]
        
        if (task_i not in reduce_hist):
            reduce_hist[task_i] = (len(np.unique(res)), subgraph)
        elif reduce_hist[task_i][0] > len(np.unique(res)):
            reduce_hist[task_i] = (len(np.unique(res)), subgraph)
            
    print('adding_blocks: ', len(adding_blocks), adding_blocks)
    
    return blocks, reduce_hist

def remove_duplicates(predictions, counts):
    
    cut = len(predictions[0]) 
    adjs = []
    uniq_adjs = []
    
    for i in range(len(predictions)):
        g = net.Graph()
        g.add_nodes_from(list(range(cut)))
        for uniq in np.unique(predictions[i]):
            path = np.arange(cut)[predictions[i]==uniq]
            if len(path)>1:
                net.add_path(g, list(path))
        # print(net.adjacency_matrix(g).todense())
        adjs += [np.array(net.adjacency_matrix(g).todense())]
        
    num_diffs = [[0 for _ in range(len(adjs))] for _ in range(len(adjs))]
    for i, adj_i in enumerate(adjs):
        for j, adj_j in enumerate(adjs):
            num_diffs[i][j] = (adj_i != adj_j).sum()//2
            # print((adj_i != adj_j).sum()//2)
    num_diffs = np.array(num_diffs)
    
    preds = list(range(num_diffs.shape[0]))
    uniq_preds = []
    agg_cnts = []
    while preds:
        pred_idx = preds[0]
        same_pred_inds = np.arange(num_diffs.shape[0])[num_diffs[pred_idx] == 0] 
        uniq_preds += [same_pred_inds[0]]
        agg_cnts += [ sum( [counts[same_ind] for same_ind in same_pred_inds] ) ]
        for _ in same_pred_inds:
            preds.remove(_) 
    
    return np.array(predictions)[uniq_preds], uniq_preds, agg_cnts
