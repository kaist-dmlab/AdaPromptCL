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
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)# where the distribution of gpus is set

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)
    
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes  if not args.clip_text_head else 512,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args
    )
    if args.pt_augmented_ptm:
        original_model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes if not args.clip_text_head else 512,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            prompt_length=args.length,
            embedding_key=args.embedding_key,
            prompt_init=args.prompt_key_init,
            prompt_pool=args.prompt_pool,
            prompt_key=args.prompt_key,
            pool_size=args.size,
            top_k=args.top_k,
            batchwise_prompt=args.batchwise_prompt,
            prompt_key_init=args.prompt_key_init,
            head_type=args.head_type,
            use_prompt_mask=args.use_prompt_mask,
            use_g_prompt=args.use_g_prompt,
            g_prompt_length=args.g_prompt_length,
            g_prompt_layer_idx=args.g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
            use_e_prompt=args.use_e_prompt,
            e_prompt_layer_idx=args.e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
            same_key_value=args.same_key_value,
            eval_prototype_clf=args.eval_prototype_clf,
            original_model=True,
            args=args
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes if not args.clip_text_head else 512,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        eval_prototype_clf=args.eval_prototype_clf,
        args=args
    )
    original_model.to(device)
    model.to(device)  
    
    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        if args.clip_text_head:
            args.freeze += ['head']
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)): # if layer is in args.freeze, then freeze
                p.requires_grad = False
    
    if args.ptm_load_msd:
        checkpoint_path = os.path.join(args.load_dir, 'checkpoint/task1_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print('Loading checkpoint from:', checkpoint_path)
            checkpoint = torch.load(checkpoint_path, )
            original_model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
    
    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _, _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    
    ema_model = None

    model_without_ddp = model
    if args.distributed:
        # print("distributed True")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                    find_unused_parameters=False,
                                                    # True if not args.clip_text_head else False
                                                    )
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    print("args.scale_lr", args.scale_lr, args.world_size)
    if args.scale_lr:
        global_batch_size = args.batch_size * args.world_size
    else:
        global_batch_size = args.batch_size
    args.lr = args.lr * (global_batch_size*args.accum_step) / 256.0
    # print(args.lr)
    print(args)

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
        print(lr_scheduler)
    elif args.sched == 'constant':
        lr_scheduler = None

    if args.task_agnostic_head:
        criterion = torch.nn.BCELoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # for i, (n, p) in enumerate(model_without_ddp.named_parameters()):
    #     print(i, n)
    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, ema_model, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    
    config = parser.parse_known_args()[-1][0]
    print(config)
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
    elif config == 'vtab_dualprompt':
        from configs.vtab_dualprompt import get_args_parser
        config_parser = subparser.add_parser('vtab_dualprompt', help='VTAB DualPrompt configs')
    elif config == 'overlap_vtab_dualprompt':
        from configs.overlap_vtab_dualprompt import get_args_parser
        config_parser = subparser.add_parser('overlap_vtab_dualprompt', help='overlap VTAB DualPrompt configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # print(args)


    main(args)
    
    sys.exit(0)