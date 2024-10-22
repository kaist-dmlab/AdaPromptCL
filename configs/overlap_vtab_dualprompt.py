import argparse

def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=50, type=int)

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--input-size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.025, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--scale_lr', default=True, type=bool,)

    # Augmentation parameters
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='overlapping-vtab-1k', type=str, help='dataset name')
    # subparsers.add_argument('--shuffle', default=True, help='shuffle the data order')
    subparsers.add_argument('--shuffle', action='store_false',
                                        help='if specified, shuffle = false')    
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=42, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--num_workers', default=4, type=int)
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    subparsers.add_argument('--dist_backend', default='nccl', help='default:nccl')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=19, type=int, help='number of sequential tasks')
    # subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--train_mask', action='store_false',
                                        help='if specified, train_mask = false')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # G-Prompt parameters
    # subparsers.add_argument('--use_g_prompt', default=True, type=bool, help='if using G-Prompt')
    subparsers.add_argument('--use_g_prompt', action='store_true', help='Use G prompt')
    subparsers.add_argument('--g_prompt_length', default=5, type=int, help='length of G-Prompt')
    subparsers.add_argument('--g_prompt_layer_idx', default=[], type=int, nargs = "+", help='the layer index of the G-Prompt')
    subparsers.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool, help='if using the prefix tune for G-Prompt')

    
    # E-Prompt parameters
    # subparsers.add_argument('--use_e_prompt', default=False, type=bool, help='if using the E-Prompt')
    subparsers.add_argument('--use_e_prompt', action='store_true', help='Use E prompt')
    subparsers.add_argument('--e_prompt_layer_idx', default=[], type=int, nargs = "+", help='the layer index of the E-Prompt')
    subparsers.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool, help='if using the prefix tune for E-Prompt')

    # Use prompt pool in L2P to implement E-Prompt
    subparsers.add_argument('--prompt_pool', default=True, type=bool,)
    subparsers.add_argument('--size', default=300, type=int,)
    subparsers.add_argument('--length', default=8,type=int, )
    subparsers.add_argument('--top_k', default=1, type=int, )
    subparsers.add_argument('--initializer', default='uniform', type=str,)
    subparsers.add_argument('--prompt_key', default=True, type=bool,)
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str)
    subparsers.add_argument('--use_prompt_mask', default=True, type=bool)
    subparsers.add_argument('--mask_first_epoch', default=False, type=bool)
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool)
    # subparsers.add_argument('--shared_prompt_pool', action='store_false',
    #                                     help='if specified, shared_prompt_pool = false')    
    subparsers.add_argument('--shared_mean', action='store_true', help='Transfer mean of prompts')
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
    subparsers.add_argument('--batchwise_prompt', default=False, type=bool)
    # subparsers.add_argument('--batchwise_prompt', action='store_false',
    #                                     help='if specified, batchwise_prompt = false')
    subparsers.add_argument('--embedding_key', default='cls', type=str)
    subparsers.add_argument('--predefined_key', default='', type=str)
    subparsers.add_argument('--pull_constraint', default=True)
    subparsers.add_argument('--pull_constraint_coeff', default=1.0, type=float)
    subparsers.add_argument('--same_key_value', default=False, type=bool)

    # CODA-prompt
    subparsers.add_argument('--coda_prompt', action='store_true', help='Use CODA-prompt: feature prompt')
    subparsers.add_argument('--num_coda_prompt', default=10, type=int,) # bc of pool_sz=10, num_coda_prompt=100

    # feature-prompt
    subparsers.add_argument('--feat_prompt', action='store_true', help='feature prompt')
    subparsers.add_argument('--num_feat_prompt', default=100, type=int,) # bc of pool_sz=10, num_coda_prompt=100
    subparsers.add_argument('--orthogonal_coeff', default=1.0, type=float)
    subparsers.add_argument('--softmax_prompt', action='store_true', help='softmax prompt')
    subparsers.add_argument('--save_attn_scores', action='store_true', help='save attn scores')
    subparsers.add_argument('--save_attn_period', default=2, type=int,)

    # force training on copied prompts
    subparsers.add_argument('--keep_prompt_freq', action='store_true', help='keep prompt freq')
    subparsers.add_argument('--feat_attn_mask', action='store_true', help='keep prompt freq')

    # copy top2bottom
    subparsers.add_argument('--copy_top2bottom', action='store_true', help='copy top2bottom')
    subparsers.add_argument('--task_free', action='store_true', help='task_free')
    subparsers.add_argument('--copy_thrh', type=float, default=0.05, help='prob of copying top2bottom')
    subparsers.add_argument('--pct_copy', type=float, default=0.1, help='pct of copying bottom')
    subparsers.add_argument('--compress_ratio', type=int, default=1, help='ratio of compressing top')
    subparsers.add_argument('--top_ratio', type=float, default=1., help='coef of copying top')
    subparsers.add_argument('--freezeQ', action='store_true', help='freezeQ')
    subparsers.add_argument('--freezeV', action='store_true', help='freezeV')
    subparsers.add_argument('--freezeK', action='store_true', help='freezeK')
    # waiting room settings
    subparsers.add_argument('--wr_prompt', action='store_true', help='waiting room prompts')
    subparsers.add_argument('--num_wr_prompt', default=0, type=int,) 
    
    # supcon on prompts & attns
    subparsers.add_argument('--supcon_prompts', action='store_true', help='supervised contrastive loss on prompts')
    subparsers.add_argument('--supcon_attns', action='store_true', help='supervised contrastive loss on attn-scores')
    subparsers.add_argument('--loss_worldsz', action='store_true', help='multiply loss by world size')
    subparsers.add_argument('--prev_cls_prompts', action='store_true', help='keep prompts of prev classes')
    subparsers.add_argument('--prev_cls_attns', action='store_true', help='keep attns of prev classes')
    subparsers.add_argument('--not_use_qnet', action='store_true', help='use q projection head')
    subparsers.add_argument('--not_use_qnet_attns', action='store_true', help='use q projection head')
    subparsers.add_argument('--q_units', default=500, type=int,)  
    subparsers.add_argument('--q_units_attns', default=500, type=int,)  
    subparsers.add_argument('--supcon_temp', type=float, default=0.1, help='temperature')
    subparsers.add_argument('--supcon_temp_attns', type=float, default=0.1, help='temperature')
    subparsers.add_argument('--coef_supcon', type=float, default=0.1, help='coefficient of supconloss')
    subparsers.add_argument('--coef_supcon_attns', type=float, default=0.1, help='coefficient of supconloss on attns')
    subparsers.add_argument('--prompt_ema_ratio', type=float, default=0.9, help='coefficient of ema')
    subparsers.add_argument('--attns_ema_ratio', type=float, default=0.9, help='coefficient of ema')
    # supcon on partial layers
    subparsers.add_argument('--partial_layers', default=[], nargs='+', type=int, help='layers to index')
    # supcon on aggragated prompts on dims
    subparsers.add_argument('--agg_dims', default=[], nargs='+', type=list, help='dimensions to aggregate')
    # anneal coef of supcon
    subparsers.add_argument('--anneal_supcon', action='store_true', help='anneal coef of supcon')
    subparsers.add_argument('--func_anneal', default='linear', type=str,) # 'linear' 'exp' 'log'
    subparsers.add_argument('--dest_coef_supcon', type=float, default=0.1, help='coefficient of supconloss')
    
    # contrastive loss
    subparsers.add_argument('--cl_prompts', action='store_true', help='contrastive loss on prompts')
    subparsers.add_argument('--cl_randaug', action='store_true', help='use randaug for contrastive loss')

    # supcls_prompts
    subparsers.add_argument('--supcls_prompts', action='store_true', help='prompt groups based on superclass')
    subparsers.add_argument('--supcls_eval_known', action='store_true', help='supcls known at evaluation')
    subparsers.add_argument('--shared_prompt', action='store_true', help='use waiting room at supcls prompts')
    subparsers.add_argument('--sep_shared_prompt', action='store_true', help='use waiting room at supcls prompts')
    subparsers.add_argument('--shared_prompt_ema_ratio', type=float, default=0.9, help='coefficient of ema')
    subparsers.add_argument('--only_shared_prompt', action='store_true', help='use waiting room at supcls prompts')
    subparsers.add_argument('--one_shared_prompt', action='store_true', help='use waiting room at supcls prompts')
    subparsers.add_argument('--num_updates', default=1, type=int, )
    # prototype_clf
    subparsers.add_argument('--eval_prototype_clf', action='store_true', help='prototype classifier')
    subparsers.add_argument('--prototype_ema_ratio', type=float, default=0.9, help='coefficient of ema')

    # original_model <- pt+ptm
    subparsers.add_argument('--pt_augmented_ptm', action='store_true', help='original_model <- pt+ptm')
    subparsers.add_argument('--ptm_load_msd', action='store_true', help='load pretrained state dict')
    subparsers.add_argument('--load_dir', default='sth', type=str, help='MSD load directory')
    subparsers.add_argument('--ptm_num_feat_prompt', default=10, type=int,) # bc of pool_sz=10, num_coda_prompt=100

    # normalize_prompt
    subparsers.add_argument('--normalize_prompt', action='store_true', help='normalize prompt')
    
    # specific-prompts
    subparsers.add_argument('--specific_prompts', action='store_true', help='specific prompts')
    subparsers.add_argument('--s_prompts_layer_idx', default=[], nargs='+', type=int, help='layer idx')
    subparsers.add_argument('--s_prompts_num_prompts', default=5, type=int,)
    subparsers.add_argument('--s_prompts_length', default=1, type=int,)
    subparsers.add_argument('--s_prompts_add', default='cat', type=str, help='{"cat","add",}')

    # evolve-prompts: data_driven_evolve, bidirectional, normalize_by_sim, bwd_ratio, fwd_ratio
    subparsers.add_argument('--data_driven_evolve', action='store_true', help='data_driven_evolve prompts')
    subparsers.add_argument('--uni_or_specific', action='store_true', help='uni_or_specific prompts')
    subparsers.add_argument('--wo_normalizing_prompts', action='store_true', help='without normalizing prompts')
    subparsers.add_argument('--compare_after_warmup', action='store_true', help='compare_after_warmup')
    subparsers.add_argument('--evolve_epoch', default=19, type=int,) 
    subparsers.add_argument('--left_1Tepochs', default=0, type=int,)
    subparsers.add_argument('--uni_or_spcf_gt', action='store_true', help='uni_or_spcf_gt')

    # film
    subparsers.add_argument('--film_train', action='store_true', help='train film')
    subparsers.add_argument('--nofilm', action='store_true', help='offset film effect')
    subparsers.add_argument('--film_combine', default='mean', type=str, help='{"cat","mean",}')
    subparsers.add_argument('--film_train_epoch_add', action='store_true', help='film_train_epoch_add')
    subparsers.add_argument('--film_train_epoch', default=10, type=int,) 
    
    # post-merge
    subparsers.add_argument('--postmerge_thrh', type=float, default=0.5, help='if > thrh, then converge')
    subparsers.add_argument('--mergable_prompts', action='store_true', help='train ')
    subparsers.add_argument('--spcf_epochs', default=100, type=int,)
    subparsers.add_argument('--uni_epochs', default=5, type=int,)
    subparsers.add_argument('--mergable_epochs', default=5, type=int,) 
    subparsers.add_argument('--max_mergable_epochs', default=50, type=int,) 
    subparsers.add_argument('--reinit_clf', default=True, type=bool)
    subparsers.add_argument('--sampling_size_for_remerge', default=1000, type=int,) 
    subparsers.add_argument('--save_prepgroup_info', action='store_true', help='save_prepgroup')
    subparsers.add_argument('--kmeans_run', default=100, type=int,) 
    subparsers.add_argument('--kmeans_top_n', default=2, type=int,) 

    # subparsers.add_argument('--film_intertwined_train', action='store_true', help='train intertwined film')
    # subparsers.add_argument('--film_inference', action='store_true', help='infer with film')
    
    subparsers.add_argument('--converge_thrh', type=float, default=0.3, help='if > thrh, then converge')
    subparsers.add_argument('--bidirectional', action='store_true', help='bidirectional converge')
    subparsers.add_argument('--normalize_by_sim', action='store_true', help='normalize_by_sim in compressing')
    subparsers.add_argument('--mix_with_sim', action='store_true', help='mix with similarity')
    subparsers.add_argument('--mix_max_sim', action='store_true', help='mix with max similarity')
    subparsers.add_argument('--mix_mean_sim', action='store_true', help='mix with mean similarity')
    subparsers.add_argument('--bwd_ratio', type=float, default=0.3, help='bwd: curr -> prev')
    subparsers.add_argument('--fwd_ratio', type=float, default=0.3, help='bwd: prev -> curr')

    subparsers.add_argument('--save_sdict', action='store_true', help='save model state')
    subparsers.add_argument('--save_sdict_by_epochs', action='store_true', help='save model state')
    subparsers.add_argument('--save_sdict_epochs', default=[], nargs='+', type=int, help='save_sdict_epochs')
    subparsers.add_argument('--save_warmup_prompts', action='store_true', help='save warmup prompts')

    # vtab datasets
    subparsers.add_argument('--vtab_datasets', default=[], nargs='+', type=str, help='vtab_datasets')
    # vtab similar tasks 
    subparsers.add_argument('--milder_vtab', action='store_true', help='milder_vtab')
    subparsers.add_argument('--no_mild_tasks', action='store_true', help='no_mild_tasks')
    subparsers.add_argument('--num_no_mild_tasks', default=8, type=int,)
    subparsers.add_argument('--clustering_based_vtabgroups', action='store_true', help='clustering_based_vtabgroups')
    subparsers.add_argument('--vtab_group_order', action='store_true', help='vtab_group_order')

    subparsers.add_argument('--milder_by_alldts', action='store_true', help='milder_by_alldts')
    subparsers.add_argument('--overlap_similarity', type=float, default=50, help='')

                            
    # num_overlaps, overlap_dataset_scale, num_overlapping_tasks, shuffle_overlaps
    subparsers.add_argument('--num_overlaps', default=1, type=int,)
    subparsers.add_argument('--num_overlapping_tasks', default=1, type=int,)
    # subparsers.add_argument('--shuffle_overlaps', action='store_true', help='shuffle_overlaps')
    subparsers.add_argument('--shuffle_overlaps', type=bool, default=True, help='shuffle_overlaps')
    subparsers.add_argument('--shuffle_overlaps_inds', default=[], nargs='+', type=int, help='shuffle_overlaps')
    subparsers.add_argument('--shuffle_prompt_inds', default=[], nargs='+', type=int, help='shuffle_prompt_inds')
    subparsers.add_argument('--overlap_dataset_scale', type=float, default=1.0, help='')
    subparsers.add_argument('--overlap_datasets', default=[], nargs='+', type=str, help='vtab_datasets')
    # subparsers.add_argument('--prompt_same_init', action='store_true', help='prompt_same_init')
    subparsers.add_argument('--prompt_same_init', type=bool, default=True, help='prompt_same_init')

    
    
    # adam_freeze: freeze E-prompts
    subparsers.add_argument('--freezeE', action='store_true', help='freeze E promptss')
    subparsers.add_argument('--freezeE_zerograd', action='store_true', help='freeze E promptss')
    subparsers.add_argument('--freezeE_t', default=-2, type=int,)
    
    subparsers.add_argument('--freezeH', action='store_true', help='freeze head')
    subparsers.add_argument('--freezeH_t', default=-2, type=int,)
    
    subparsers.add_argument('--merge_pt', action='store_true', help='merge ptm and tuned model')

    # Use clip embedding + G-Prompt
    subparsers.add_argument('--clip_emb', default=False, type=bool,)
    subparsers.add_argument('--text_img_prob', type=float, default=0.5, help='prob for text-emb; (1-prob) for img-emb')
    subparsers.add_argument('--grad_accum', default=False, type=bool,)
    subparsers.add_argument('--accum_step', default=1, type=int, )
    
    subparsers.add_argument('--freezeG', action='store_true', help='freezeG or not')
    subparsers.add_argument('--freezeG_t', default=-2, type=int,)
    subparsers.add_argument('--freezeG_e', default=-2, type=int,)
    subparsers.add_argument('--freeze_clipproj', default=False, type=bool,)
    subparsers.add_argument('--freeze_clipproj_t', default=1, type=int,)

    # Use task_agnostic_head
    subparsers.add_argument('--task_agnostic_head', default=False, type=bool,)

    # Use Clip-text-encoder-Head + clip-embedding + G-Prompt
    subparsers.add_argument('--clip_text_head', default=False, type=bool,)
    
    # Inaccurate task boundary information
    subparsers.add_argument('--noisy_task_boundary', action='store_true', help='Noisy task boundary')
    subparsers.add_argument('--noisy_pct', type=float, default=0.1, help='Noise percentage')
    subparsers.add_argument('--noisy_type', default='start', type=str, help='{"start","scattered","uniform"}')
    subparsers.add_argument('--noisy_policy', default='1-prev', type=str,
                 help='"k-prev" or "all-prev"')
    # Small dataset
    subparsers.add_argument('--small_dataset', action='store_true', help='Use small dataset')
    subparsers.add_argument('--small_dataset_scale', type=float, default=1.0, help='dataset percentage')

    # Use noisy labels
    subparsers.add_argument('--noisy_labels', action='store_true', help='Noisy labels')
    subparsers.add_argument('--in_task_noise', action='store_true', help='Noisy labels within a task')
    subparsers.add_argument('--noisy_labels_type', default='symmetry', type=str, help='"symmetry" or "pair"')
    subparsers.add_argument('--noisy_labels_rate', type=float, default=0.1, help='Noise percentage')


    # Use blurry CL
    subparsers.add_argument('--blurryCL', action='store_true', help='blurryCL')
    subparsers.add_argument('--blurryM', type=float, default=0.9, help='Major classes percentage')
    subparsers.add_argument('--balanced_softmax', action='store_true', help='Balanced softmax')
    subparsers.add_argument('--online_cls_mask', action='store_true', help='online_cls_mask')

    # Separate specializations
    subparsers.add_argument('--sep_specialization', action='store_true', help='Separate specialization')
    subparsers.add_argument('--sep_criterion', default='random', type=str, help='separate criterion')
    subparsers.add_argument('--wo_sep_mask', action='store_true', help='wo Separation mask')
    subparsers.add_argument('--task_lvl_sep_mask', action='store_true', help='wo Separation mask')

    subparsers.add_argument('--wo_sep_mask_e', default=-1, type=int,) # Def: No use sep_mask
    subparsers.add_argument('--w_sep_mask_e', default=-1, type=int,) # Def: No use sep_mask

    subparsers.add_argument('--wo_sep_mask_t', default=-1, type=int,) # Def: No use sep_mask
    subparsers.add_argument('--w_sep_mask_t', default=-1, type=int,) # Def: No use sep_mask

    subparsers.add_argument('--eval_known_prompt', action='store_true', help='Known prompt at eval')
    subparsers.add_argument('--eval_only_acc', action='store_true', help='')

    # Evaluate anytime-inference
    subparsers.add_argument('--anytime_inference', action='store_true', help='Metric: anytime inference')
    subparsers.add_argument('--anytime_inference_period', default=4, type=int,)

    # Use only G-Prompt
    subparsers.add_argument('--only_G', action='store_true', help='only_G')


    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')

    subparsers.add_argument('--freezeE_e', default=-2, type=int,)
    subparsers.add_argument('--unfreezeE', action='store_true', help='freeze E promptss')
    subparsers.add_argument('--unfreezeE_e', default=-2, type=int,)
