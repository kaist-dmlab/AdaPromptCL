import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from pytorch_metric_learning import losses
import torch.distributed as dist
from continual_datasets.dataset_utils import get_cls2supcls_cifar100, get_cls2supcls_imr
import numpy as np

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 coda_prompt=False, num_coda_prompt=10, feat_prompt=False, num_feat_prompt=100,
                 softmax_feat=False, original_model=False, args=None):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        self.coda_prompt = coda_prompt
        self.num_coda_prompt = num_coda_prompt

        self.feat_prompt = feat_prompt
        self.num_feat_prompt = num_feat_prompt
        self.softmax_feat = softmax_feat
        self.orthogonal_coeff = args.orthogonal_coeff
        if self.softmax_feat:
            self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.save_attn_scores = args.save_attn_scores

        
        self.copy_top2bottom = args.copy_top2bottom
        self.feat_attn_mask = args.feat_attn_mask

        self.keep_prompt_freq = args.keep_prompt_freq
        self.task_free = args.task_free

        self.with_wr_prompt = args.wr_prompt
        self.num_wr_prompt = args.num_wr_prompt
        self.wr_s = 0
        
        self.cl_prompts = args.cl_prompts
        
        self.supcon_prompts = args.supcon_prompts
        self.supcon_attns = args.supcon_attns
        self.prev_cls_prompts = args.prev_cls_prompts
        self.prev_cls_attns = args.prev_cls_attns
        self.loss_worldsz = args.loss_worldsz
        self.partial_layers = args.partial_layers
        strdim2intdim = {
            'dual':1, 'layers':2, 'length':3, 'heads':4, 'emb':5, 
        }
        args.agg_dims = [''.join(_) for _ in args.agg_dims]
        # print('args.agg_dims', args.agg_dims)
        self.agg_dims = [strdim2intdim[strdim] for strdim in args.agg_dims]
        # print('self.agg_dims', self.agg_dims)
        
        self.specific_prompts = args.specific_prompts
        self.s_prompts_layer_idx = args.s_prompts_layer_idx
        self.s_prompts_num_prompts = args.s_prompts_num_prompts
        self.s_prompts_length = args.s_prompts_length
        self.s_prompts_num_layers = len(self.s_prompts_layer_idx)
        self.s_prompts_add = args.s_prompts_add
        
        if self.copy_top2bottom:
            self.keep_prompt_freq = True
            self.copy_thrh = args.copy_thrh
            self.pct_copy = args.pct_copy
            self.task_free = args.task_free
            self.attn_mask = None # torch.ones(args.num_feat_prompt, device=device)
        
        if self.keep_prompt_freq:
            self.freq_dict_prev = None
            self.freq_dict_curr = torch.ones(num_feat_prompt \
                                             if feat_prompt else self.pool_size,
                                              device=torch.device(args.device))
            # +self.num_wr_prompt            
            self.task_id = -1

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt: # set to true
                assert embed_dim % self.num_heads == 0
                if self.same_key_value: # set to false
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else: # set to true
                    if self.coda_prompt:
                        # 2: key and val?
                        prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.num_coda_prompt, self.length, 
                                            self.num_heads, embed_dim // self.num_heads)
                    elif self.feat_prompt:
                        if self.specific_prompts:
                            prompt_pool_shape = (self.s_prompts_num_layers, 2, self.s_prompts_num_prompts, 
                                    self.s_prompts_length, self.num_heads, embed_dim // self.num_heads)
                        else: # 
                            prompt_pool_shape = (self.num_layers, 2, self.num_feat_prompt, self.length, 
                                                self.num_heads, embed_dim // self.num_heads)
                    else:
                        prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                            self.num_heads, embed_dim // self.num_heads)
                    
                    if args.prompt_same_init:
                        prompt_pool_shape = (self.num_layers, 2, 1, self.length, self.num_heads,
                                        embed_dim // self.num_heads) # 1: self.pool_size
                        self.prompt = torch.randn(prompt_pool_shape) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads 
                        nn.init.uniform_(self.prompt, -1, 1)
                        print(self.pool_size)
                        self.prompt = torch.nn.Parameter(self.prompt.repeat(1,1,self.pool_size,1,1,1))
                    else: 
                        if prompt_init == 'zero':
                            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                        elif prompt_init == 'uniform': # set to true
                            if len(args.shuffle_prompt_inds)>0: # shuffle prompts ids
                                self.prompt = torch.randn(prompt_pool_shape) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                                nn.init.uniform_(self.prompt, -1, 1)
                                self.prompt = torch.nn.Parameter(self.prompt[:,:,args.shuffle_prompt_inds])
                                
                            else:
                                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                                nn.init.uniform_(self.prompt, -1, 1)
                        
                    # when specific_prompts, init shared_prompt
                    if self.specific_prompts: # num_feats = 10; but only using the 1st prompt
                        shared_prompt_pool_shape = (self.num_layers, 2, 10,
                                            self.length, self.num_heads, embed_dim // self.num_heads)
                        if prompt_init == 'zero':
                            self.one_prompt = nn.Parameter(torch.zeros(shared_prompt_pool_shape))
                        elif prompt_init == 'uniform': # set to true
                            self.one_prompt = nn.Parameter(torch.randn(shared_prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                            nn.init.uniform_(self.one_prompt, -1, 1)
                            
                    # Use waiting room prompts 
                    if self.with_wr_prompt:
                        wr_prompt_pool_shape = (self.num_layers, 2, self.num_wr_prompt,
                                            self.length, self.num_heads, embed_dim // self.num_heads)
                        if prompt_init == 'zero':
                            self.wr_prompt = nn.Parameter(torch.zeros(wr_prompt_pool_shape))
                        elif prompt_init == 'uniform': # set to true
                            self.wr_prompt = nn.Parameter(torch.randn(wr_prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                            nn.init.uniform_(self.wr_prompt, -1, 1)
                        self.wr_prompt.requires_grad = False
                        self.wr_prompt_mask = torch.zeros(self.num_wr_prompt, device=torch.device(args.device))

            else: # set to false
                prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                                        
        # if using learnable prompt keys
        if prompt_key: # set to true
            if self.specific_prompts:
                key_shape = (self.s_prompts_num_prompts, embed_dim)
            else:
                key_shape = (num_feat_prompt, embed_dim) if self.feat_prompt else (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform': # set to true
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)

            if self.coda_prompt:
                coda_prompt_key_shape = (pool_size, self.num_coda_prompt, embed_dim)
                if prompt_key_init == 'zero':
                    self.coda_prompt_key = nn.Parameter(torch.zeros(coda_prompt_key_shape))
                elif prompt_key_init == 'uniform': # set to true
                    self.coda_prompt_key = nn.Parameter(torch.randn(coda_prompt_key_shape))
                    nn.init.uniform_(self.coda_prompt_key, -1, 1)

            # Use waiting room prompts 
            if self.with_wr_prompt:
                wr_key_shape = (self.num_wr_prompt, embed_dim)
                if prompt_init == 'zero':
                    self.wr_key_prompt = nn.Parameter(torch.zeros(wr_key_shape))
                elif prompt_init == 'uniform': # set to true
                    self.wr_key_prompt = nn.Parameter(torch.randn(wr_key_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                    nn.init.uniform_(self.wr_key_prompt, -1, 1)
                self.wr_key_prompt.requires_grad = False

        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 

        # Use query prompts for feat-prompts
        if self.coda_prompt or self.feat_prompt:
            if self.specific_prompts:
                attn_trsf_shape = (self.s_prompts_num_prompts, embed_dim)
            else:
                attn_trsf_shape = (pool_size, self.num_coda_prompt, embed_dim) if self.coda_prompt else \
                            (self.num_feat_prompt, embed_dim)
            if prompt_key_init == 'zero':
                self.attn_trsf = nn.Parameter(torch.zeros(attn_trsf_shape))
            elif prompt_key_init == 'uniform': # set to true
                self.attn_trsf = nn.Parameter(torch.randn(attn_trsf_shape))
                nn.init.uniform_(self.attn_trsf, -1, 1)
            
            # Use waiting room prompts 
            if self.with_wr_prompt:
                wr_query_shape = (self.num_wr_prompt, embed_dim)
                if prompt_init == 'zero':
                    self.wr_query_prompt = nn.Parameter(torch.zeros(wr_query_shape))
                elif prompt_init == 'uniform': # set to true
                    self.wr_query_prompt = nn.Parameter(torch.randn(wr_query_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                    nn.init.uniform_(self.wr_query_prompt, -1, 1)
                self.wr_query_prompt.requires_grad = False
                
        self.supcls_prompts = args.supcls_prompts
        self.shared_prompt = args.shared_prompt
        self.sep_shared_prompt = args.sep_shared_prompt
        self.only_shared_prompt = args.only_shared_prompt
        # if original_model:
        #     self.one_shared_prompt = True
        # else:
        #     self.one_shared_prompt = args.one_shared_prompt
        self.one_shared_prompt = args.one_shared_prompt
        
        self.shared_prompt_ema_ratio = args.shared_prompt_ema_ratio
        self.shared_prompt_ema = None
        
        self.normalize_prompt = args.normalize_prompt
        
        # args.data_driven_evolve or uni
        self.converged_p = torch.tensor([]) 
        self.uni_or_specific = args.uni_or_specific
            
        # film_layer
        if args.film_train:
            self.film = nn.Linear(embed_dim, 2*embed_dim) # gamma and beta
        self.use_film = False
        self.mergable_cands = None
        self.film_combine = args.film_combine
        self.nofilm = args.nofilm

        # mergable_prompts
        if args.mergable_prompts:
            self.mergable_prompts = True
            self.mergable_task = False # change by task
            self.mergable_prompts_start_idx = args.num_tasks
            self.training_mergable_pids = None
            self.reclustered_mergable_pids = []
        else:
            self.mergable_prompts = False
        
    def get_prompt_range(self, target=None, supcls=None):
        # target: (bs,)
        if target is None: # supcls is not None
            supcls = list(supcls.cpu().numpy())
            supcls = supcls # (bs,)
        else: # target is not None
            supcls = [self.cls2supcls[trg.item()] for trg in target.cpu()] # (bs,)
            
        prmptinds = torch.tensor([self.supcls2prmptinds[scls] for scls in supcls]) # (bs, num_prompt_per_supcls)
        return torch.tensor(supcls), prmptinds # (bs,), (bs, prompt-range: 5 or 10) 
            
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm       
    
    def orth_loss(self, matrix):
        # matrix: (M, D)
        loss = torch.norm(torch.mm(matrix, matrix.T) - torch.eye(matrix.size(0)).to(matrix.device)) # (M,M)
        return loss

    def orthogonal_loss(self, prompts=None, keys=None, attn_trsf=None,):
        p_l = self.orth_loss(prompts) if prompts is not None else 0 # (100, 100)
        k_l = self.orth_loss(keys) if keys is not None else 0 # (100, 100)
        a_l = self.orth_loss(attn_trsf) if attn_trsf is not None else 0 # (100, 100)
        orth_losses = p_l + k_l + a_l
        return orth_losses

    def forward(self, x_embed, target=None, prompt_mask=None, aug_cls_features=None, 
                cls_features=None, task_id=None, ):
        out = dict()
        # print("###idx=0",self.prompt.data[:,:,0][0][0])
        # print("###idx=1",self.prompt.data[:,:,1][0][0])
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls': # set to true; CLS token
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            elif self.embedding_key == 'cls_max': # set to true; CLS token
                # if cls_features is None:
                #     x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                # else:
                x_embed_mean = 0.7*cls_features + 0.3*torch.max(x_embed, dim=1)[0]
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) 
            # (num_feats_prompt, C) if feat_prompt; (Pool_size, C) otherwise
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) 
            # (pool_size, bs) or (num_feats_prompt, bs)
            similarity = similarity.t() 
            # (bs, pool_size) or (bs, num_feats_prompt): similarity between batch and pool_keys
            
            if self.task_free and self.keep_prompt_freq: # l2p + task_free
                if self.copy_top2bottom and (self.freq_dict_prev is not None):
                    freq_mat = 1/self.freq_dict_prev # mat using freq_dict_prev
                    similarity = similarity * freq_mat
                else: 
                    freq_mat = 1/self.freq_dict_curr # mat using freq_dict_prev
                    similarity = similarity * freq_mat

            elif self.keep_prompt_freq and task_id > 0: # l2p + non task-free
                if task_id > self.task_id: # new task
                    self.freq_dict_prev = deepcopy(self.freq_dict_curr) # (pool_size)
                    self.task_id = task_id
               
                freq_mat = 1/self.freq_dict_prev # mat using freq_dict_prev
                similarity = similarity * freq_mat
            
            # data_driven_evolve: if bi-d & eval_known_prompts, ignore below
            if len(self.converged_p) > 0: # => self.converged_p is not None :
                if self.uni_or_specific and self.mergable_prompts: # 
                    # converged_p = uni-or-spcf + reclustered
                    # converged_p = torch.cat( (self.converged_p[self.converged_p!=-1], 
                    #                           torch.tensor(self.reclustered_mergable_pids)) )
                    # print('pids4inf', self.pids4inf)
                    not_converged_p = torch.from_numpy(np.setdiff1d(np.arange(similarity.size(1)),
                                                                    self.pids4inf.numpy()))
                    similarity[:,not_converged_p] = -1 # (bs, pool_size)

                elif self.uni_or_specific: # at eval, select prompts only from converged prompts 
                    not_converged_p = torch.from_numpy(np.setdiff1d(np.arange(similarity.size(1)),
                                                                    self.converged_p.numpy()))
                    similarity[:,not_converged_p] = -1 # (bs, pool_size)
                    # print('similarity', similarity[0])
                
                else: # data_driven_evolve or initprompts
                    similarity[:,self.converged_p] = -1 # (bs, pool_size)
            
            
            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # bs, top_k
            # print('idx: ', idx.size())
            out['similarity'] = similarity
                
            if self.batchwise_prompt: # set to true; only used for test
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # print('major_prompt_id: ', major_prompt_id.size())
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
                # print('idx: ', idx.size())*q
            
            if prompt_mask is not None and (not self.keep_prompt_freq): # only used for train; set to true
                idx = prompt_mask # (B, top_k): essentially, task_ids
                if (self.uni_or_specific) and ( self.converged_p.numel() == task_id+1 ): 
                    # after warmup(deciding between uni. and specific), use converged_prompt
                    if self.mergable_prompts: 
                        if self.mergable_task: # mergable 
                            idx = np.random.choice(self.training_mergable_pids, size=idx.size(0), replace=True, )
                            idx = torch.from_numpy(idx).unsqueeze(-1) # (B,top_k=1)
                            print('mixed ids of mergable prompts', idx[0])
                        else: # uni or spcf
                            idx = self.converged_p[prompt_mask].long() # task_ids -> prompt_ids: (B,top_k=1)
                    else:
                        idx = self.converged_p[prompt_mask].long() # task_ids -> prompt_ids: (B,top_k=1)
                # print('idx', idx[0])
            
            if (self.keep_prompt_freq) and (not self.feat_prompt): # update self.freq_dict_curr
                flatten_id, cnts  = torch.unique(idx.reshape(-1), return_counts=True)
                self.freq_dict_curr[flatten_id] += cnts
            
                
            out['prompt_idx'] = idx
            
            if self.use_prefix_tune_for_e_prompt: # set to true
                # self.prompt: (self.num_layers, 2, feat_prompt_nums, self.length, 
                                        # self.num_heads, embed_dim // self.num_heads)

                # (num_layers, 2, bs, top_k, length, C) or (num_layers, 2, bs, feature_num, top_k, length, C)
                if self.feat_prompt:
                    bs = x_embed_norm.size(0)
                    batched_attn_trsf = self.attn_trsf.unsqueeze(0).expand(bs,-1,-1) # (bs, feat_prompt, D)
                    batched_key = self.prompt_key.unsqueeze(0).expand(bs,-1,-1) # (bs, feat_prompt, D)
                    if self.with_wr_prompt:
                        batched_wr_query = self.wr_query_prompt.unsqueeze(0).expand(bs,-1,-1) # (bs, wr, D)
                        batched_wr_key = self.wr_key_prompt.unsqueeze(0).expand(bs,-1,-1) # (bs, wr, D)
                        batched_attn_trsf = torch.cat((batched_attn_trsf,batched_wr_query), dim=1)
                        batched_key = torch.cat((batched_key,batched_wr_key), dim=1)
                        # (bs, feat_prompt+wr, D)
                        
                    qA = x_embed_norm.unsqueeze(1) * batched_attn_trsf # (bs, feat_prompt(or +wr), D)
                    # qA = x_embed_norm.unsqueeze(1).expand(-1, batched_attn_trsf.size(1),-1) + \
                    #     0 * x_embed_norm.unsqueeze(1) * batched_attn_trsf # (bs, feat_prompt(or +wr), D)# (bs, feat_prompt(or +wr), D)
                    # qAK < cos(qA, key): (bs, 1, num_coda_prompt)
                    qAK = self.l2_normalize(qA,dim=-1) * self.l2_normalize(batched_key,dim=-1) # (bs,feat_num,D)
                    qAK = qAK.sum(dim=-1).unsqueeze(1) # (bs,1,feat_prompt_nums(or +wr))
                    if self.softmax_feat:
                        qAK = self.softmax(qAK)
                    if self.save_attn_scores:
                        out['attn_scores'] = qAK.squeeze(1).cpu() # (bs, feat_prompt_nums)
                    
                    if (self.copy_top2bottom) and (self.feat_attn_mask) and (self.attn_mask is not None): # qAK: (bs,1,feat_prompt_nums)
                        qAK[:,:,:self.num_feat_prompt]= qAK[:,:,:self.num_feat_prompt] * self.attn_mask.unsqueeze(0).unsqueeze(0)

                    if self.with_wr_prompt: # exclude inactive wr-prompts
                        qAK[:,:,-self.num_wr_prompt:] = qAK[:,:,-self.num_wr_prompt:]*self.wr_prompt_mask.unsqueeze(0).unsqueeze(0)

                    if self.keep_prompt_freq and self.feat_prompt: # update self.freq_dict_curr
                        freq = torch.abs(qAK.detach().squeeze(1)) # (bs, feat_prompt_nums+wr) like similarity
                        if self.with_wr_prompt: # for attn-hist, exclude attn of all wr-prompts
                            freq = freq[:,:self.num_feat_prompt]
                        _, top_idx = torch.topk(freq, k=self.top_k, dim=1) # bs, top_k
                        flatten_id, cnts  = torch.unique(top_idx.reshape(-1), return_counts=True)
                        self.freq_dict_curr[flatten_id] += cnts
                        
                    if  torch.sum(qAK>1) > 0 or  torch.sum(qAK<-1) > 0:
                        print("attn score=qAK not in (-1,1)")
                                            
                    num_layers, dual, feat_num, length, num_heads, heads_embed_dim = self.prompt.shape
                    # print('qAK', qAK.size())
                    batched_prompt_raw = self.prompt.transpose(0,2).contiguous().reshape(feat_num,-1)
                    # if self.normalize_prompt:
                    #     batched_prompt_raw = self.l2_normalize(batched_prompt_raw, dim=-1)
                    # (feat_prompt_nums, -1: 2, num_layers, length, num_heads, embed_dim // num_heads)
                    batched_prompt_raw = batched_prompt_raw.unsqueeze(0).expand(bs,-1,-1)
                    # print('batched_prompt_raw', batched_prompt_raw.size())
                    # (bs, feat_prompt_nums, -1: 2, num_layers, length, num_heads, embed_dim // num_heads)
                    if self.with_wr_prompt:
                        batched_wr_prompt_raw = self.wr_prompt.transpose(0,2).contiguous().reshape(self.num_wr_prompt,-1)
                        batched_wr_prompt_raw = batched_wr_prompt_raw.unsqueeze(0).expand(bs,-1,-1)
                        batched_prompt_raw = torch.cat((batched_prompt_raw, batched_wr_prompt_raw), dim=1)
                        # (bs, feat_prompt+wr_prompts, -1)
                    batched_prompt_raw = torch.bmm(qAK, batched_prompt_raw).squeeze(1)  # (bs,1,D) -> (bs,D)
                    # batched_prompt_raw = batched_prompt_raw.squeeze(1)+0*torch.bmm(qAK, batched_prompt_raw).squeeze(1)  # (bs,1,D) -> (bs,D)
                    if self.with_wr_prompt: # normalize by #prompts
                        # self.num_feat_prompt, self.num_wr_prompt
                        batched_prompt_raw = batched_prompt_raw*(feat_num/(feat_num+self.wr_prompt_mask.sum()))

                    #### get aug_feat_prompt #################################################################
                    if (self.cl_prompts) and (target is not None):
                        aug_x_embed_norm = self.l2_normalize(aug_cls_features, dim=-1)
                        aug_qA = aug_x_embed_norm.unsqueeze(1) * batched_attn_trsf # (bs, feat_prompt(or +wr), D)
                        aug_qAK = self.l2_normalize(aug_qA,dim=-1) * self.l2_normalize(batched_key,dim=-1) # (bs,feat_num,D)
                        aug_qAK = aug_qAK.sum(dim=-1).unsqueeze(1) # (bs,1,feat_prompt_nums(or +wr))
                        aug_batched_prompt_raw = self.prompt.transpose(0,2).contiguous().reshape(feat_num,-1)
                        aug_batched_prompt_raw = aug_batched_prompt_raw.unsqueeze(0).expand(bs,-1,-1)
                        aug_batched_prompt_raw = torch.bmm(aug_qAK, aug_batched_prompt_raw).squeeze(1)  # (bs,D)
                    #### ################### #################################################################
                    
                    out['con_loss'] = 0. 
                    out['supcon_loss'] = 0. 
                    out['supcon_attns_loss'] = 0. 
                    
                    batched_prompt_raw = batched_prompt_raw.reshape(bs, dual, num_layers, length, num_heads, heads_embed_dim)
                    batched_prompt_raw = batched_prompt_raw.transpose(0,2).contiguous() # (num_layers,dual,bs,length,num_heads,heads_embed_dim,)
                    
                    batched_prompt = batched_prompt_raw.reshape(
                    num_layers, bs, dual, self.top_k * length, num_heads, heads_embed_dim
                    )
                    
                        
                    if self.shared_prompt:
                        if self.sep_shared_prompt:
                            shared_prompt = self.prompt[:,:,self.num_feat_prompt//2:].mean(dim=2, keepdim=True).expand(-1,-1,bs,-1,-1,-1)
                        elif self.one_shared_prompt:
                            if self.specific_prompts:
                                shared_prompt = self.one_prompt[:,:,[0]]
                                shared_prompt = shared_prompt.expand(-1,-1,bs,-1,-1,-1)
                            else:
                                shared_prompt = self.prompt[:,:,[0]]
                                shared_prompt = shared_prompt.expand(-1,-1,bs,-1,-1,-1)
                        else:
                            shared_prompt = self.prompt.mean(dim=2, keepdim=True).expand(-1,-1,bs,-1,-1,-1)
                        shared_prompt = shared_prompt.reshape(
                        self.num_layers, bs, dual, self.length, num_heads, heads_embed_dim
                        ) # (num_layers, bs, dual, 1*length, num_heads, heads_embed_dim)
                        
                        # batched_prompt = (batched_prompt + shared_prompt)/2 # torch.cat((batched_prompt,shared_prompt), dim=3)
                        # # (num_layers, dual, bs, 1*length, num_heads, heads_embed_dim)
                        if self.shared_prompt_ema is None:
                            shared_prompt_ema = shared_prompt.mean(1, keepdim=True)# .detach()
                            self.shared_prompt_ema = shared_prompt.mean(1, keepdim=True).detach()
                        else:
                            shared_prompt_ema = self.shared_prompt_ema_ratio*self.shared_prompt_ema +\
                                (1-self.shared_prompt_ema_ratio)*shared_prompt.mean(1, keepdim=True)# .detach()
                            self.shared_prompt_ema = self.shared_prompt_ema_ratio*self.shared_prompt_ema +\
                                (1-self.shared_prompt_ema_ratio)*shared_prompt.mean(1, keepdim=True).detach()
                        
                        if self.only_shared_prompt:
                            batched_prompt = shared_prompt_ema.expand(-1,bs,-1,-1,-1,-1) + batched_prompt*0
                        elif self.specific_prompts:
                            batched_prompt_list = []
                            # print(self.num_layers)
                            for layer_i in range(self.num_layers):
                                if layer_i in self.s_prompts_layer_idx:
                                    s_layer_i = self.s_prompts_layer_idx.index(layer_i)
                                    if self.s_prompts_add=='add':
                                        layer_p = 0.9*shared_prompt_ema.expand(-1,bs,-1,-1,-1,-1)[layer_i]+\
                                            0.1*batched_prompt[s_layer_i]    
                                            
                                    elif self.s_prompts_add=='cat':
                                        layer_p = torch.cat((batched_prompt[s_layer_i],
                                            shared_prompt_ema.expand(-1,bs,-1,-1,-1,-1)[layer_i]), dim=2)
                                        
                                    batched_prompt_list += [layer_p]
                                else:
                                    batched_prompt_list += [shared_prompt_ema.expand(-1,bs,-1,-1,-1,-1)[layer_i]]
                                # print(batched_prompt_list[-1].size())
                            batched_prompt = batched_prompt_list
                            
                        else:
                            batched_prompt = torch.cat((batched_prompt,
                                                    shared_prompt_ema.expand(-1,bs,-1,-1,-1,-1)), dim=3)
                        
                        # (num_layers, dual, bs, 2*length, num_heads, heads_embed_dim)
                        
                        # batched_prompt = torch.cat((batched_prompt,shared_prompt), dim=3)
                        # (num_layers, dual, bs, 2*length, num_heads, heads_embed_dim)
                        
                        # print('supcls_shared_prompt, batched_prompt', batched_prompt.size())

                        
                    if self.orthogonal_coeff > 0.:
                        # get orthogonal loss: prompt, key, attn_trsf
                        prompts = self.prompt 
                        # (num_layers, 2, num_feat_prompt, length, num_heads, embed_dim // self.num_heads)
                        prompts = prompts.transpose(0,2).contiguous()
                        prompts = self.l2_normalize(prompts.reshape(self.num_feat_prompt, -1), dim=-1) # (100, D)
                        keys = self.l2_normalize(self.prompt_key, dim=-1) # (100, D)
                        attn_trsf = self.l2_normalize(self.attn_trsf, dim=-1) # (100, D)
                    
                        orth_losses = self.orthogonal_loss(prompts=prompts, keys=keys, attn_trsf=attn_trsf,)
                        out['orthogonal_losses'] = orth_losses
                    else: 
                        out['orthogonal_losses'] = 0.
                        
                else: # for l2p, dualprompt
                    batched_prompt_raw = self.prompt[:,:,idx]  
                    num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    )
                    
                    if self.shared_prompt:
                        shared_prompt = self.prompt.mean(dim=2, keepdim=True).expand(-1,-1,batch_size,-1,-1,-1)
                        shared_prompt = shared_prompt.reshape(
                        num_layers, batch_size, dual, length, num_heads, heads_embed_dim
                        ) # (num_layers, bs, dual, length, num_heads, heads_embed_dim)
                        batched_prompt = torch.cat((batched_prompt,shared_prompt), dim=3)
                        # (num_layers, dual, bs, (top_k+1)*length, num_heads, heads_embed_dim)
                        
                    batched_key_norm = prompt_key_norm[idx] # B, top_k=1, C

                    out['selected_key'] = batched_key_norm
                    out['prompt_key_norm'] = prompt_key_norm
                    out['x_embed_norm'] = x_embed_norm
            else:
                batched_prompt_raw = self.prompt[:,idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

                batched_key_norm = prompt_key_norm[idx] # B, top_k=1, C

                out['selected_key'] = batched_key_norm
                out['prompt_key_norm'] = prompt_key_norm
                out['x_embed_norm'] = x_embed_norm

            if not (self.feat_prompt or self.coda_prompt): # for l2p, dualprompt
                # Put pull_constraint loss calculation inside
                x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
                sim = batched_key_norm * x_embed_norm # B, top_k, C
                reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
                
                out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt
        return out
