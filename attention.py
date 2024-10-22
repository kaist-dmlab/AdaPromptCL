# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lae=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_lae = use_lae
        if self.use_lae:
            self.register_parameter("scale_k", nn.Parameter(torch.tensor(1.0)))
            self.register_parameter("scale_v", nn.Parameter(torch.tensor(1.0)))

    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q or k or v: (B, H, N, C//H)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            if self.use_lae:
                # print('lae scaling')
                key_prefix = key_prefix * self.scale_k
                value_prefix = value_prefix * self.scale_v
            k = torch.cat([key_prefix, k], dim=2) # (B,H,prompt_len+N,C//H)
            v = torch.cat([value_prefix, v], dim=2) # (B,H,prompt_len+N,C//H)

        # print('q..size(), k..size() v..size()', q.size(), k.size(), v.size(),
            #   q.device, k.device, v.device)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # (B,H,N,prompt_len+N)
        
        # LAE baseline: compensate 
        if self.use_lae:
            # print('lae compensating prefix')
            attn = self.compensate_LAE(attn)
        
        # print('att.device', attn.device)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # (B,H,N,C//H) -> (B,N,H,C//H) -> (B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def compensate_LAE(self,attn):
        # attn: (B,H,N,prompt_len+N)
        
        s = attn.size(-1)-attn.size(-2) # prompt_len
        t = s+1
        # print('s,t: ', s,t)
        
        # [:s] <- prompt; [s:t] <- CLS; [t:] <- img_emb
        lamb = attn[..., :s].sum(dim=-1, keepdim=True)
        # print('lamb: ', lamb[0,0,:10])
        attn1 = attn[..., :s] / lamb.clamp(min=1e-12) # prompt compensation
        # print('attn1:', attn1[0,0,0,:10])
        attn2 = attn[..., s:t]  
        attn3 = attn[..., t:]
        attn = torch.cat([attn1, attn2, attn3], dim=-1)

        return attn