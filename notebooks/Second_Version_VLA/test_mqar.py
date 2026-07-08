import math, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB = 128

def make_mqar_old(B, n_pairs, vocab=VOCAB, device=DEVICE):
    sep       = vocab-1
    key_range = max(vocab-1, n_pairs+1)
    T         = 2*n_pairs+1+n_pairs
    x=torch.full((B,T),sep,dtype=torch.long,device=device)
    y=torch.full((B,T),-100,dtype=torch.long,device=device)
    for b in range(B):
        raw=torch.randperm(key_range,device=device)[:n_pairs]
        keys=raw%(vocab-1)
        vals=torch.randint(0,vocab-1,(n_pairs,),device=device)
        for i in range(n_pairs):
            x[b,2*i]=keys[i]; x[b,2*i+1]=vals[i]
        x[b,2*n_pairs]=sep
        perm=torch.randperm(n_pairs,device=device)
        x[b,2*n_pairs+1:]=keys[perm]
        y[b,2*n_pairs+1:]=vals[perm]
    return x,y

def make_mqar_new(B, n_pairs, vocab=VOCAB, device=DEVICE):
    sep = vocab - 1
    key_range = max(vocab - 1, n_pairs + 1)
    T = 2 * n_pairs + 1 + n_pairs
    x = torch.full((B, T), sep, dtype=torch.long, device=device)
    y = torch.full((B, T), -100, dtype=torch.long, device=device)
    
    r = torch.rand((B, key_range), device=device)
    raw = torch.argsort(r, dim=1)[:, :n_pairs]
    keys = raw % (vocab - 1)
    vals = torch.randint(0, vocab - 1, (B, n_pairs), device=device)
    
    x[:, 0:2*n_pairs:2] = keys
    x[:, 1:2*n_pairs:2] = vals
    x[:, 2*n_pairs] = sep
    
    r2 = torch.rand((B, n_pairs), device=device)
    perm = torch.argsort(r2, dim=1)
    b_idx = torch.arange(B, device=device).unsqueeze(1)
    
    q_keys = keys[b_idx, perm]
    q_vals = vals[b_idx, perm]
    
    x[:, 2*n_pairs+1:] = q_keys
    y[:, 2*n_pairs+1:] = q_vals
    return x, y

print(f"Device: {DEVICE}")

# warmup
_, _ = make_mqar_old(2, 2)
_, _ = make_mqar_new(2, 2)

import time
torch.cuda.synchronize() if DEVICE == 'cuda' else None

t0 = time.time()
for _ in range(10):
    x, y = make_mqar_old(32, 32)
torch.cuda.synchronize() if DEVICE == 'cuda' else None
print("Old 10 iters:", time.time() - t0)

t0 = time.time()
for _ in range(10):
    x, y = make_mqar_new(32, 32)
torch.cuda.synchronize() if DEVICE == 'cuda' else None
print("New 10 iters:", time.time() - t0)
