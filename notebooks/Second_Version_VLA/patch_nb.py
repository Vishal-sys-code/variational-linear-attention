import json
from pathlib import Path

fp = Path('12e_Hetero_vs_Uniform.ipynb')
nb = json.load(open(fp, 'r', encoding='utf-8'))

task_code = """def make_mqar(B, n_pairs, vocab=VOCAB, device=DEVICE):
    \"\"\"Standard MQAR: [k1 v1 ... kn vn SEP q1 ... qn].  T = 3n+1.
    This is the ONLY format that has ever converged in this project.
    \"\"\"
    sep       = vocab-1
    key_range = max(vocab-1, n_pairs+1)
    T         = 2*n_pairs+1+n_pairs
    x=torch.full((B,T),sep,dtype=torch.long,device=device)
    y=torch.full((B,T),-100,dtype=torch.long,device=device)
    
    r = torch.rand((B, key_range), device=device)
    raw = torch.argsort(r, dim=1)[:, :n_pairs]
    keys = raw % (vocab-1)
    vals = torch.randint(0, vocab-1, (B, n_pairs), device=device)
    
    x[:, 0:2*n_pairs:2] = keys
    x[:, 1:2*n_pairs:2] = vals
    
    r2 = torch.rand((B, n_pairs), device=device)
    perm = torch.argsort(r2, dim=1)
    b_idx = torch.arange(B, device=device).unsqueeze(1)
    
    x[:, 2*n_pairs+1:] = keys[b_idx, perm]
    y[:, 2*n_pairs+1:] = vals[b_idx, perm]
    return x,y

print('MQAR format checks:')
for n in [8,32,128,200]:
    x,y=make_mqar(4,n)
    T_exp=3*n+1
    assert x.shape==(4,T_exp)
    assert (y!=-100).sum().item()==4*n
    assert x.max().item()<=VOCAB-1
    print(f'  n={n:4d}: T={T_exp:4d}  targets={4*n}  OK')
print(f'Random baseline = {1/(VOCAB-1):.4f}')
"""

models_code = """# ── VLAv3 single-head with CHUNKED GRADIENT CHECKPOINTING ────────────
# Key change from v1: the T-step recurrence is split into CHUNK-sized
# segments.  Each segment is wrapped in torch.utils.checkpoint so that
# only one chunk's intermediates are live at a time during backward.
# This reduces activation memory from O(T * dh^2) to O(CHUNK * dh^2).

class VLAv3Head(nn.Module):
    def __init__(self, dh, lam=0.1, eps=1e-4, per_eps=1e-3, period=20, chunk=CHUNK):
        super().__init__()
        self.dh=dh; self.lam=lam; self.eps=eps
        self.per_eps=per_eps; self.period=period; self.chunk=chunk
        # Each a standard nn.Linear(dh,dh) — PyTorch uses kaiming_uniform
        # with fan_in=dh, which is CORRECT. No batched-param bug here.
        self.Wq=nn.Linear(dh,dh); self.Wk=nn.Linear(dh,dh)
        self.Wv=nn.Linear(dh,dh); self.Wu=nn.Linear(dh,dh,bias=False)
        self.Wo=nn.Linear(dh,dh); self.norm=nn.LayerNorm(dh)

    def _recurrence_chunk(self, kf_c, Q_c, V_c, U_c, A, S, zk, I, c_start):
        \"\"\"Process one chunk of the Sherman–Morrison recurrence in float32.
        Called via ckpt.checkpoint — intermediates are freed after each chunk.\"\"\"
        # Disable autocast so bmm/einsum stay in float32 for numerical safety
        with torch.amp.autocast(device_type=kf_c.device.type, enabled=False):
            B, C_len, d = kf_c.shape
            isq = 1/math.sqrt(d)
            ys = []
            for t in range(C_len):
                gt = c_start + t
                u = U_c[:,t,:] * isq
                zsm = torch.bmm(A, u.unsqueeze(-1)).squeeze(-1)
                dlt = (1 + (u*zsm).sum(-1)).clamp(min=self.eps)
                A = A - torch.bmm(zsm.unsqueeze(-1), zsm.unsqueeze(1)) / dlt.view(B,1,1)
                if (gt+1) % self.period == 0:
                    A = A + self.per_eps * I.unsqueeze(0)
                kn = F.normalize(kf_c[:,t,:], p=2, dim=-1)
                alpha = torch.bmm(A, kn.unsqueeze(-1)).squeeze(-1)
                alphn = F.normalize(alpha, p=2, dim=-1)
                e = V_c[:,t,:] - torch.bmm(S, kn.unsqueeze(-1)).squeeze(-1)
                S = S + torch.bmm(e.unsqueeze(-1), alphn.unsqueeze(1))
                qt = Q_c[:,t,:]
                zk = zk + kf_c[:,t,:]
                yt = torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1)
                ys.append(yt / (zk*qt).sum(-1, keepdim=True).clamp(min=self.eps))
            return torch.stack(ys, 1), A, S, zk

    def forward(self, x):
        B, T, d = x.shape
        # ── Projections (benefit from AMP float16 if active) ──
        kr = self.Wk(x); kf = F.elu(kr) + 1.0
        Q  = F.elu(self.Wq(x)) + 1.0; V = self.Wv(x)
        U  = F.normalize(self.Wu(kr), p=2, dim=-1)
        # ── Cast to float32 for numerically stable recurrence ──
        kf, Q, V, U = kf.float(), Q.float(), V.float(), U.float()
        I  = torch.eye(d, device=x.device, dtype=torch.float32)
        A  = (1/self.lam) * I.unsqueeze(0).expand(B,-1,-1).clone()
        S  = torch.zeros(B, d, d, device=x.device, dtype=torch.float32)
        zk = torch.zeros(B, d, device=x.device, dtype=torch.float32)
        # ── Chunked recurrence with gradient checkpointing ──
        all_ys = []
        for c0 in range(0, T, self.chunk):
            c1 = min(c0 + self.chunk, T)
            chunk_ys, A, S, zk = ckpt.checkpoint(
                self._recurrence_chunk,
                kf[:,c0:c1], Q[:,c0:c1], V[:,c0:c1], U[:,c0:c1],
                A, S, zk, I, c0,
                use_reentrant=False
            )
            all_ys.append(chunk_ys)
        return self.Wo(self.norm(torch.cat(all_ys, 1)))


# ── Multi-head wrappers ────────────────────────────────────────────────────
class UniformVLA(nn.Module):
    \"\"\"Standard uniform VLA: H heads, all dh = d_model // H.\"\"\"
    def __init__(self, d_model, H=UNIF_H):
        super().__init__()
        self.H=H; self.dh=d_model//H
        self.heads=nn.ModuleList([VLAv3Head(self.dh) for _ in range(H)])
        self.Wo=nn.Linear(d_model,d_model); self.norm=nn.LayerNorm(d_model)
    def capacity(self): return self.H*self.dh
    def forward(self,x):
        B,T,D=x.shape; dh=self.dh
        
        # Batch over heads
        kfs, Qs, Vs, Us = [], [], [], []
        for i,h in enumerate(self.heads):
            xi = x[:,:,i*dh:(i+1)*dh]
            kr = h.Wk(xi); kf = F.elu(kr) + 1.0
            Q = F.elu(h.Wq(xi)) + 1.0; V = h.Wv(xi)
            U = F.normalize(h.Wu(kr), p=2, dim=-1)
            kfs.append(kf); Qs.append(Q); Vs.append(V); Us.append(U)
            
        kf = torch.cat(kfs, dim=0).float()
        Q = torch.cat(Qs, dim=0).float()
        V = torch.cat(Vs, dim=0).float()
        U = torch.cat(Us, dim=0).float()
        
        I = torch.eye(dh, device=x.device, dtype=torch.float32)
        A = (1/self.heads[0].lam) * I.unsqueeze(0).expand(B*self.H,-1,-1).clone()
        S = torch.zeros(B*self.H, dh, dh, device=x.device, dtype=torch.float32)
        zk = torch.zeros(B*self.H, dh, device=x.device, dtype=torch.float32)
        
        all_ys = []
        for c0 in range(0, T, CHUNK):
            c1 = min(c0 + CHUNK, T)
            chunk_ys, A, S, zk = ckpt.checkpoint(
                self.heads[0]._recurrence_chunk,
                kf[:,c0:c1], Q[:,c0:c1], V[:,c0:c1], U[:,c0:c1],
                A, S, zk, I, c0,
                use_reentrant=False
            )
            all_ys.append(chunk_ys)
            
        ys = torch.cat(all_ys, 1)
        
        outs = []
        for i,h in enumerate(self.heads):
            yi = ys[i*B:(i+1)*B]
            outs.append(h.Wo(h.norm(yi)))
            
        return self.Wo(self.norm(torch.cat(outs,-1)))


class HeteroVLA(nn.Module):
    \"\"\"Heterogeneous VLA: fast heads (small dh) + slow heads (large dh).
    Capacity = fast_H*fast_dh + slow_H*slow_dh.
    Prop 2 holds per head independently (each head is a VLAv3Head).
    Corollary: total capacity = sum of per-head capacities.
    \"\"\"
    def __init__(self,
                 fast_H=FAST_H, fast_dh=FAST_DH,
                 slow_H=SLOW_H, slow_dh=SLOW_DH):
        super().__init__()
        self.fast_H=fast_H; self.fast_dh=fast_dh
        self.slow_H=slow_H; self.slow_dh=slow_dh
        self._fast_dim = fast_H*fast_dh
        self._d = fast_H*fast_dh + slow_H*slow_dh
        self.fast_heads=nn.ModuleList([VLAv3Head(fast_dh) for _ in range(fast_H)])
        self.slow_heads=nn.ModuleList([VLAv3Head(slow_dh) for _ in range(slow_H)])
        self.Wo=nn.Linear(self._d,self._d); self.norm=nn.LayerNorm(self._d)
    def capacity(self): return self.fast_H*self.fast_dh + self.slow_H*self.slow_dh
    def forward(self,x):
        B, T, D = x.shape
        fd=self._fast_dim; outs=[]
        xf=x[:,:,:fd]; xs=x[:,:,fd:]
        
        # Fast heads batched
        if self.fast_H > 0:
            kfs, Qs, Vs, Us = [], [], [], []
            for i,h in enumerate(self.fast_heads):
                xi = xf[:,:,i*self.fast_dh:(i+1)*self.fast_dh]
                kr = h.Wk(xi); kfs.append(F.elu(kr) + 1.0)
                Qs.append(F.elu(h.Wq(xi)) + 1.0); Vs.append(h.Wv(xi))
                Us.append(F.normalize(h.Wu(kr), p=2, dim=-1))
            
            kf = torch.cat(kfs, dim=0).float()
            Q = torch.cat(Qs, dim=0).float()
            V = torch.cat(Vs, dim=0).float()
            U = torch.cat(Us, dim=0).float()
            
            I = torch.eye(self.fast_dh, device=x.device, dtype=torch.float32)
            A = (1/self.fast_heads[0].lam) * I.unsqueeze(0).expand(B*self.fast_H,-1,-1).clone()
            S = torch.zeros(B*self.fast_H, self.fast_dh, self.fast_dh, device=x.device, dtype=torch.float32)
            zk = torch.zeros(B*self.fast_H, self.fast_dh, device=x.device, dtype=torch.float32)
            
            all_ys = []
            for c0 in range(0, T, CHUNK):
                c1 = min(c0 + CHUNK, T)
                chunk_ys, A, S, zk = ckpt.checkpoint(
                    self.fast_heads[0]._recurrence_chunk,
                    kf[:,c0:c1], Q[:,c0:c1], V[:,c0:c1], U[:,c0:c1],
                    A, S, zk, I, c0,
                    use_reentrant=False
                )
                all_ys.append(chunk_ys)
                
            ys = torch.cat(all_ys, 1)
            for i,h in enumerate(self.fast_heads):
                yi = ys[i*B:(i+1)*B]
                outs.append(h.Wo(h.norm(yi)))
                
        # Slow heads batched
        if self.slow_H > 0:
            kfs, Qs, Vs, Us = [], [], [], []
            for i,h in enumerate(self.slow_heads):
                xi = xs[:,:,i*self.slow_dh:(i+1)*self.slow_dh]
                kr = h.Wk(xi); kfs.append(F.elu(kr) + 1.0)
                Qs.append(F.elu(h.Wq(xi)) + 1.0); Vs.append(h.Wv(xi))
                Us.append(F.normalize(h.Wu(kr), p=2, dim=-1))
            
            kf = torch.cat(kfs, dim=0).float()
            Q = torch.cat(Qs, dim=0).float()
            V = torch.cat(Vs, dim=0).float()
            U = torch.cat(Us, dim=0).float()
            
            I = torch.eye(self.slow_dh, device=x.device, dtype=torch.float32)
            A = (1/self.slow_heads[0].lam) * I.unsqueeze(0).expand(B*self.slow_H,-1,-1).clone()
            S = torch.zeros(B*self.slow_H, self.slow_dh, self.slow_dh, device=x.device, dtype=torch.float32)
            zk = torch.zeros(B*self.slow_H, self.slow_dh, device=x.device, dtype=torch.float32)
            
            all_ys = []
            for c0 in range(0, T, CHUNK):
                c1 = min(c0 + CHUNK, T)
                chunk_ys, A, S, zk = ckpt.checkpoint(
                    self.slow_heads[0]._recurrence_chunk,
                    kf[:,c0:c1], Q[:,c0:c1], V[:,c0:c1], U[:,c0:c1],
                    A, S, zk, I, c0,
                    use_reentrant=False
                )
                all_ys.append(chunk_ys)
                
            ys = torch.cat(all_ys, 1)
            for i,h in enumerate(self.slow_heads):
                yi = ys[i*B:(i+1)*B]
                outs.append(h.Wo(h.norm(yi)))
                
        return self.Wo(self.norm(torch.cat(outs,-1)))


# ── Backbone ──────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self,attn,d,ff=2):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.attn=attn
        self.ff=nn.Sequential(nn.Linear(d,d*ff),nn.GELU(),nn.Linear(d*ff,d))
    def forward(self,x): return x+self.ff(self.ln2(x+self.attn(self.ln1(x))))

class TinyLM(nn.Module):
    def __init__(self,attn_fn,d,vocab=VOCAB,n_layers=N_LAYERS):
        super().__init__()
        self.tok=nn.Embedding(vocab,d); self.pos=nn.Embedding(8192,d)
        self.blocks=nn.ModuleList([Block(attn_fn(d),d) for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(d)
        self.head=nn.Linear(d,vocab,bias=False); self.head.weight=self.tok.weight
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)
    def forward(self,idx):
        B,T=idx.shape
        x=self.tok(idx)+self.pos(torch.arange(T,device=idx.device).unsqueeze(0))
        for b in self.blocks: x=b(x)
        return self.ln_f(x)@self.tok.weight.T


MODEL_REGISTRY = {
    'Uniform-VLA': (lambda d: UniformVLA(d,H=UNIF_H), D_UNIF,  C['uniform']),
    'Hetero-VLA':  (lambda d: HeteroVLA(),             D_HETERO, C['hetero']),
}

print('NaN + checkpoint sanity:')
for name,(factory,d_model,_) in MODEL_REGISTRY.items():
    m=TinyLM(factory,d_model,VOCAB).to(DEVICE)
    x=torch.randint(0,VOCAB,(2,25),device=DEVICE)
    with torch.amp.autocast('cuda', enabled=(DEVICE=='cuda')):
        o=m(x)
    p=sum(q.numel() for q in m.parameters())/1e6
    cap=factory(d_model).capacity() if hasattr(factory(d_model),'capacity') else '?'
    print(f'  {name:14s}: NaN={torch.isnan(o).any().item()}  params={p:.2f}M  '
          f'd={d_model}  capacity={cap}')
    del m; gc.collect()
    if DEVICE=='cuda': torch.cuda.empty_cache()
"""

for c in nb['cells']:
    if c.get('id') == 'task':
        c['source'] = task_code
    elif c.get('id') == 'models':
        c['source'] = models_code

json.dump(nb, open(fp, 'w', encoding='utf-8'), indent=1)
print("Notebook patched.")
