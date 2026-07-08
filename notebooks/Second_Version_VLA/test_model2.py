import math, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHUNK = 32
B = 16
T = 97
dh = 128
H = 8

def recurrence_chunk(kf_c, Q_c, V_c, U_c, A, S, zk, I, c_start, eps=1e-4, period=20, per_eps=1e-3):
    with torch.amp.autocast(device_type=kf_c.device.type, enabled=False):
        B, C_len, d = kf_c.shape
        isq = 1/math.sqrt(d)
        ys = []
        for t in range(C_len):
            gt = c_start + t
            u = U_c[:,t,:] * isq
            zsm = torch.bmm(A, u.unsqueeze(-1)).squeeze(-1)
            dlt = (1 + (u*zsm).sum(-1)).clamp(min=eps)
            A = A - torch.bmm(zsm.unsqueeze(-1), zsm.unsqueeze(1)) / dlt.view(B,1,1)
            if (gt+1) % period == 0:
                A = A + per_eps * I.unsqueeze(0)
            kn = F.normalize(kf_c[:,t,:], p=2, dim=-1)
            alpha = torch.bmm(A, kn.unsqueeze(-1)).squeeze(-1)
            alphn = F.normalize(alpha, p=2, dim=-1)
            e = V_c[:,t,:] - torch.bmm(S, kn.unsqueeze(-1)).squeeze(-1)
            S = S + torch.bmm(e.unsqueeze(-1), alphn.unsqueeze(1))
            qt = Q_c[:,t,:]
            zk = zk + kf_c[:,t,:]
            yt = torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1)
            ys.append(yt / (zk*qt).sum(-1, keepdim=True).clamp(min=eps))
        return torch.stack(ys, 1), A, S, zk

class VLAv3HeadBatched(nn.Module):
    def __init__(self, dh, lam=0.1, eps=1e-4, per_eps=1e-3, period=20, chunk=CHUNK):
        super().__init__()
        self.dh=dh; self.lam=lam; self.eps=eps
        self.per_eps=per_eps; self.period=period; self.chunk=chunk
        self.Wq=nn.Linear(dh,dh); self.Wk=nn.Linear(dh,dh)
        self.Wv=nn.Linear(dh,dh); self.Wu=nn.Linear(dh,dh,bias=False)
        self.Wo=nn.Linear(dh,dh); self.norm=nn.LayerNorm(dh)

    def forward_projections(self, x):
        kr = self.Wk(x); kf = F.elu(kr) + 1.0
        Q  = F.elu(self.Wq(x)) + 1.0; V = self.Wv(x)
        U  = F.normalize(self.Wu(kr), p=2, dim=-1)
        return kf, Q, V, U

    def forward_output(self, ys):
        return self.Wo(self.norm(ys))

class UniformVLABatched(nn.Module):
    def __init__(self, d_model, H=8):
        super().__init__()
        self.H = H; self.dh = d_model//H
        self.heads = nn.ModuleList([VLAv3HeadBatched(self.dh) for _ in range(H)])
        self.Wo = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        dh = self.dh
        
        # 1. Gather all projections
        kfs, Qs, Vs, Us = [], [], [], []
        for i, h in enumerate(self.heads):
            xi = x[:, :, i*dh:(i+1)*dh]
            kf, Q, V, U = h.forward_projections(xi)
            kfs.append(kf); Qs.append(Q); Vs.append(V); Us.append(U)
            
        kf = torch.cat(kfs, dim=0) # (B*H, T, dh)
        Q = torch.cat(Qs, dim=0)
        V = torch.cat(Vs, dim=0)
        U = torch.cat(Us, dim=0)
        
        # 2. Run recurrence ONCE for all heads
        kf, Q, V, U = kf.float(), Q.float(), V.float(), U.float()
        I = torch.eye(dh, device=x.device, dtype=torch.float32)
        A = (1/self.heads[0].lam) * I.unsqueeze(0).expand(B*self.H, -1, -1).clone()
        S = torch.zeros(B*self.H, dh, dh, device=x.device, dtype=torch.float32)
        zk = torch.zeros(B*self.H, dh, device=x.device, dtype=torch.float32)
        
        all_ys = []
        for c0 in range(0, T, CHUNK):
            c1 = min(c0 + CHUNK, T)
            chunk_ys, A, S, zk = ckpt.checkpoint(
                recurrence_chunk,
                kf[:,c0:c1], Q[:,c0:c1], V[:,c0:c1], U[:,c0:c1],
                A, S, zk, I, c0, self.heads[0].eps, self.heads[0].period, self.heads[0].per_eps,
                use_reentrant=False
            )
            all_ys.append(chunk_ys)
        
        ys = torch.cat(all_ys, 1) # (B*H, T, dh)
        
        # 3. Output projections
        outs = []
        for i, h in enumerate(self.heads):
            ys_i = ys[i*B:(i+1)*B]
            outs.append(h.forward_output(ys_i))
            
        return self.Wo(self.norm(torch.cat(outs, -1)))

model = UniformVLABatched(1024, 8).to(DEVICE)
x = torch.randn(32, 97, 1024, device=DEVICE)

# warmup
y = model(x)
loss = y.sum()
loss.backward()

torch.cuda.synchronize() if DEVICE == 'cuda' else None
t0 = time.time()
for _ in range(2):
    y = model(x)
    loss = y.sum()
    loss.backward()
torch.cuda.synchronize() if DEVICE == 'cuda' else None

print("Batched Model 2 fwd-bwd passes:", time.time() - t0)
