import time
import torch

from test_model import model, x, DEVICE

torch.cuda.synchronize() if DEVICE == 'cuda' else None
t0 = time.time()
y = model(x)
torch.cuda.synchronize() if DEVICE == 'cuda' else None
print("1 FWD:", time.time() - t0)

t0 = time.time()
loss = y.sum()
loss.backward()
torch.cuda.synchronize() if DEVICE == 'cuda' else None
print("1 BWD:", time.time() - t0)
