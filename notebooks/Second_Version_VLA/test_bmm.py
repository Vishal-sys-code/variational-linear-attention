import time
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)
if DEVICE == 'cuda':
    print("GPU:", torch.cuda.get_device_name())

A = torch.randn(256, 128, 128, device=DEVICE)
B = torch.randn(256, 128, 1, device=DEVICE)

torch.cuda.synchronize() if DEVICE == 'cuda' else None
t0 = time.time()
for _ in range(1000):
    C = torch.bmm(A, B)
torch.cuda.synchronize() if DEVICE == 'cuda' else None
print("1000 BMMs:", time.time() - t0)
