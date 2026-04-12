import torch
import torch.nn as nn
from src.models.attention.vla import VLALayer
from scripts.benchmark_retrieval import ModelWrapper, generate_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(model_cls, name):
    model = ModelWrapper(model_cls, d_model=64, vocab_size=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining {name} for 20 steps:")
    for step in range(21):
        x, y = generate_batch(16, 64, 32, 10, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        acc = (logits.argmax(-1) == y).float().mean()
        if step % 5 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

test_model(VLALayer, "VLA")
