import torch
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM
from foundational_model.dataset import get_mixed_dataloader
from torch.amp import autocast

def train_pt():
    """
    Pre-Training (PT) Phase: Mass token ingestion across mixed datasets.
    """
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing VLA Pre-Training on {device}...")
    model = VLACausalLM(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda') # For mixed precision
    
    batch_size = 4 # Reduced to prevent Kaggle T4 OOM
    seq_len = 256
    gradient_accumulation_steps = 16 # Simulate batch_size = 64
    
    dataloader = get_mixed_dataloader(batch_size, seq_len)
    
    model.train()
    
    print("Starting Pre-Training Loop...")
    step_count = 0
    total_loss = 0.0
    
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Mixed precision forward pass
        with autocast('cuda'):
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
            
        # Backward pass
        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            step_count += 1
            if step_count % 10 == 0:
                print(f"PT Step {step_count} | Running Loss: {total_loss / 10:.4f}")
                total_loss = 0.0
                
        # Break after some steps for Kaggle prototyping
        if step_count >= 5000:
            break
            
    print("Pre-Training complete! Saving checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vla_pt_12M.pth")

if __name__ == "__main__":
    train_pt()
