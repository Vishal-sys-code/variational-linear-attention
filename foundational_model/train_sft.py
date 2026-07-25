import torch
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM

def create_sft_batch(batch_size=4, seq_len=256):
    """
    Mock SFT batch generator for testing.
    In production, use HuggingFaceH4/ultrachat_200k.
    SFT requires a custom collator that applies an ignore_index to the User prompt
    so the model only calculates loss on the Assistant's response.
    """
    # Random tokens
    x = torch.randint(0, 32000, (batch_size, seq_len))
    # In SFT, we mask the prompt tokens with -100
    y = x.clone()
    y[:, :seq_len//2] = -100 # Mask first half (simulating User prompt)
    return x, y

def train_sft():
    """
    Supervised Fine-Tuning (SFT) Phase: Instruction alignment.
    """
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing VLA SFT on {device}...")
    model = VLACausalLM(config).to(device)
    if device == 'cuda':
        print("Compiling model for blazing fast training...")
        model = torch.compile(model)
    
    # In SFT, we load the pre-trained weights
    pt_checkpoint = "checkpoints/vla_pt_12M.pth"
    if os.path.exists(pt_checkpoint):
        model.load_state_dict(torch.load(pt_checkpoint))
        print("Loaded PT checkpoint!")
    else:
        print("Warning: No PT checkpoint found, starting from scratch for SFT.")
        
    # Learning rate is much lower in SFT, but bumped for Kaggle prototype
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    batch_size = 1
    gradient_accumulation_steps = 16
    
    model.train()
    print("Starting SFT Loop...")
    
    for step in range(100):
        optimizer.zero_grad()
        for micro_step in range(gradient_accumulation_steps):
            x, y = create_sft_batch(batch_size=batch_size, seq_len=256)
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 1 == 0:
            print(f"SFT Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")

    print("SFT complete! Saving checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vla_sft_12M.pth")

if __name__ == "__main__":
    train_sft()
