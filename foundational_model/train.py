import torch
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM
from foundational_model.dataset import get_dataloader
import time

def train():
    # 1. Initialize Configuration and Model
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing VLA India Foundational Model on {device}...")
    model = VLACausalLM(config).to(device)
    
    # 2. Setup Optimizer and Dataloader
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    batch_size = 16
    seq_len = 256
    
    dataloader = get_dataloader(batch_size, config.vocab_size, seq_len)
    
    # 3. Simple Training Loop
    model.train()
    epochs = 1
    
    print("Starting training...")
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (crucial for RNN/Linear Attention variants)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
    print("Training complete! You can now save the weights or evaluate.")
    # torch.save(model.state_dict(), "vla_foundational_india_12M.pth")

if __name__ == "__main__":
    train()
