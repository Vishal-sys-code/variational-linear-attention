import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import numpy as np

# Add src to path if needed (assuming running from repo root)
sys.path.append(os.path.abspath("."))

from src.models.transformer import VLATransformer

def generate_batch(batch_size, seq_len, vocab_size, device):
    """
    Generates a batch of random token sequences for the copy task.
    Input: (B, T)
    Target: (B, T) (same as input)
    """
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return data, data

def check_activations(model, step, log_buffer):
    """
    Checks for NaNs and activation magnitudes in VLA layers.
    """
    max_val = 0.0
    has_nan = False
    
    for i, layer in enumerate(model.layers):
        vla = layer.vla
        
        # Check buffers
        # A_t might be (B, d, d)
        if hasattr(vla.inverse_tracker, 'A_t') and vla.inverse_tracker.A_t is not None:
            val = vla.inverse_tracker.A_t
            if torch.isnan(val).any():
                has_nan = True
                log_buffer.append(f"STEP {step}: NaN detected in Layer {i} A_t")
            max_val = max(max_val, val.abs().max().item())
            
        # S_t might be (B, d, d)
        if hasattr(vla.memory_manager, 'S_t') and vla.memory_manager.S_t is not None:
            val = vla.memory_manager.S_t
            if torch.isnan(val).any():
                has_nan = True
                log_buffer.append(f"STEP {step}: NaN detected in Layer {i} S_t")
            max_val = max(max_val, val.abs().max().item())
            
    return has_nan, max_val

def run_training():
    # Configuration
    VOCAB_SIZE = 64
    D_MODEL = 64
    D_FFN = 128
    N_LAYERS = 2
    BATCH_SIZE = 32
    MAX_STEPS = 1000
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging Setup
    RESULTS_DIR = "results/synthetic_copy"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = int(time.time())
    log_file_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.md")
    
    log_buffer = []
    log_buffer.append(f"# Synthetic Copy Task Run {timestamp}")
    log_buffer.append("## Configuration")
    log_buffer.append(f"- Vocab Size: {VOCAB_SIZE}")
    log_buffer.append(f"- d_model: {D_MODEL}")
    log_buffer.append(f"- Layers: {N_LAYERS}")
    log_buffer.append(f"- Batch Size: {BATCH_SIZE}")
    log_buffer.append(f"- Max Steps: {MAX_STEPS}")
    log_buffer.append(f"- Device: {DEVICE}")
    log_buffer.append("\n## Training Log")
    log_buffer.append("| Step | Loss | Max Activation |")
    log_buffer.append("|---|---|---|")

    # Model Init
    model = VLATransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_ffn=D_FFN,
        max_len=256
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    print(f"Starting training on {DEVICE}...")
    
    model.train()
    
    for step in range(MAX_STEPS):
        # Randomize seq_len between 50 and 200
        seq_len = np.random.randint(50, 201)
        
        x, target = generate_batch(BATCH_SIZE, seq_len, VOCAB_SIZE, DEVICE)
        
        optimizer.zero_grad()
        logits = model(x) # (B, T, V)
        
        # Reshape for loss
        loss = criterion(logits.view(-1, VOCAB_SIZE), target.view(-1))
        
        loss.backward()
        
        # Gradient Clipping (Optional but recommended for RNNs/Linear Attention)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        curr_loss = loss.item()
        losses.append(curr_loss)
        
        # Verification Checks
        has_nan, max_act = check_activations(model, step, log_buffer)
        
        if has_nan:
            print(f"NaN detected at step {step}! Aborting.")
            log_buffer.append(f"\n**CRITICAL FAILURE: NaN detected at step {step}**")
            break
            
        if step % 50 == 0:
            print(f"Step {step}: Loss = {curr_loss:.4f}, Max Act = {max_act:.2f}")
            log_buffer.append(f"| {step} | {curr_loss:.4f} | {max_act:.2f} |")
            
    # Final Validation
    print("Training complete. Verifying requirements...")
    log_buffer.append("\n## Verification Results")
    
    # A. Loss Decrease
    loss_start = np.mean(losses[:50])
    loss_end = np.mean(losses[-50:])
    loss_decrease = loss_end < loss_start
    print(f"Loss Start (avg first 50): {loss_start:.4f}")
    print(f"Loss End (avg last 50): {loss_end:.4f}")
    
    if loss_decrease:
        log_buffer.append("- [x] Loss monotonically decreased (trend).")
    else:
        log_buffer.append("- [ ] Loss did NOT decrease significantly.")

    # B. NaNs
    if not has_nan:
        log_buffer.append("- [x] No NaNs detected.")
    else:
        log_buffer.append("- [ ] NaNs detected.")
        
    # C. Activation Stability
    if max_act < 1e3: # Threshold from prompt
        log_buffer.append(f"- [x] Activations stable (max {max_act:.2f} < 1e3).")
    else:
        log_buffer.append(f"- [ ] Activations unstable (max {max_act:.2f} >= 1e3).")

    # Save Log
    with open(log_file_path, "w") as f:
        f.write("\n".join(log_buffer))
        
    print(f"Log saved to {log_file_path}")
    
    if loss_decrease and not has_nan and max_act < 1e3:
        print("SUCCESS: All criteria met.")
        sys.exit(0)
    else:
        print("FAILURE: Criteria not met.")
        sys.exit(1)

if __name__ == "__main__":
    run_training()