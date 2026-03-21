import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import LRAModel
from src.data.lra_dataloaders import get_lra_dataloader

def compute_survival(model, step):
    # Retrieve the state of inverse_tracker/memory_manager periodically 
    # to log survival factor approximations.
    # Currently a proxy metric based on W&B custom dashboards
    pass

def train_worker(args):
    # Offline fallback
    if not os.environ.get("WANDB_API_KEY"):
        wandb.init(mode="offline", project="variational-linear-attention", entity="vla-research", name=f"{args.task}-{args.model}-seed{args.seed}")
    else:
        wandb.init(project="variational-linear-attention", entity="vla-research", name=f"{args.task}-{args.model}-seed{args.seed}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Task specific lengths, vocabs, and classes
    lengths = {"listops": 2000, "retrieval": 4096, "pathfinder": 1024, "cqa": 1024, "clutrr": 1024}
    # ListOps chars (ASCII) map up to ~125 ('}'), so 256 covers all safely like other byte-level tasks
    vocabs = {"listops": 256, "retrieval": 256, "pathfinder": 256, "cqa": 256, "clutrr": 256}
    
    # ListOps typically has 10 classes (0-9). CQA has 5. CLUTRR has 21 (v1). Pathfinder/Retrieval have 2.
    classes = {"listops": 10, "retrieval": 2, "pathfinder": 2, "cqa": 5, "clutrr": 21}
    
    max_len = lengths[args.task]
    vocab_size = vocabs[args.task]
    num_classes = classes[args.task]
    
    try:
        train_loader = get_lra_dataloader("data/lra", args.task, "train", batch_size=32, seed=args.seed)
        val_loader = get_lra_dataloader("data/lra", args.task, "validation", batch_size=32, seed=args.seed)
    except FileNotFoundError:
        print(f"Could not load data for {args.task}. Ensure download_lra.py has been run.")
        return

    model = LRAModel(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        d_ffn=1024,
        max_len=max_len,
        dropout=0.1,
        attention_type=args.model,
        num_classes=num_classes
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1): # typically more depending on standard but limiting for default
        model.train()
        for step, batch in enumerate(train_loader):
            ts = time.time()
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Request states for returning metrics every 100 steps if VLA
            return_states = (args.model == "vla" and step % 100 == 0)
            
            output = model(x, pool=True, return_states=return_states)
            
            if return_states:
                logits, states = output
                # logging logic
                if "S_norm" in states:
                    s_norm_mean = states["S_norm"].mean().item()
                    wandb.log({"norm_S_t": s_norm_mean}, step=step)
            else:
                logits = output
                
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            dur = time.time() - ts
            throughput = x.size(0) * x.size(1) / dur
            
            if step % 50 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "training_step_time": dur,
                    "tokens_per_second": throughput,
                    "gpu_memory_usage": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                })
                
            if step >= 1000: # for testing early stopping. Real runs use epochs
                break
                
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                x = val_batch["input_ids"].to(device)
                y = val_batch["labels"].to(device)
                logits = model(x, pool=True)
                val_loss += criterion(logits, y).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                if total > 500: # partial eval for speed in this mock
                    break
        
        v_acc = correct / total
        v_loss = val_loss / len(val_loader)
        wandb.log({
            "validation_accuracy": v_acc,
            "validation_loss": v_loss
        })
        print(f"Run {args.task}-{args.model}-seed{args.seed} finished. Val Acc: {v_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_worker(args)
