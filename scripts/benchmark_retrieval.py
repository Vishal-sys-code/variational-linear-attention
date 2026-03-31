import os
import time
import json
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.attention.vla import VLALayer
from src.models.attention.deltanet import DeltaNetLayer
from src.models.attention.linear_transformer import LinearTransformerLayer

class CausalConv1d(nn.Module):
    """
    Standard short convolution used in sub-quadratic architectures (Mamba, DeltaNet)
    to locally mix tokens (e.g., contiguous key-value pairs).
    """
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=self.padding, groups=d_model)
        
    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        h = x.transpose(1, 2)
        h = self.conv(h)
        # Slice padding to retain causality
        h = h[:, :, :-self.padding]
        h = h.transpose(1, 2) # (B, T, D)
        return nn.functional.silu(h)

class ModelWrapper(nn.Module):
    def __init__(self, layer_cls, d_model, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        # Causal convolution fuses chronologically adjacent key-value pairs conceptually
        self.conv = CausalConv1d(d_model, kernel_size=4)
        self.norm = nn.LayerNorm(d_model)
        self.layer = layer_cls(d_model=d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        h = self.emb(x)
        h = self.conv(h)
        h = self.norm(h)
        out = self.layer(h)
        if isinstance(out, tuple):
            out = out[0] # Handle return_states optionally
        logits = self.head(out[:, -1, :])
        return logits

def generate_batch(batch_size, sequence_length, vocab_size, num_pairs, device):
    """
    Dynamically generates batched associative retrieval sequences to prevent overfitting.
    Pattern: [k1, v1, k2, v2, ... q_k]
    Target: q_v
    """
    seqs = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        keys = torch.randperm(vocab_size - 1, device=device)[:num_pairs] + 1
        values = torch.randint(1, vocab_size, (num_pairs,), device=device)
        
        for i in range(num_pairs):
            seqs[b, 2*i] = keys[i]
            seqs[b, 2*i + 1] = values[i]
            
        target_idx = torch.randint(0, num_pairs, (1,)).item()
        query_key = keys[target_idx]
        target_value = values[target_idx]
        
        seqs[b, -1] = query_key
        targets[b] = target_value
        
    return seqs, targets

def train_and_eval(model, args, device):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    csv_path = f"results/retrieval_{args.model}_{args.seq_len}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_acc", "epoch_time_sec"])
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            start_time = time.time()
            
            # Simulated training batches per epoch
            batches_per_epoch = args.num_train_samples // args.batch_size
            for _ in range(batches_per_epoch):
                x, y = generate_batch(args.batch_size, args.seq_len, args.vocab_size, args.num_pairs, device)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / batches_per_epoch
            
            # Validation
            model.eval()
            correct = 0
            test_batches = args.num_test_samples // args.batch_size
            with torch.no_grad():
                for _ in range(test_batches):
                    x, y = generate_batch(args.batch_size, args.seq_len, args.vocab_size, args.num_pairs, device)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y).sum().item()
            
            acc = correct / (test_batches * args.batch_size)
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f} | Time: {epoch_time:.2f}s")
            writer.writerow([epoch, avg_loss, acc, epoch_time])
            
    return acc

def main():
    parser = argparse.ArgumentParser(description="Associative Retrieval Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["VLA", "DeltaNet", "Linear"], help="Model architecture")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence Length")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_train_samples", type=int, default=1024, help="Samples per epoch for training")
    parser.add_argument("--num_test_samples", type=int, default=256, help="Samples for validation")
    parser.add_argument("--vocab_size", type=int, default=32, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_pairs", type=int, default=5, help="Key-Value pairs to memorize")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {args.model} with SeqLen={args.seq_len}")
    
    models = {
        "VLA": VLALayer,
        "DeltaNet": DeltaNetLayer,
        "Linear": LinearTransformerLayer
    }
    
    layer_cls = models[args.model]
    model = ModelWrapper(layer_cls, args.d_model, args.vocab_size).to(device)
    
    # Execute loop
    final_acc = train_and_eval(model, args, device)
    print(f"\nFinal Test Accuracy for {args.model} at Length {args.seq_len}: {final_acc:.4f}")

if __name__ == "__main__":
    main()