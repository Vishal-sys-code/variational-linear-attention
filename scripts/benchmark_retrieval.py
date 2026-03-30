import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.models.attention.vla import VLALayer
from src.models.attention.deltanet import DeltaNetLayer
from src.models.attention.linear_transformer import LinearTransformerLayer

class AssociativeDataset(Dataset):
    def __init__(self, num_samples, sequence_length, vocab_size=32, num_pairs=5):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_pairs = num_pairs
        
        # Precompute the dataset for faster iterations
        self.data = []
        for _ in range(num_samples):
            seq = torch.zeros(self.sequence_length, dtype=torch.long)
            # Keys and values from [1, vocab_size-1]
            keys = torch.randperm(self.vocab_size - 1)[:self.num_pairs] + 1
            values = torch.randint(1, self.vocab_size, (self.num_pairs,))
            
            # Place them at the beginning
            for i in range(self.num_pairs):
                seq[2*i] = keys[i]
                seq[2*i + 1] = values[i]
                
            # Randomly select one query key
            target_idx = torch.randint(0, self.num_pairs, (1,)).item()
            query_key = keys[target_idx]
            target_value = values[target_idx]
            
            # Put query key at the end
            seq[-1] = query_key
            
            self.data.append((seq, target_value))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

class ModelWrapper(nn.Module):
    def __init__(self, layer_cls, d_model, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layer = layer_cls(d_model=d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        h = self.emb(x)
        out = self.layer(h)
        logits = self.head(out[:, -1, :])
        return logits

def train_and_eval(model, train_loader, test_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sequence_lengths = [1000, 5000, 10000]
    num_train_samples = 200
    num_test_samples = 50
    vocab_size = 32
    d_model = 64
    batch_size = 16
    
    results = {
        "sequence_lengths": sequence_lengths,
        "VLA_accuracy": [],
        "DeltaNet_accuracy": [],
        "Linear_accuracy": []
    }
    
    models = {
        "VLA": VLALayer,
        "DeltaNet": DeltaNetLayer,
        "Linear": LinearTransformerLayer
    }
    
    print("Running the associative retrieval benchmark...")
    
    for N in sequence_lengths:
        print(f"\nSequence Length: {N}")
        
        train_ds = AssociativeDataset(num_train_samples, N, vocab_size)
        test_ds = AssociativeDataset(num_test_samples, N, vocab_size)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        for name, layer_cls in models.items():
            model = ModelWrapper(layer_cls, d_model, vocab_size).to(device)
            # The prompt says we need evidence, not large scale training, so a quick training loop
            # is appropriate here (10 epochs for this toy task)
            acc = train_and_eval(model, train_loader, test_loader, device, epochs=10)
            
            print(f"{name} Accuracy: {acc:.2f}")
            results[f"{name}_accuracy"].append(acc)
            
    os.makedirs('results', exist_ok=True)
    with open('results/associative_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to results/associative_results.json")

if __name__ == "__main__":
    main()