import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class LRADataset(Dataset):
    """
    Standard PyTorch Dataset for LRA tasks.
    Reads pre-cached .npy files.
    """
    def __init__(self, data_dir: str, task: str, split: str = "train"):
        super().__init__()
        self.task = task
        self.split = split
        self.data_dir = Path(data_dir) / task
        
        inputs_path = self.data_dir / f"{split}_inputs.npy"
        labels_path = self.data_dir / f"{split}_labels.npy"
        
        if not inputs_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"LRA datasets not found at {self.data_dir}. Run download_lra.py first.")
            
        # load arrays using mmap_mode='r' to keep memory usage extremely low
        self.inputs = np.load(str(inputs_path), mmap_mode='r')
        self.labels = np.load(str(labels_path), mmap_mode='r')
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # returns tensors directly
        # For character/byte tokens, sequence padding naturally contains 0.
        x = torch.from_numpy(self.inputs[idx].astype(np.int64))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # We assume 0 is the padding token across all LRA subsets.
        # This matches standard setup. Sequences were padded/truncated
        # appropriately in the export script.
        attention_mask = (x != 0).float()
        
        return {
            "input_ids": x,
            "attention_mask": attention_mask,
            "labels": y
        }

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def get_lra_dataloader(
    data_dir: str, 
    task: str, 
    split: str, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4,
    seed: int = 42
) -> DataLoader:
    """
    Creates dataloaders deterministically according to LRA standards.
    """
    # Determinism
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataset = LRADataset(data_dir=data_dir, task=task, split=split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=(split == 'train')
    )
