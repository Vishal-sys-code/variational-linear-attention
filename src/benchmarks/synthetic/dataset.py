import torch
from torch.utils.data import Dataset
import random

class CopyTaskDataset(Dataset):
    """
    Generates random integer sequences of given length from a vocabulary.
    Target is identical to the input.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 200, vocab_size: int = 10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence of tokens
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        return seq, seq.clone()


class DelayedRecallDataset(Dataset):
    """
    To predict a token D steps after it is presented.
    We generate random sequences, and target shifts them by D, with PAD token before.
    E.g., for D=3, seq=[a, b, c, d, e], target=[PAD, PAD, PAD, a, b]
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 200, delay: int = 5, vocab_size: int = 10, pad_token: int = 0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.delay = delay
        self.vocab_size = vocab_size
        self.pad_token = pad_token

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random sequence, reserve token 0 for PAD.
        # Assuming vocab starts at pad_token + 1 to avoid overlap.
        seq = torch.randint(1, self.vocab_size, (self.seq_len,))
        
        target = torch.full_like(seq, self.pad_token)
        if self.delay < self.seq_len:
            target[self.delay:] = seq[:-self.delay].clone()
            
        return seq, target


class AssociativeRecallDataset(Dataset):
    """
    Input contains pairs of (Key, Value) and later queries for Keys.
    Target outputs Value at Query position.
    
    Structure:
    [K1, V1, K2, V2, ..., K_N, V_N, Q1, Q2, ...]
    """
    def __init__(self, num_samples: int = 1000, num_pairs: int = 10, num_queries: int = 5, num_distractors: int = 0, vocab_size: int = 100, pad_token: int = 0):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.num_distractors = num_distractors
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        
        # New: Reserve a specific index for the QUERY token
        self.query_token = vocab_size - 1
        
        self.seq_len = (num_pairs * 2) + num_distractors + 1 + num_queries # +1 for QUERY token

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Keys and values from distinct spaces or same? Same vocab space.
        # Ensure keys are unique within a sequence
        # We reserve 0 for PAD, vocab_size-1 for QUERY. 
        # So we draw from [1, vocab_size - 2].
        keys = torch.randperm(self.vocab_size - 2)[:self.num_pairs] + 1
        values = torch.randint(1, self.vocab_size - 1, (self.num_pairs,))
        
        seq = []
        # Add pairs
        for k, v in zip(keys, values):
            seq.extend([k.item(), v.item()])
            
        # Add distractors (if any)
        if self.num_distractors > 0:
            distractors = torch.randint(1, self.vocab_size - 1, (self.num_distractors,))
            seq.extend(distractors.tolist())
            
        # Add QUERY token
        seq.append(self.query_token)
            
        # Add queries
        query_indices = torch.randint(0, self.num_pairs, (self.num_queries,))
        queries = keys[query_indices]
        seq.extend(queries.tolist())
        
        seq = torch.tensor(seq, dtype=torch.long)
        
        # Target: Pad everywhere except query positions
        # The query positions start right after the QUERY token.
        target = torch.full_like(seq, self.pad_token)
        query_start_idx = len(seq) - self.num_queries
        target[query_start_idx:] = values[query_indices]
        
        return seq, target
