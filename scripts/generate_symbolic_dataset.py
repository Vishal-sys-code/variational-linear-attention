import os
import torch
import random
from typing import List, Tuple

# Vocab Definition
# Special: PAD=0
# Relations: 1=parent, 2=sibling, 3=friend, 4=cause, 5=part_of, 6=ancestor, 7=child
# Entities: 10 to 35 (A-Z)
RELATIONS = {
    'parent': 1, 'sibling': 2, 'friend': 3, 'cause': 4, 'part_of': 5,
    'ancestor': 6, 'child': 7
}
ENTITIES = {chr(65+i): 10+i for i in range(26)}
REVERSE_ENTRIES = {v: k for k, v in dict(**RELATIONS, **ENTITIES, PAD=0).items()}

def generate_sample() -> Tuple[List[int], int, torch.Tensor]:
    """
    Generates a single causal reasoning sequence of length ~8-12 tokens.
    Format: [E1, R1, E2, E2, R2, E3, Q_E1, Q_E3] -> Label
    Returns: (sequence, label, A_rel)
    """
    # Choose a pattern
    pattern = random.choice(['transitive_parent', 'transitive_part', 'transitive_sibling', 'reverse_parent', 'symmetric_friend'])
    
    e1, e2, e3 = random.sample(list(ENTITIES.values()), 3)
    
    seq = []
    label = 0
    
    if pattern == 'transitive_parent':
        seq = [e1, RELATIONS['parent'], e2, e2, RELATIONS['parent'], e3, e1, e3]
        label = RELATIONS['ancestor']
    elif pattern == 'transitive_part':
        seq = [e1, RELATIONS['part_of'], e2, e2, RELATIONS['part_of'], e3, e1, e3]
        label = RELATIONS['part_of']
    elif pattern == 'transitive_sibling':
        seq = [e1, RELATIONS['sibling'], e2, e2, RELATIONS['sibling'], e3, e1, e3]
        label = RELATIONS['sibling']
    elif pattern == 'reverse_parent':
        seq = [e1, RELATIONS['parent'], e2, e2, e1]
        label = RELATIONS['child']
    elif pattern == 'symmetric_friend':
        seq = [e1, RELATIONS['friend'], e2, e2, e1]
        label = RELATIONS['friend']
        
    T = len(seq)
    A_rel = torch.zeros(T, T)
    
    # Rule 1: Same Entity
    for i in range(T):
        if seq[i] in ENTITIES.values():
            for j in range(T):
                if seq[i] == seq[j]:
                    A_rel[i, j] = 1.0
                    A_rel[j, i] = 1.0
                    
    # Rule 2: Same triple / relational fact (Blocks of 3)
    # A fact is a slice [E1, R, E2]
    # For a pattern of length 8: [0,1,2] is a fact, [3,4,5] is a fact.
    # For length 5: [0,1,2] is a fact.
    if T >= 5:
        # Fact 1 -> 0,1,2
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                A_rel[i, j] = 1.0
                
    if T >= 8:
        # Fact 2 -> 3,4,5
        for i in [3, 4, 5]:
            for j in [3, 4, 5]:
                A_rel[i, j] = 1.0
                
        # Rule 3: Transitive chain (A -> B, B -> C implies A <-> C)
        # E1 is at 0, E3 is at 5
        A_rel[0, 5] = 1.0
        A_rel[5, 0] = 1.0
        
    return seq, label, A_rel

def create_dataset(num_samples: int):
    xs, ys, As = [], [], []
    max_len = 0
    
    for _ in range(num_samples):
        seq, label, A_rel = generate_sample()
        xs.append(torch.tensor(seq, dtype=torch.long))
        ys.append(torch.tensor(label, dtype=torch.long))
        As.append(A_rel)
        max_len = max(max_len, len(seq))
        
    # Pad to max_len
    x_padded = torch.zeros(num_samples, max_len, dtype=torch.long)
    A_padded = torch.zeros(num_samples, max_len, max_len)
    y_tensor = torch.stack(ys)
    
    for i in range(num_samples):
        L = len(xs[i])
        x_padded[i, :L] = xs[i]
        A_padded[i, :L, :L] = As[i]
        
    return {
        'x': x_padded,
        'y': y_tensor,
        'A_rel': A_padded
    }

def main():
    out_dir = "data/symbolic_reasoning"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating train set (5000)...")
    train_data = create_dataset(5000)
    torch.save(train_data, os.path.join(out_dir, "train.pt"))
    
    print("Generating val set (1000)...")
    val_data = create_dataset(1000)
    torch.save(val_data, os.path.join(out_dir, "val.pt"))
    
    print("Generating test set (1000)...")
    test_data = create_dataset(1000)
    torch.save(test_data, os.path.join(out_dir, "test.pt"))
    
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
