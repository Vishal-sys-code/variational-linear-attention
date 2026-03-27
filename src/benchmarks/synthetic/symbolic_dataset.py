import torch
from torch.utils.data import Dataset
import random

class SymbolicReasoningDataset(Dataset):
    """
    Synthetic multi-hop reasoning dataset.
    Given facts like: "A IS ANCESTOR OF B [SEP] B IS ANCESTOR OF C"
    Query: "QUERY A IS ANCESTOR OF C" -> Target: TRUE or FALSE
    
    Extracts A_rel where A_rel[i, j] = 1 if token i and token j represent the same entity (entity coreference).
    """
    def __init__(self, num_samples: int = 2000, num_facts: int = 4):
        self.num_samples = num_samples
        self.num_facts = num_facts
        
        # Vocab: PAD=0, SEP=1, QUERY=2, TRUE=3, FALSE=4, IS=5, ANCESTOR=6, OF=7
        self.pad_token = 0
        self.sep = 1
        self.query = 2
        self.true = 3
        self.false = 4
        self.is_t = 5
        self.anc = 6
        self.of = 7
        
        self.entities = list(range(8, 34)) # A-Z
        self.vocab_size = 35
        
        self.data = []
        self._generate_data()

    def _generate_data(self):
        for _ in range(self.num_samples):
            # Form a linear tree/chain
            num_nodes = self.num_facts + 1
            selected_entities = random.sample(self.entities, num_nodes)
            
            facts = []
            # Facts: E_i IS ANCESTOR OF E_{i+1}
            for i in range(self.num_facts):
                facts.append((selected_entities[i], selected_entities[i+1]))
                
            random.shuffle(facts)
            
            sequence = []
            for e1, e2 in facts:
                sequence.extend([e1, self.is_t, self.anc, self.of, e2, self.sep])
                
            # Question: E_a IS ANCESTOR OF E_b?
            # True path if a < b. False path if b < a (or unlinked, though all are linked in this chain)
            if random.random() > 0.5:
                # True
                a_idx = random.randint(0, num_nodes - 2)
                b_idx = random.randint(a_idx + 1, num_nodes - 1)
                ans = self.true
            else:
                # False (reverse direction)
                b_idx = random.randint(0, num_nodes - 2)
                a_idx = random.randint(b_idx + 1, num_nodes - 1)
                ans = self.false
                
            e_a = selected_entities[a_idx]
            e_b = selected_entities[b_idx]
            
            sequence.extend([self.query, e_a, self.is_t, self.anc, self.of, e_b])
            
            # Create Adjacency Matrix for "Same Entity"
            seq_len = len(sequence)
            A_rel = torch.zeros(seq_len, seq_len)
            
            # find all entities in sequence
            for i in range(seq_len):
                if sequence[i] in self.entities:
                    for j in range(seq_len):
                        if sequence[j] == sequence[i]:
                            A_rel[i, j] = 1.0

            x = torch.tensor(sequence, dtype=torch.long)
            y = torch.tensor(ans, dtype=torch.long)
            
            self.data.append((x, y, A_rel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_symbolic(batch):
    xs, ys, As = zip(*batch)
    
    # lengths are all equal in this simple formulation (num_facts * 6 + 6 = fixed length)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    A_rel = torch.stack(As, dim=0)
    
    return x, y, A_rel
