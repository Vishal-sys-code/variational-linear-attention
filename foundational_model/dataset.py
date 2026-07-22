import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import os

class MixedFoundationalDataset(IterableDataset):
    """
    Streams and mixes data from multiple global sources:
    1. English/Reasoning (e.g., FineWeb-Edu)
    2. Code (e.g., The Stack or CodeParrot)
    3. Indic Languages (e.g., Hindi Wikipedia)
    """
    def __init__(self, seq_len: int, tokenizer_name: str = "bert-base-multilingual-cased"):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load streaming datasets to avoid massive memory overhead on Kaggle
        print("Loading datasets for mixing...")
        ds_en = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        ds_code = load_dataset("codeparrot/github-code", streaming=True, split="train") # Example code dataset
        ds_hi = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
        
        # We need to map all datasets to yield a 'text' column
        def extract_text_en(x): return {"text": x["text"]}
        def extract_text_code(x): return {"text": x["code"]}
        def extract_text_hi(x): return {"text": x["text"]}
        
        ds_en = ds_en.map(extract_text_en, remove_columns=list(ds_en.features))
        ds_code = ds_code.map(extract_text_code, remove_columns=list(ds_code.features))
        ds_hi = ds_hi.map(extract_text_hi, remove_columns=list(ds_hi.features))
        
        # Mix the datasets! DeepMind/OpenAI usually do ~70% English/Reasoning, 15% Code, 15% Multilingual
        self.mixed_dataset = interleave_datasets(
            [ds_en, ds_code, ds_hi], 
            probabilities=[0.7, 0.15, 0.15],
            seed=42
        )

    def __iter__(self):
        buffer = []
        for example in self.mixed_dataset:
            text = example['text']
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)
            
            # Yield chunks of size (seq_len + 1) to form x and y
            while len(buffer) > self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                x = tensor_chunk[:-1]
                y = tensor_chunk[1:]
                yield x, y

def get_mixed_dataloader(batch_size: int, seq_len: int):
    dataset = MixedFoundationalDataset(seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size)
