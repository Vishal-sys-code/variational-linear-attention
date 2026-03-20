import os
import argparse
from pathlib import Path
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: `pip install datasets`")
    import sys
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/lra/clutrr", help="Output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="Max byte length for characters")
    args = parser.parse_args()

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CLUTRR via HuggingFace datasets...")
    # `kendrivp/CLUTRR_v1_extracted` is a pre-extracted Parquet version of CLUTRR/v1
    # which avoids the `trust_remote_code=True` error on the Hub.
    dataset = load_dataset('kendrivp/CLUTRR_v1_extracted')
    
    # We will build a vocabulary of target relations on the fly or just collect them
    relation_map = {}

    for split_key in ['train', 'test']:
        if split_key not in dataset:
            continue
            
        ds = dataset[split_key]
        print(f"Processing split: {split_key} ({len(ds)} examples)")
        
        inputs_list = []
        labels_list = []
        
        # In CLUTRR v1 extracted, typical keys are `story`, `query`, `target_text`, etc.
        # Format string: "Story: {story} Query: {query}"
        for i, example in enumerate(ds):
            # Fallbacks for different possible keys in the dataset
            story = example.get('story', example.get('text', ''))
            query = example.get('query', '')
            
            # The target is the kinship relation (e.g., 'grandfather')
            target = example.get('target_text', example.get('target', example.get('relation', '')))
            
            # Map target string to integer ID
            if target not in relation_map:
                relation_map[target] = len(relation_map)
            label_idx = relation_map[target]
            
            formatted_text = f"Story: {story}\nQuery: {query}\n"
            
            chars = [ord(c) for c in formatted_text if c != '\n']
            if len(chars) > args.max_length:
                chars = chars[:args.max_length]
            else:
                chars = chars + [0] * (args.max_length - len(chars))
                
            inputs_list.append(chars)
            labels_list.append(label_idx)
            
            if (i+1) % 5000 == 0:
                print(f"  Tokenized {i+1} examples...")

        inputs_np = np.array(inputs_list, dtype=np.int32)
        labels_np = np.array(labels_list, dtype=np.int64)
        
        # Save as standard names for LRA pipeline: train_inputs.npy, test_inputs.npy
        save_split = "validation" if split_key == "test" else split_key  # LRA uses validation usually
        np.save(data_dir / f"{save_split}_inputs.npy", inputs_np)
        np.save(data_dir / f"{save_split}_labels.npy", labels_np)

    # Save mapping for evaluation
    with open(data_dir / "relation_map.txt", "w") as f:
        for rel, idx in relation_map.items():
            f.write(f"{rel}\t{idx}\n")

    print(f"Success! CLUTRR data written to {data_dir}. Found {len(relation_map)} relations.")

if __name__ == "__main__":
    main()
