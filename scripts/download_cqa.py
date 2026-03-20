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
    parser.add_argument("--output_dir", type=str, default="data/lra/cqa", help="Output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="Max byte length for characters")
    args = parser.parse_args()

    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CommonsenseQA via HuggingFace datasets...")
    dataset = load_dataset('commonsense_qa')

    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    for split_name, hf_split in [("train", "train"), ("validation", "validation")]:
        ds = dataset[hf_split]
        print(f"Processing split: {split_name} ({len(ds)} examples)")
        
        inputs_list = []
        labels_list = []
        
        for i, example in enumerate(ds):
            question = example['question']
            choices = example['choices']  
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            
            # Format: 'Q: <question>\nA: <choice_a>\nB: <choice_b>...'
            formatted_text = f"Q: {question}\n"
            for lbl, text in zip(labels, texts):
                formatted_text += f"{lbl}: {text}\n"

            # Byte-level tokenization (ordinances) akin to LRA Text Classification
            chars = [ord(c) for c in formatted_text if c != '\n']
            
            # Pad or truncate
            if len(chars) > args.max_length:
                chars = chars[:args.max_length]
            else:
                chars = chars + [0] * (args.max_length - len(chars))
                
            ans_key = example['answerKey']
            label_idx = label_map.get(ans_key, 0)
            
            inputs_list.append(chars)
            labels_list.append(label_idx)
            
            if (i+1) % 2000 == 0:
                print(f"  Tokenized {i+1} examples...")

        inputs_np = np.array(inputs_list, dtype=np.int32)
        labels_np = np.array(labels_list, dtype=np.int64)
        
        np.save(data_dir / f"{split_name}_inputs.npy", inputs_np)
        np.save(data_dir / f"{split_name}_labels.npy", labels_np)

    print("Success! CommonsenseQA data written to", data_dir)

if __name__ == "__main__":
    main()
