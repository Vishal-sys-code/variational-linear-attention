import os
import sys
import subprocess
import argparse
from pathlib import Path
import json

def check_idempotency(data_dir: Path, tasks: list[str]) -> bool:
    all_exist = True
    for task in tasks:
        train_inputs = data_dir / task / 'train_inputs.npy'
        if not train_inputs.exists():
            all_exist = False
            break
    return all_exist

def preprocess_listops(dataset):
    import numpy as np
    # ListOps char-level tokenization
    # We will build a simple vocabulary over the characters on the fly, 
    # but optimally we just encode to bytes and map uniquely.
    # To be extremely safe and match standard approaches: 
    # We can just use byte values for characters since vocab is ~15
    # or build a specific char->int map.
    # Sequences are padded/truncated to 2000.
    MAX_LEN = 2000
    
    inputs_list = []
    labels_list = []
    
    # We will iterate over the dataset and get strings
    for idx, ex in enumerate(dataset):
        src = ex['source_text'].numpy().decode('utf-8') if 'source_text' in ex else ex['Source'].numpy().decode('utf-8')
        label = ex['Target'].numpy() if 'Target' in ex else ex['label'].numpy()
        
        # char-level
        chars = [ord(c) for c in src if c != ' '] # typically ignoring spaces is better, or just ord(c)
        # However, to be perfectly safe, let's just encode standard ascii bytes.
        
        # pad/truncate
        if len(chars) > MAX_LEN:
            chars = chars[:MAX_LEN]
        else:
            chars = chars + [0] * (MAX_LEN - len(chars)) # pad with 0
            
        inputs_list.append(chars)
        labels_list.append(label)
        
        if idx % 10000 == 0:
            print(f"Processed {idx} listops examples...")
            
    return np.array(inputs_list, dtype=np.int32), np.array(labels_list, dtype=np.int64)

def preprocess_retrieval(dataset):
    import numpy as np
    # sequence = document_A + document_B
    # Byte level, length 4096
    MAX_LEN = 4096
    
    inputs_list = []
    labels_list = []
    
    for idx, ex in enumerate(dataset):
        # depending on feature names in tfds
        # AAN uses 'text1' and 'text2' or 'source1' and 'source2'
        keys = list(ex.keys())
        # Typically for AAN retrieval it's 'text1', 'text2', 'label'
        d1 = ex.get('text1', ex.get('source1', tf.constant(b''))).numpy()
        d2 = ex.get('text2', ex.get('source2', tf.constant(b''))).numpy()
        label = ex.get('label', tf.constant(0)).numpy()
        
        # byte level
        seq = list(d1) + list(d2) # lists of ints 0-255
        
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        else:
            seq = seq + [0] * (MAX_LEN - len(seq))
            
        inputs_list.append(seq)
        labels_list.append(label)
        
        if idx % 10000 == 0:
            print(f"Processed {idx} retrieval examples...")
            
    return np.array(inputs_list, dtype=np.int32), np.array(labels_list, dtype=np.int64)

def preprocess_pathfinder(dataset):
    import numpy as np
    MAX_LEN = 1024 # 32x32
    
    inputs_list = []
    labels_list = []
    
    for idx, ex in enumerate(dataset):
        img = ex.get('image', ex.get('inputs', tf.constant([]))).numpy()
        label = ex.get('label', tf.constant(0)).numpy()
        
        # Flatten image
        seq = img.flatten().tolist()
        
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        else:
            seq = seq + [0] * (MAX_LEN - len(seq))
            
        inputs_list.append(seq)
        labels_list.append(label)
        
        if idx % 10000 == 0:
            print(f"Processed {idx} pathfinder examples...")
            
    return np.array(inputs_list, dtype=np.int32), np.array(labels_list, dtype=np.int64)

def main():
    data_dir = Path("data/lra")
    tasks = ["listops", "retrieval", "pathfinder"]

    if check_idempotency(data_dir, tasks):
        print("dataset already prepared")
        return

    repo_dir = Path("long-range-arena")
    if not repo_dir.exists():
        print("Cloning LRA repository...")
        subprocess.run(["git", "clone", "https://github.com/google-research/long-range-arena.git", str(repo_dir)], check=True)

    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError:
        print("TensorFlow/TFDS not found. Install with: pip install tensorflow tensorflow-datasets")
        sys.exit(1)

    # Use TFDS to load or generate datasets
    # We will try loading existing tfds datasets first
    datasets_fns = {
        "listops": ("lra_listops", preprocess_listops),
        "retrieval": ("lra_retrieval", preprocess_retrieval),
        "pathfinder": ("lra_pathfinder", preprocess_pathfinder)
    }

    # First attempt to run the specified TFDS command to ensure generation runs
    print("Running TFDS dataset preparation within LRA...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir.absolute()}:{env.get('PYTHONPATH', '')}"
    
    import numpy as np
    import csv

    # Natively inject LRA repository to PYTHONPATH for the current script
    sys.path.insert(0, str(repo_dir.absolute()))
    
    # Try loading from the repo to register the builders
    try:
        # Ignore import errors for lra_benchmarks as the TFDS builders are mostly broken/missing 
        # in the official repo (e.g. retrieval doesn't exist, pathfinder has undefined globals).
        pass
    except ImportError as e:
        print(f"Note on importing LRA modules: {e}")

    for task_name, (tfds_name, process_fn) in datasets_fns.items():
        if (data_dir / task_name / 'train_inputs.npy').exists():
            print(f"{task_name} already generated, skipping.")
            continue
            
        print(f"Processing {task_name}...")
        
        # Retrieval (AAN) and Pathfinder are no longer available in the official GCP buckets (403 Forbidden)
        # and their TFDS builders are missing or broken in the repository.
        if task_name in ['retrieval', 'pathfinder']:
            print(f"Warning: The official dataset for '{task_name}' is no longer publicly accessible or its TFDS builder is broken.")
            print("Skipping to avoid crashing. If you have the raw data, you can build it manually.\n")
            continue

        if task_name == "listops":
            try:
                # 1. Generate TSV files directly using the LRA raw data generator
                print("  Generating raw listops TSVs via lra_benchmarks script (this will take a minute)...")
                tsv_dir = data_dir / "listops" / "raw_tsv"
                tsv_dir.mkdir(parents=True, exist_ok=True)
                
                # We call the python script natively
                subprocess.run(
                    [sys.executable, "-m", "lra_benchmarks.data.listops",
                     f"--output_dir={tsv_dir.absolute()}",
                     "--num_train_samples=96000",
                     "--num_valid_samples=2000",
                     "--num_test_samples=2000"],
                    cwd=str(repo_dir.absolute()),
                    env=env,
                    check=True
                )
                
                # 2. Process TSV into numpy arrays (char-level tokenization)
                splits = {
                    "train": tsv_dir / "basic_train.tsv",
                    "validation": tsv_dir / "basic_val.tsv",
                    "test": tsv_dir / "basic_test.tsv"
                }
                
                for split_name, tsv_file in splits.items():
                    print(f"  Tokenizing split: {split_name} from {tsv_file}...")
                    inputs_list = []
                    labels_list = []
                    MAX_LEN = 2000
                    
                    with open(tsv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f, delimiter='\t')
                        for idx, row in enumerate(reader):
                            src = row['Source']
                            label = int(row['Target'])
                            
                            chars = [ord(c) for c in src if c != ' ']
                            if len(chars) > MAX_LEN:
                                chars = chars[:MAX_LEN]
                            else:
                                chars = chars + [0] * (MAX_LEN - len(chars))
                                
                            inputs_list.append(chars)
                            labels_list.append(label)
                            
                    inputs_np = np.array(inputs_list, dtype=np.int32)
                    labels_np = np.array(labels_list, dtype=np.int64)
                    
                    np.save(data_dir / task_name / f"{split_name}_inputs.npy", inputs_np)
                    np.save(data_dir / task_name / f"{split_name}_labels.npy", labels_np)
                    
                print(f"Successfully generated {task_name}!\n")

            except Exception as e:
                print(f"Failed to generate {task_name} natively: {e}")

if __name__ == "__main__":
    main()
