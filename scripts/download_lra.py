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
        "retrieval": ("lra_retrieval", preprocess_retrieval),  # might be lra_aan or lra_retrieval
        "pathfinder": ("lra_pathfinder", preprocess_pathfinder)
    }

    # First attempt to run the specified TFDS command to ensure generation runs
    print("Running TFDS dataset preparation within LRA...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir.absolute()}:{env.get('PYTHONPATH', '')}"
    
    # We will try to download sequentially
    import numpy as np
    for task_name in datasets_fns.keys():
        os.makedirs(data_dir / task_name, exist_ok=True)

    # Natively inject LRA repository to PYTHONPATH for the current script
    sys.path.insert(0, str(repo_dir.absolute()))
    
    # Try loading from the repo to register the builders
    try:
        import lra_benchmarks.data.listops
        import lra_benchmarks.data.retrieval
        import lra_benchmarks.data.pathfinder
    except ImportError as e:
        print(f"Note on importing LRA modules: {e}")

    for task_name, (tfds_name, process_fn) in datasets_fns.items():
        if (data_dir / task_name / 'train_inputs.npy').exists():
            print(f"{task_name} already generated, skipping.")
            continue
            
        print(f"Processing {task_name}...")
        try:
            # We try standard variants of names for LRA
            # In official LRA, datasets are often registered as e.g. 'lra_listops'
            # If not found, we fallback to downloading using the scripts or manual loading.
            builder = tfds.builder(tfds_name)
            builder.download_and_prepare()
            
            for split in ['train', 'validation', 'test']:
                print(f"  Generating split: {split}")
                ds = builder.as_dataset(split=split)
                inputs, labels = process_fn(ds)
                
                np.save(data_dir / task_name / f"{split}_inputs.npy", inputs)
                np.save(data_dir / task_name / f"{split}_labels.npy", labels)
                
        except Exception as e:
            print(f"Failed to generate {task_name} via TFDS directly: {e}")
            print("To manually fix: make sure LRA TFDS builders are available in PYTHONPATH.")

if __name__ == "__main__":
    main()
