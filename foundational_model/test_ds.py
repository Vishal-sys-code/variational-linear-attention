from datasets import load_dataset
try:
    print("Testing HuggingFaceH4/CodeAlpaca_20K")
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", streaming=True, split="train")
    print(next(iter(ds)))
    print("Success H4")
except Exception as e:
    print("Failed H4:", e)

try:
    print("Testing iamtarun/python_code_instructions_18k_alpaca")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", streaming=True, split="train")
    print(next(iter(ds)))
    print("Success iamtarun")
except Exception as e:
    print("Failed iamtarun:", e)
