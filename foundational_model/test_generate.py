import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM

def test():
    config = VLAConfig()
    model = VLACausalLM(config)
    model.eval()
    
    # Dummy prompt (Batch=1, SeqLen=10)
    prompt = torch.randint(0, 32000, (1, 10))
    print("Prompt shape:", prompt.shape)
    
    # Generate 5 tokens
    output = model.generate(prompt, max_new_tokens=5)
    print("Output shape:", output.shape)
    print("Success!")

if __name__ == "__main__":
    test()
