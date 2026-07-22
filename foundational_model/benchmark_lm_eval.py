import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval import evaluator
except ImportError:
    print("Please install lm-eval: pip install lm-eval")
    LM = object

# Register the custom VLA model with the LM Evaluation Harness
@register_model("vla_foundational")
class VLAEvalWrapper(LM):
    """
    Wrapper for Variational Linear Attention model to interface with EleutherAI LM Evaluation Harness.
    This enables automatic benchmarking on MMLU, GSM8K, HellaSwag, etc.
    """
    def __init__(self, checkpoint_path=None, device="cuda", **kwargs):
        super().__init__()
        self._device = device if torch.cuda.is_available() else "cpu"
        
        config = VLAConfig()
        self.model = VLACausalLM(config).to(self._device)
        self.model.eval()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
            print(f"Loaded checkpoint from {checkpoint_path}")
            
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def loglikelihood(self, requests):
        """
        Calculates log-likelihood of a continuation given a context.
        Used for multiple-choice tasks (like HellaSwag, MMLU).
        """
        res = []
        with torch.no_grad():
            for context, continuation in [req.args for req in requests]:
                ctx_enc = self.tokenizer.encode(context, add_special_tokens=False)
                cont_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
                
                inp = torch.tensor([ctx_enc + cont_enc], dtype=torch.long).to(self._device)
                
                # Forward pass
                logits, _ = self.model(inp)
                
                # Extract logits corresponding to the continuation
                cont_logits = logits[0, len(ctx_enc)-1 : len(ctx_enc)-1 + len(cont_enc), :]
                cont_logprobs = torch.nn.functional.log_softmax(cont_logits, dim=-1)
                
                # Gather logprobs of actual continuation tokens
                cont_enc_tensor = torch.tensor(cont_enc, dtype=torch.long).to(self._device).unsqueeze(1)
                token_logprobs = torch.gather(cont_logprobs, 1, cont_enc_tensor).squeeze(-1)
                
                is_greedy = (cont_logits.argmax(dim=-1) == cont_enc_tensor.squeeze(1)).all().item()
                
                res.append((token_logprobs.sum().item(), is_greedy))
        return res

    def generate_until(self, requests):
        """
        Generates text until a stopping criteria is met.
        Used for generative tasks (like GSM8k).
        """
        res = []
        with torch.no_grad():
            for req in requests:
                context = req.args[0]
                until = req.args[1].get('until', ['\n'])
                max_gen_toks = req.args[1].get('max_gen_toks', 256)
                
                ctx_enc = torch.tensor([self.tokenizer.encode(context)], dtype=torch.long).to(self._device)
                
                # Use our model's generation function
                out_tokens = self.model.generate(ctx_enc, max_new_tokens=max_gen_toks)
                
                # Extract generated portion
                gen_tokens = out_tokens[0, ctx_enc.shape[1]:]
                text = self.tokenizer.decode(gen_tokens)
                
                # Simple string truncation for stopping criteria
                for stop_str in until:
                    if stop_str in text:
                        text = text[:text.index(stop_str)]
                
                res.append(text)
        return res

if __name__ == "__main__":
    print("VLA Eval Wrapper initialized.")
    # Example usage (uncomment when running with real data and weights):
    # import lm_eval
    # results = lm_eval.simple_evaluate(
    #     model="vla_foundational",
    #     model_args="checkpoint_path=checkpoints/vla_final_grpo_12M.pth",
    #     tasks=["hellaswag", "piqa"],
    #     limit=10,
    # )
    # print(results)
