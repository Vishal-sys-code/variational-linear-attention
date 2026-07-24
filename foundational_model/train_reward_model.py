import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import LRATransformerBlock
from foundational_model.config import VLAConfig

class VLARewardModel(nn.Module):
    """
    Reward Model built on top of the VLA architecture.
    Instead of predicting the next token, it outputs a single scalar reward
    evaluating the quality/reasoning of the entire sequence.
    """
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            LRATransformerBlock(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                dropout=config.dropout,
                attention_type=config.attention_type,
                vla_lambda_0=config.vla_lambda_0,
                vla_penalty_rank=config.vla_penalty_rank,
                vla_gamma=config.vla_gamma,
                vla_fixed_lambda=config.vla_fixed_lambda,
                vla_enable_stabilization=config.vla_enable_stabilization
            )
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Reward Head: projects final hidden state to a single scalar
        self.score_head = nn.Linear(config.d_model, 1, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        
        # Calculate score for every token (B, T, 1) -> squeeze to (B, T)
        scores = self.score_head(x).squeeze(-1)
        
        # We generally take the score of the LAST token in the sequence as the holistic reward
        final_scores = scores[:, -1]
        return final_scores, scores

def create_preference_batch(batch_size=2, seq_len=128):
    """
    Mock data generation for Reward Modeling.
    In production, use Anthropic/hh-rlhf or a reasoning preference dataset.
    You get a 'chosen' response and a 'rejected' response for the same prompt.
    """
    chosen = torch.randint(0, 32000, (batch_size, seq_len))
    rejected = torch.randint(0, 32000, (batch_size, seq_len))
    return chosen, rejected

def train_reward_model():
    """
    Phase 1 of RLHF: Train the Reward Model to distinguish good vs bad reasoning.
    """
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing VLA Reward Model on {device}...")
    reward_model = VLARewardModel(config).to(device)
    
    # Initialize from the SFT model weights (so it understands language & format)
    sft_checkpoint = "checkpoints/vla_sft_12M.pth"
    if os.path.exists(sft_checkpoint):
        # We load strictly the shared layers, ignoring lm_head/score_head mismatch
        state_dict = torch.load(sft_checkpoint)
        # Filter out the lm_head weights from the SFT checkpoint
        state_dict = {k: v for k, v in state_dict.items() if 'lm_head' not in k}
        reward_model.load_state_dict(state_dict, strict=False)
        print("Loaded SFT backbone for Reward Model!")
    else:
        print("Warning: No SFT checkpoint found, starting RM from scratch.")

    optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    batch_size = 1
    gradient_accumulation_steps = 4
    
    reward_model.train()
    print("Starting Reward Model Training Loop...")
    
    for step in range(50):
        optimizer.zero_grad()
        
        # Accumulate gradients
        total_loss = 0
        total_acc = 0
        for micro_step in range(gradient_accumulation_steps):
            chosen, rejected = create_preference_batch(batch_size=batch_size, seq_len=128)
            chosen, rejected = chosen.to(device), rejected.to(device)
            
            with torch.amp.autocast('cuda'):
                chosen_rewards, _ = reward_model(chosen)     # (B,)
                rejected_rewards, _ = reward_model(rejected) # (B,)
                
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_acc += (chosen_rewards > rejected_rewards).float().mean().item()
            
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            acc = total_acc / gradient_accumulation_steps
            print(f"RM Step {step} | Loss: {total_loss:.4f} | RM Accuracy: {acc*100:.1f}%")

    print("Reward Model training complete! Saving checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(reward_model.state_dict(), "checkpoints/vla_reward_model_12M.pth")

if __name__ == "__main__":
    train_reward_model()
