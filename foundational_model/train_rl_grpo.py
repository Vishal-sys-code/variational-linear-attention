import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM

def compute_grpo_loss(policy_model, ref_model, prompts, completions, advantages, beta=0.04):
    """
    Computes Group Relative Policy Optimization (GRPO) loss.
    Instead of an external critic network, GRPO calculates the advantage by taking the relative 
    reward within a group of multiple generated completions for the same prompt.
    """
    # Dummy forward passes for illustration
    # In reality, you gather log-probs of the completions under both policy and ref models.
    policy_logits, _ = policy_model(completions)
    with torch.no_grad():
        ref_logits, _ = ref_model(completions)
        
    # Calculate log probabilities of the actual completion tokens
    # (Simplified representation)
    policy_logprobs = F.log_softmax(policy_logits, dim=-1).mean() 
    ref_logprobs = F.log_softmax(ref_logits, dim=-1).mean()
    
    # GRPO Objective: maximize advantage while staying close to ref model (KL penalty)
    ratio = torch.exp(policy_logprobs - ref_logprobs.detach())
    
    # Proximal clipping (standard in PPO/GRPO)
    epsilon = 0.2
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    
    # Calculate KL divergence penalty analytically or via logprobs
    kl_div = ref_logprobs - policy_logprobs
    
    # Final loss (negative because we maximize)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    total_loss = pg_loss + beta * kl_div
    return total_loss.mean()

def train_rl_grpo():
    """
    Reinforcement Learning Phase via GRPO (DeepSeek's technique).
    Highly memory efficient as it requires no separate Critic/Reward model.
    """
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Initializing GRPO Alignment on {device}...")
    
    # 1. Load active Policy Model (trainable)
    policy_model = VLACausalLM(config).to(device)
    
    # 2. Load frozen Reference Model (for KL penalty)
    ref_model = VLACausalLM(config).to(device)
    ref_model.eval()
    
    sft_checkpoint = "checkpoints/vla_sft_12M.pth"
    if os.path.exists(sft_checkpoint):
        policy_model.load_state_dict(torch.load(sft_checkpoint))
        ref_model.load_state_dict(torch.load(sft_checkpoint))
        print("Loaded SFT checkpoint for RL!")
        
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-6)
    
    print("Starting GRPO Loop...")
    for step in range(50):
        # 1. Sample prompt (e.g. math or reasoning question)
        # 2. Generate G multiple completions (e.g., G=4) from policy_model
        # 3. Score them using a rule-based system (e.g., did it reach the correct final answer?)
        # 4. Calculate Group Advantages: A_i = (Reward_i - mean(Rewards)) / std(Rewards)
        
        # Simulating dummy advantages and completions for pipeline structural testing:
        prompts = torch.randint(0, 32000, (4, 64)).to(device)
        completions = torch.randint(0, 32000, (4, 128)).to(device)
        advantages = torch.tensor([1.5, 0.5, -0.5, -1.5]).to(device) # Normalized relative rewards
        
        loss = compute_grpo_loss(policy_model, ref_model, prompts, completions, advantages)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"GRPO Step {step} | Policy Loss: {loss.item():.4f}")

    print("GRPO alignment complete! Saving final foundational model...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy_model.state_dict(), "checkpoints/vla_final_grpo_12M.pth")

if __name__ == "__main__":
    train_rl_grpo()
