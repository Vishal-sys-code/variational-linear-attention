import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from foundational_model.config import VLAConfig
from foundational_model.vla_llm import VLACausalLM
from foundational_model.train_reward_model import VLARewardModel

def compute_ppo_loss(logprobs, ref_logprobs, advantages, eps_clip=0.2):
    """
    Computes the Proximal Policy Optimization clipped surrogate objective.
    """
    ratio = torch.exp(logprobs - ref_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    # We want to maximize this objective, so we return the negative for gradient descent
    return -torch.min(surr1, surr2).mean()

def train_rlhf_ppo():
    """
    Phase 2 of RLHF: Proximal Policy Optimization (PPO).
    Uses the trained Reward Model to align the Policy Model.
    Requires 4 models in memory: Policy, Reference, Reward, and Value.
    """
    config = VLAConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing 4-Model PPO RLHF on {device}...")
    
    # 1. Active Policy Model (Learnable)
    policy_model = VLACausalLM(config).to(device)
    
    # 2. Frozen Reference Model (Prevents policy from exploiting the reward model and outputting gibberish)
    ref_model = VLACausalLM(config).to(device)
    ref_model.eval()
    
    # 3. Frozen Reward Model (Provides the scalar reward score for the generated text)
    reward_model = VLARewardModel(config).to(device)
    reward_model.eval()
    
    # 4. Active Value Model (Learnable Baseline to calculate Advantages)
    value_model = VLARewardModel(config).to(device)
    
    # Load checkpoints (Simulated here. In reality, you load the respective SFT and RM checkpoints)
    sft_ckpt = "checkpoints/vla_sft_12M.pth"
    rm_ckpt = "checkpoints/vla_reward_model_12M.pth"
    
    if os.path.exists(sft_ckpt):
        policy_model.load_state_dict(torch.load(sft_ckpt))
        ref_model.load_state_dict(torch.load(sft_ckpt))
        print("Loaded SFT checkpoints into Policy and Reference models.")
        
    if os.path.exists(rm_ckpt):
        reward_model.load_state_dict(torch.load(rm_ckpt))
        # Value model is often initialized from the Reward model
        value_model.load_state_dict(torch.load(rm_ckpt))
        print("Loaded RM checkpoints into Reward and Value models.")
        
    # Optimizers
    opt_policy = optim.AdamW(policy_model.parameters(), lr=1e-6)
    opt_value = optim.AdamW(value_model.parameters(), lr=5e-6)
    
    print("Starting PPO Loop...")
    for step in range(50):
        # 1. Sample a prompt
        prompts = torch.randint(0, 32000, (1, 32)).to(device)
        
        # 2. Generate responses from Policy Model
        # (In production, use policy_model.generate)
        completions = torch.randint(0, 32000, (1, 32)).to(device)
        full_seqs = torch.cat([prompts, completions], dim=1)
        
        # 3. Calculate Rewards using the frozen Reward Model
        with torch.no_grad():
            rewards, _ = reward_model(full_seqs) # (B,)
            
        # 4. Calculate Baselines using the Value Model
        values, _ = value_model(full_seqs) # (B,)
        
        # 5. Calculate Advantages (Reward - Baseline)
        advantages = (rewards - values).detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 6. PPO Update (Multiple epochs over the same batch in standard PPO)
        for _ in range(2): # PPO epochs
            # Get current logprobs
            logits, _ = policy_model(full_seqs)
            logprobs = F.log_softmax(logits, dim=-1).mean(dim=[1, 2])
            
            # Get reference logprobs
            with torch.no_grad():
                ref_logits, _ = ref_model(full_seqs)
                ref_logprobs = F.log_softmax(ref_logits, dim=-1).mean(dim=[1, 2])
            
            # Policy Loss
            pg_loss = compute_ppo_loss(logprobs, ref_logprobs, advantages)
            
            # Value Loss (MSE between predicted values and actual rewards)
            v_preds, _ = value_model(full_seqs)
            v_loss = F.mse_loss(v_preds, rewards)
            
            # Step Policy
            opt_policy.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            opt_policy.step()
            
            # Step Value
            opt_value.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            opt_value.step()
            
        if step % 1 == 0:
            print(f"PPO Step {step} | Policy Loss: {pg_loss.item():.4f} | Value Loss: {v_loss.item():.4f} | Avg Reward: {rewards.mean().item():.4f}")

    print("RLHF PPO complete! Saving final aligned model...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy_model.state_dict(), "checkpoints/vla_final_rlhf_ppo_12M.pth")

if __name__ == "__main__":
    train_rlhf_ppo()
