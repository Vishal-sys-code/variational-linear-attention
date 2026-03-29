import os
import json
import torch
import torch.nn as nn
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.attention.vla import VLALayer

class VLASmokeModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Simple token embedding generator/projection if needed,
        # but the spec asks for "predict next token embedding using MSE loss",
        # so our input is directly continuous embeddings, and output matches.
        self.vla = VLALayer(d_model=d_model, enable_stabilization=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, return_states=False):
        O, states = self.vla(x, return_states=return_states)
        # O shape: (B, T, d_model)
        preds = self.proj(O)
        if return_states:
            return preds, states
        return preds


def main():
    # configuration
    config = {
        "sequence_length": 8,
        "embedding_dimension": 32,
        "batch_size": 2,
        "random_seed": 42,
        "iterations": 3,
        "learning_rate": 1e-3
    }

    # Set random seed
    torch.manual_seed(config["random_seed"])

    # Initialize model
    model = VLASmokeModel(d_model=config["embedding_dimension"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    norm_A_t_max = 0.0
    norm_S_t_max = 0.0
    final_loss = 0.0

    print("Starting smoke training...")
    for step in range(config["iterations"]):
        # Generate random continuous embedding sequence (Task: predict next token embedding)
        # Input shape: (B, T+1, d_model) to get input X and target Y
        raw_seq = torch.randn(config["batch_size"], config["sequence_length"] + 1, config["embedding_dimension"])
        x = raw_seq[:, :-1, :] # (B, T, d_model)
        y = raw_seq[:, 1:, :]  # (B, T, d_model)

        optimizer.zero_grad()
        
        preds, states = model(x, return_states=True)
        
        loss = criterion(preds, y)
        loss.backward()

        # Part 2 Checks: loss condition
        assert torch.isfinite(loss), f"Loss is not finite at step {step}: {loss.item()}"
        assert not torch.isnan(loss), f"Loss is NaN at step {step}"

        # Gradient checks
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Gradient is not finite at step {step}"
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        assert grad_norm < 1e6, f"Gradient norm exploded at step {step}: {grad_norm}"

        optimizer.step()

        # Update final loss
        final_loss = loss.item()

        # Part 3 Checks: Numerical stability logs
        # A matrix is shape (B, T, d_model, d_model)
        # S_norm represents Frobenius norm of memory matrix across batch and seq
        A_t = states["A"]
        S_norm = states["S_norm"]

        max_A = torch.max(torch.norm(A_t, p="fro", dim=(-2, -1))).item()
        max_S = torch.max(S_norm).item()

        norm_A_t_max = max(norm_A_t_max, max_A)
        norm_S_t_max = max(norm_S_t_max, max_S)

        assert norm_A_t_max < 1e6, f"Norm of A_t exploded: {norm_A_t_max}"
        assert norm_S_t_max < 1e6, f"Norm of S_t exploded: {norm_S_t_max}"

        print(f"Step {step}: Loss = {loss.item():.6f}, grad_norm = {grad_norm:.6f}, max_A = {max_A:.6f}, max_S = {max_S:.6f}")

    print("smoke_training passed")
    
    # Part 4 Checks: Artifacts storage
    os.makedirs("ci_artifacts", exist_ok=True)
    
    with open("ci_artifacts/config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    metrics = {
        "final_loss": final_loss,
        "norm_A_t": norm_A_t_max,
        "norm_S_t": norm_S_t_max,
        "grad_norm": grad_norm
    }
    with open("ci_artifacts/final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    torch.save(model.state_dict(), "ci_artifacts/model.pt")
    
    # Write expected exact match outputs for CI grep/parsing
    print(f"final_loss = {final_loss}")
    print(f"norm_A_t = {norm_A_t_max}")
    print(f"norm_S_t = {norm_S_t_max}")
    print("Artifacts saved to ci_artifacts/")


if __name__ == "__main__":
    main()
