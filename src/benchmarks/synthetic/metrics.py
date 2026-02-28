import torch
import time

class PerformanceLogger:
    """Logs wall-clock timing and GPU memory."""
    def __init__(self):
        self.start_time = None
        
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - self.start_time

    def get_memory_stats(self):
        stats = {}
        if torch.cuda.is_available():
            stats['allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats['reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
            stats['max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            # Reset peak memory stats for next iteration
            torch.cuda.reset_peak_memory_stats()
        return stats


def compute_survival_matrix(alphas: torch.Tensor, q_t: torch.Tensor, v_list: torch.Tensor):
    """
    Computes survival metric: survival_{i->t} = ||v_i (\alpha_i^T q_t)||
    Actually, survival relates to how much memory i influences output t.
    Output t component from memory i is: v_i * (alpha_i^T q_t)
    So survival = ||v_i * (alpha_i^T q_t)|| = |alpha_i^T q_t| * ||v_i||
    
    Args:
        alphas: (T, B, d) tensor of alphas
        q_t: (B, d) query at time t
        v_list: (T, B, d) tensor of values
        
    Returns:
        survival: (T, B) tensor where item i is survival_i->t
    """
    # alpha_i^T q_t -> dot product
    # alphas is (T, B, d), q_t is (B, d)
    # we want dot(alpha_i, q_t) for each i
    # (T, B, d) * (1, B, d) -> sum(dim=-1) -> (T, B)
    dots = (alphas * q_t.unsqueeze(0)).sum(dim=-1) # (T, B)
    
    # ||v_i|| 
    v_norms = torch.norm(v_list, dim=-1) # (T, B)
    
    survival = torch.abs(dots) * v_norms # (T, B)
    return survival

