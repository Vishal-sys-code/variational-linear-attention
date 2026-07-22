from dataclasses import dataclass

@dataclass
class VLAConfig:
    """
    Configuration for the ~12M Parameter VLA Foundational Model.
    Designed for rapid prototyping and training on Kaggle (T4/P100 GPUs).
    """
    vocab_size: int = 119547     # Matches 'bert-base-multilingual-cased' tokenizer vocab size
    d_model: int = 256           # Embedding dimension
    n_layers: int = 8            # Number of VLA layers
    d_ffn: int = 1024            # FeedForward network dimension
    dropout: float = 0.1
    max_len: int = 4096          # Max context length during training (VLA can extrapolate beyond this)
    
    # VLA specific hyper-parameters
    attention_type: str = "vla"
    vla_lambda_0: float = 1.0
    vla_penalty_rank: int = 1
    vla_gamma: float = 0.0
    vla_fixed_lambda: float = None
    vla_enable_stabilization: bool = True
