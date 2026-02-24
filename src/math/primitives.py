"""
Compatibility shim exposing the primitives implemented in src.maths.
"""
from ..maths.primitives import (
    inner_product_score,
    sherman_morrison_update,
    multiple_rank1_updates,
    memory_update,
    recover_alpha,
)

__all__ = [
    "inner_product_score",
    "sherman_morrison_update",
    "multiple_rank1_updates",
    "memory_update",
    "recover_alpha",
]
