import torch

from src.models.attention.vla_v3 import VLAv3


def test_vla_v3_forward_shape_and_finite():
    torch.manual_seed(0)
    model = VLAv3(d_model=32)
    x = torch.randn(2, 12, 32)
    y = model(x)
    assert y.shape == (2, 12, 32)
    assert torch.isfinite(y).all()
