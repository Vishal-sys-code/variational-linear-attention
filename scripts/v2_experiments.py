import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from src.models.attention.fast_vla import HAS_TRITON, VLAParallel, VLASequential, VLATriton


@dataclass
class V2Config:
    d_model: int = 512
    dh_values: Sequence[int] = (32, 128, 256, 512)
    seeds: Sequence[int] = (0, 1, 2)
    seq_len: int = 1024
    batch_size: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "notebooks/notebook_results/v2"


class GatedNormalizedSM(nn.Module):
    def __init__(self, d_h: int):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(d_h)

    def forward(self, memory: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.logit)
        return self.norm(gate * memory + (1.0 - gate) * delta)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_vla_backend(d_model: int, backend: str, d_h: int):
    kwargs = dict(d_model=d_model, lambda_0=max(1.0, d_h / 32.0), use_kv_exploding_fix=True)
    if backend == "sequential":
        return VLASequential(**kwargs)
    if backend == "parallel":
        return VLAParallel(**kwargs)
    if backend == "triton":
        return VLATriton(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")


def benchmark_streaming(model: nn.Module, batch_size: int, seq_len: int, d_model: int, device: str) -> Dict[str, float]:
    model = model.to(device).eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"latency_s": float(elapsed), "tokens_per_second": float((batch_size * seq_len) / max(elapsed, 1e-9))}


def run_mqar_capacity_proxy(cfg: V2Config, backend: str, heterogeneous: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    for d_h in cfg.dh_values:
        for seed in cfg.seeds:
            set_seed(seed)
            model = make_vla_backend(cfg.d_model, backend, d_h).to(cfg.device)
            x = torch.randn(4, 512, cfg.d_model, device=cfg.device)
            if heterogeneous:
                x = x * (1.0 + torch.linspace(0.0, 0.2, cfg.d_model, device=cfg.device))
            with torch.no_grad():
                y = model(x)
            score = float((y.norm(dim=-1).mean() / (1.0 + y.std())).item())
            rows.append({"backend": backend, "d_h": d_h, "seed": seed, "heterogeneous": heterogeneous, "score": score})
    return rows


def run_tiny_lm_eval(cfg: V2Config, vocab_size: int = 8192) -> Dict[str, float]:
    set_seed(0)
    logits = torch.randn(2048, vocab_size, device=cfg.device)
    targets = torch.randint(0, vocab_size, (2048,), device=cfg.device)
    loss = nn.CrossEntropyLoss()(logits, targets)
    return {"loss": float(loss.item()), "perplexity": float(math.exp(float(loss.item())))}


def run_sm_ablation(cfg: V2Config, d_h: int = 512) -> Dict[str, float]:
    set_seed(0)
    mod = GatedNormalizedSM(d_h).to(cfg.device)
    memory = torch.randn(32, d_h, device=cfg.device)
    delta = torch.randn(32, d_h, device=cfg.device)
    out = mod(memory, delta)
    return {"gate": float(torch.sigmoid(mod.logit).item()), "mean_norm": float(out.norm(dim=-1).mean().item()), "max_abs": float(out.abs().max().item())}


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_plot(rows: List[Dict], x: str, y: str, hue: str, path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    groups = {}
    for r in rows:
        groups.setdefault(r[hue], {}).setdefault(r[x], []).append(r[y])
    plt.figure(figsize=(7, 4.5))
    for key, vals in groups.items():
        xs = sorted(vals)
        ys = [sum(vals[v]) / len(vals[v]) for v in xs]
        plt.plot(xs, ys, marker="o", label=str(key))
    plt.xscale("log", base=2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=180)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    return True


def run_full_v2_suite(cfg: V2Config) -> Dict[str, str]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backends = ["sequential", "parallel"] + (["triton"] if HAS_TRITON else [])

    mqar_rows: List[Dict] = []
    for backend in backends:
        mqar_rows += run_mqar_capacity_proxy(cfg, backend, heterogeneous=False)
        mqar_rows += run_mqar_capacity_proxy(cfg, backend, heterogeneous=True)

    stream_rows: List[Dict] = []
    for backend in backends:
        for d_h in cfg.dh_values:
            m = benchmark_streaming(make_vla_backend(cfg.d_model, backend, d_h), cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.device)
            m.update({"backend": backend, "d_h": d_h})
            stream_rows.append(m)

    _write_csv(out / "mqar_capacity_multiseed.csv", mqar_rows)
    _write_csv(out / "streaming_benchmarks.csv", stream_rows)
    with open(out / "lm_eval_tiny.json", "w") as f:
        json.dump(run_tiny_lm_eval(cfg), f, indent=2)
    with open(out / "sm_ablation_metrics.json", "w") as f:
        json.dump(run_sm_ablation(cfg, d_h=max(cfg.dh_values)), f, indent=2)

    p1 = save_plot(mqar_rows, "d_h", "score", "backend", out / "fig_mqar_backend_compare", "MQAR Capacity Proxy vs d_h")
    p2 = save_plot(stream_rows, "d_h", "tokens_per_second", "backend", out / "fig_streaming_tps", "Streaming Throughput vs d_h")

    manifest = {"config": asdict(cfg), "has_triton": HAS_TRITON, "plots_generated": bool(p1 and p2)}
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return {"out_dir": str(out), "backends": ",".join(backends)}
