# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.

"""Local TQK benchmark utilities."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tqk.format import TQKFile, TQKMetadata
from tqk.validator import TQKValidator


@dataclass
class BenchmarkConfig:
    """Benchmark dimensions for synthetic KV tensors."""

    num_layers: int = 8
    num_heads: int = 4
    seq_len: int = 512
    head_dim: int = 64
    dtype: str = "float16"


def _torch_dtype(name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _make_kv(config: BenchmarkConfig) -> dict[str, torch.Tensor]:
    dtype = _torch_dtype(config.dtype)
    kv: dict[str, torch.Tensor] = {}
    for layer in range(config.num_layers):
        shape = (config.num_heads, config.seq_len, config.head_dim)
        kv[f"layer_{layer}_keys"] = torch.randn(shape, dtype=dtype)
        kv[f"layer_{layer}_values"] = torch.randn(shape, dtype=dtype)
    return kv


def run_roundtrip_benchmark(config: BenchmarkConfig, out_path: Path) -> dict[str, Any]:
    """Run a local save/load + quality benchmark and return metrics."""
    kv = _make_kv(config)
    raw_bytes = int(sum(int(t.nbytes) for t in kv.values()))

    metadata = TQKMetadata(
        source_model="benchmark-local",
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
    )

    tqk = TQKFile(kv, metadata)
    t0 = time.perf_counter()
    tqk.save(out_path)
    save_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    loaded = TQKFile.load(out_path)
    load_ms = (time.perf_counter() - t1) * 1000.0

    restored = loaded.to_cache_entry()
    validator = TQKValidator(threshold=0.99)
    quality = validator.validate(kv, restored)

    file_bytes = int(out_path.stat().st_size)
    ratio = raw_bytes / max(1, file_bytes)

    return {
        "config": asdict(config),
        "raw_bytes": raw_bytes,
        "file_bytes": file_bytes,
        "compression_ratio_x": ratio,
        "save_ms": save_ms,
        "load_ms": load_ms,
        "cosine_similarity": float(quality.cosine_similarity),
        "mse": float(quality.mse),
        "passed_threshold": bool(quality.passed),
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render benchmark result as Markdown table."""
    cfg = result.get("config", {})
    lines = [
        "# TQK Local Benchmark",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Layers | {cfg.get('num_layers', 'n/a')} |",
        f"| Heads | {cfg.get('num_heads', 'n/a')} |",
        f"| Seq len | {cfg.get('seq_len', 'n/a')} |",
        f"| Head dim | {cfg.get('head_dim', 'n/a')} |",
        f"| Dtype | {cfg.get('dtype', 'n/a')} |",
        f"| Raw bytes | {result.get('raw_bytes', 0)} |",
        f"| File bytes | {result.get('file_bytes', 0)} |",
        f"| Compression ratio | {float(result.get('compression_ratio_x', 0.0)):.3f}x |",
        f"| Save latency | {float(result.get('save_ms', 0.0)):.2f} ms |",
        f"| Load latency | {float(result.get('load_ms', 0.0)):.2f} ms |",
        f"| Cosine similarity | {float(result.get('cosine_similarity', 0.0)):.6f} |",
        f"| MSE | {float(result.get('mse', 0.0)):.8f} |",
        f"| Quality pass | {result.get('passed_threshold', False)} |",
    ]
    return "\n".join(lines) + "\n"


def run_and_write(
    *,
    output_json: Path,
    output_md: Path,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run benchmark and write JSON + Markdown outputs."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    temp_tqk = output_json.with_suffix(".tmp.tqk")
    result = run_roundtrip_benchmark(config, temp_tqk)

    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")

    if temp_tqk.exists():
        temp_tqk.unlink()

    return result
