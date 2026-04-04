"""Benchmark module and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tqk.benchmark import BenchmarkConfig, run_and_write
from tqk.cli import main


def test_run_and_write_creates_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "bench" / "result.json"
    out_md = tmp_path / "bench" / "result.md"
    config = BenchmarkConfig(num_layers=2, num_heads=2, seq_len=16, head_dim=8, dtype="float16")

    result = run_and_write(output_json=out_json, output_md=out_md, config=config)
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["config"]["num_layers"] == 2
    assert result["compression_ratio_x"] > 0
    assert "| Compression ratio |" in out_md.read_text(encoding="utf-8")


def test_cli_benchmark_command_runs(tmp_path: Path) -> None:
    out_json = tmp_path / "cli_result.json"
    out_md = tmp_path / "cli_result.md"
    argv = [
        "tqk",
        "benchmark",
        "--layers",
        "1",
        "--heads",
        "1",
        "--seq-len",
        "8",
        "--head-dim",
        "8",
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]

    with patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    assert out_json.exists()
    assert out_md.exists()
