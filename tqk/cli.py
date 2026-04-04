# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""tqk/cli.py — CLI interface for TQK memory format management."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqk import __version__
from tqk.benchmark import BenchmarkConfig, run_and_write
from tqk.format import TQKFile


def info_command(args: argparse.Namespace) -> int:
    """Print detailed information about a .tqk file."""
    try:
        tqk_file = TQKFile.load(args.file)
        details = tqk_file.info()
        for key, value in details.items():
            print(f"{key}: {value}")
        return 0
    except Exception as e:
        print(f"ERROR: Could not read file: {e}")
        return 1


def validate_command(args: argparse.Namespace) -> int:
    """Validate structure and magic bytes of a .tqk file."""
    try:
        # TQKFile.load already checks magic and version
        _ = TQKFile.load(args.file, verify_integrity=not args.no_integrity)
        print("OK")
        return 0
    except Exception as e:
        print(f"INVALID: {e}")
        return 1


def convert_command(args: argparse.Namespace) -> int:
    """
    Placeholder for conversion command.

    Conversion requires a trained projector.
    See: github.com/RemizovDenis/tqk/projectors
    """
    _ = args  # Mark as used for linters
    print("Conversion requires a trained projector.")
    print("See: github.com/RemizovDenis/tqk/projectors")
    return 1


def benchmark_command(args: argparse.Namespace) -> int:
    """Run local TQK roundtrip benchmark and write JSON/Markdown reports."""
    try:
        config = BenchmarkConfig(
            num_layers=int(args.layers),
            num_heads=int(args.heads),
            seq_len=int(args.seq_len),
            head_dim=int(args.head_dim),
            dtype=str(args.dtype),
        )
        result = run_and_write(
            output_json=Path(args.output_json),
            output_md=Path(args.output_md),
            config=config,
        )
        print(f"compression_ratio_x: {result['compression_ratio_x']:.3f}")
        print(f"save_ms: {result['save_ms']:.2f}")
        print(f"load_ms: {result['load_ms']:.2f}")
        print(f"json: {args.output_json}")
        print(f"md: {args.output_md}")
        return 0
    except Exception as e:
        print(f"ERROR: benchmark failed: {e}")
        return 1


def main() -> None:
    """CLI entry point for the tqk package."""
    parser = argparse.ArgumentParser(prog="tqk", description="TQK Memory Format CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    info_parser = subparsers.add_parser("info", help="Get info about a .tqk file")
    info_parser.add_argument("file", help="Path to the .tqk file")

    # validate
    val_parser = subparsers.add_parser("validate", help="Check file validity")
    val_parser.add_argument("file", help="Path to the .tqk file")
    val_parser.add_argument(
        "--no-integrity",
        action="store_true",
        help="Skip SHA-256 integrity verification for legacy/corrupted payload debugging",
    )

    # convert
    conv_parser = subparsers.add_parser("convert", help="Convert between models")
    conv_parser.add_argument("input", help="Source .tqk file")
    conv_parser.add_argument("output", help="Target .tqk file")
    conv_parser.add_argument(
        "--target-model", required=True, help="Name of the target architecture"
    )

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run local format benchmark")
    bench_parser.add_argument("--layers", type=int, default=8)
    bench_parser.add_argument("--heads", type=int, default=4)
    bench_parser.add_argument("--seq-len", type=int, default=512)
    bench_parser.add_argument("--head-dim", type=int, default=64)
    bench_parser.add_argument("--dtype", type=str, default="float16")
    bench_parser.add_argument("--output-json", type=str, default="tqk_benchmark.json")
    bench_parser.add_argument("--output-md", type=str, default="tqk_benchmark.md")

    # version
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "info":
        sys.exit(info_command(args))
    elif args.command == "validate":
        sys.exit(validate_command(args))
    elif args.command == "convert":
        sys.exit(convert_command(args))
    elif args.command == "benchmark":
        sys.exit(benchmark_command(args))
    elif args.command == "version":
        print(__version__)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
