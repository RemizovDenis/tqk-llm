"""tqk/cli.py — CLI interface for TQK memory format management."""

from __future__ import annotations

import argparse
import sys

from tqk import __version__
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
        _ = TQKFile.load(args.file)
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


def main() -> None:
    """CLI entry point for the tqk package."""
    parser = argparse.ArgumentParser(
        prog="tqk", description="TQK Memory Format CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    info_parser = subparsers.add_parser("info", help="Get info about a .tqk file")
    info_parser.add_argument("file", help="Path to the .tqk file")

    # validate
    val_parser = subparsers.add_parser("validate", help="Check file validity")
    val_parser.add_argument("file", help="Path to the .tqk file")

    # convert
    conv_parser = subparsers.add_parser("convert", help="Convert between models")
    conv_parser.add_argument("input", help="Source .tqk file")
    conv_parser.add_argument("output", help="Target .tqk file")
    conv_parser.add_argument(
        "--target-model", required=True, help="Name of the target architecture"
    )

    # version
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "info":
        sys.exit(info_command(args))
    elif args.command == "validate":
        sys.exit(validate_command(args))
    elif args.command == "convert":
        sys.exit(convert_command(args))
    elif args.command == "version":
        print(__version__)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
