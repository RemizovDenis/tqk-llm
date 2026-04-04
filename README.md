# tqk

[![CI](https://github.com/RemizovDenis/tqk-llm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RemizovDenis/tqk-llm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RemizovDenis/tqk-llm)](https://github.com/RemizovDenis/tqk-llm/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-orange.svg)](./LICENSE)

Portable memory format for large language models.

## What it does

`tqk` serializes compressed KV-cache tensors into a portable `.tqk` file so context can be transferred across model sessions and architectures.

Core goals:

- preserve expensive prefill work
- reduce repeated token processing
- standardize memory exchange format

## Installation

Current recommended path is source install:

```bash
git clone https://github.com/RemizovDenis/tqk-llm.git
cd tqk-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,transformers]"
```

Note: the package name is being moved to `tqk-llm` to avoid conflict with an unrelated existing `tqk` package on PyPI.

## Quick start

```python
from tqk.turboquant_bridge import TQKPipeline

pipeline = TQKPipeline(source_model, tokenizer)
pipeline.save_context("Your long technical documentation...", "memory.tqk")

target_model = pipeline.load_context("memory.tqk", target_model, target_tokenizer)
```

## CLI

```bash
tqk info memory.tqk
tqk validate memory.tqk
```

## Benchmarks

Reference benchmark and validation assets:

- [verify_quality.py](./verify_quality.py)
- [experiments/real_benchmark.py](./experiments/real_benchmark.py)
- [projectors/README.md](./projectors/README.md)

## Project standards

- [Contributing](./CONTRIBUTING.md)
- [Security](./SECURITY.md)
- [Code of conduct](./CODE_OF_CONDUCT.md)
- [Support](./SUPPORT.md)

## License

Business Source License 1.1 (BUSL-1.1).
Non-commercial use is free.
Commercial use requires a license agreement.
Converts to Apache-2.0 on 2030-04-01.
