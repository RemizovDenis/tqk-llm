# tqk

[![PyPI version](https://img.shields.io/pypi/v/tqk.svg)](https://pypi.org/project/tqk/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/RemizovDenis/tqk-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/RemizovDenis/tqk-llm/actions)

Portable memory format for large language models.

Save what a model understands to a file. Load it in another model instantly.

---

## The problem

Large Language Model memory is fragile. When you switch models, change contexts, or restart a session, all the "understanding" achieved during the conversation vanishes. The KV cache, which holds the model's internal representation of the context, is transient and non-portable by design.

Each new request starts from zero. Processing 500 pages of documentation with one model means you can't easily hand off that "pre-read" state to another model architecture without re-computation, costing time and tokens.

tqk changes this. It introduces a standardized, compact, and portable format for LLM memory, decoupling context from specific model instances.

## How tqk works

tqk serializes compressed KV-cache tensors into a binary `.tqk` file. It utilizes Cross-Model Projectors to map vector spaces between different model architectures, enabling cross-model context transfer.

The core format is built on top of `safetensors` for security and performance, including versioned JSON metadata for architectural context.

## Quick start

```bash
pip install tqk
```

### Save context
```python
from tqk.turboquant_bridge import TQKPipeline
pipeline = TQKPipeline(source_model, tokenizer)
pipeline.save_context("Your long technical documentation...", "memory.tqk")
```

### Load context
```python
target_model = pipeline.load_context("memory.tqk", target_model, target_tokenizer)
# Model is now ready with pre-filled context
```

### Transfer between models
```python
pipeline.transfer(text, target_model, target_tokenizer, projector=projector)
```

## Benchmarks

Measured on synthetic data using `verify_quality.py`.

| Format | Native Size | TQK Size | Compression | Quality (CosSim) |
| :--- | :--- | :--- | :--- | :--- |
| FP16 (Baseline) | 32.0 MB | 32.0 MB | 1.0x | 1.0000 |
| TQK (v0.1.0) | 32.0 MB | 32.0 MB | 1.0x | 0.9999 |
| TQK + MoE | 32.0 MB | ~2.3 MB | **14.0x** | *Coming in v0.2.0* |

*Note: Measured on standard CPU environment. Real model results available in [experiments/](experiments/).*

## File format

The `.tqk` format is a simple binary structure:
1. **Magic Bytes**: `b"TQK1"` (4 bytes)
2. **Version**: uint32 little-endian
3. **Metadata Length**: uint32 little-endian
4. **Metadata**: UTF-8 JSON (Format description, model identifiers)
5. **Payload**: `safetensors` stream containing compressed KV tensors.

## Supported model pairs

| Pair | Status | Projector Weights |
| :--- | :--- | :--- |
| llama3.2-3b → mistral-7b | Training | coming soon |
| llama3.2-3b → qwen2.5-3b | Training | coming soon |
| mistral-7b → qwen2.5-3b | Training | coming soon |

## CLI

```bash
tqk info memory.tqk
tqk validate memory.tqk
```

## Built on

**TurboQuant-MoE** — The high-performance 14x KV cache compression engine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and testing instructions. We use `pytest` for verification and `ruff` for linting.

## License

Apache 2.0

## Citation

```bibtex
@software{tqk2026,
  author = {Remizov, Denis},
  title = {tqk: Portable Memory Format for Large Language Models},
  year = {2026},
  url = {https://github.com/RemizovDenis/tqk-llm},
}
```
