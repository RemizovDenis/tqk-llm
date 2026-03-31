# tqk

Portable memory format for large language models.

A .tqk file stores a compressed KV cache that can be transferred
between different model architectures. One model reads a document,
saves its understanding to a .tqk file, and another model loads it
instantly — without re-reading the original text.

Built on TurboQuant-MoE: github.com/RemizovDenis/turboquant


## What this solves

Every time you switch between models you lose context.
Read 500 pages with one model, switch to another, start over.
tqk eliminates that.


## Status

Early development. Format specification in progress.
First projector weights (llama3.2-3b to mistral-7b) coming soon.


## Installation

pip install tqk


## Quick start

from tqk import TQKFile

file = TQKFile.from_text("your document here", model, tokenizer)
file.save("memory.tqk")

loaded = TQKFile.load("memory.tqk")


## Built on

TurboQuant-MoE — github.com/RemizovDenis/turboquant


## License

Apache 2.0
