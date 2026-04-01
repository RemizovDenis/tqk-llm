# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""tqk/extractor.py — Extract KV cache from HuggingFace models."""

from __future__ import annotations

import importlib.util
import warnings
from typing import Any

import torch
import torch.nn as nn

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class KVExtractor:
    """Extractor for KV-cache from HuggingFace CausalLM models."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu") -> None:
        """
        Initialize KVExtractor.

        Args:
            model: The HuggingFace model.
            tokenizer: The associated tokenizer.
            device: Device to run the extraction on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def extract(self, text: str, max_length: int = 2048) -> dict[str, torch.Tensor]:
        """
        Extract KV cache for a given text.

        Args:
            text: Input string to process.
            max_length: Maximum sequence length.

        Returns:
            Dictionary mapped as {"layer_N_keys": Tensor, "layer_N_values": Tensor}.
        """
        if not HAS_TRANSFORMERS:
            warnings.warn(
                "Transformers package is missing. Extraction may not behave as expected.",
                stacklevel=2,
            )

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True, output_attentions=False)

        if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
            warnings.warn(
                f"Model of type {type(self.model)} did not return past_key_values.", stacklevel=2
            )
            return {}

        kv_cache: dict[str, torch.Tensor] = {}
        # HF past_key_values is typically a tuple of tuples: ((k_layer0, v_layer0), (k_layer1, v_layer1), ...)
        for i, (k, v) in enumerate(outputs.past_key_values):
            kv_cache[f"layer_{i}_keys"] = k
            kv_cache[f"layer_{i}_values"] = v

        return kv_cache

    def extract_batch(
        self, texts: list[str], max_length: int = 2048
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract KV cache for a batch of strings.

        Args:
            texts: List of input strings.
            max_length: Maximum sequence length.

        Returns:
            List of KV cache dictionaries.
        """
        return [self.extract(t, max_length=max_length) for t in texts]

    @staticmethod
    def model_info(model: nn.Module) -> dict[str, Any]:
        """
        Extract architecture details from a HuggingFace model.

        Args:
            model: The model instance.

        Returns:
            Dictionary containing model_type, num_layers, num_heads, head_dim.
        """
        config = getattr(model, "config", None)
        if config is None:
            return {}

        # Handle various config attribute names used in HF
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_heads = getattr(
            config, "num_attention_heads", getattr(config, "n_head", 0)
        )
        hidden_size = getattr(
            config, "hidden_size", getattr(config, "n_embd", 0)
        )
        head_dim = (
            getattr(config, "head_dim", hidden_size // num_heads)
            if num_heads > 0
            else 0
        )

        return {
            "model_type": getattr(config, "model_type", "unknown"),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
