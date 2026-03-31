"""tqk/format.py — TQK1 binary format implementation."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

MAGIC = b"TQK1"
VERSION = 1


@dataclass
class TQKMetadata:
    """Metadata for the TQK memory format."""

    source_model: str
    created_at: float = field(default_factory=time.time)
    text_hash: str = ""
    compression_ratio: float = 0.0
    num_layers: int = 0
    head_dim: int = 0
    num_heads: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class TQKFile:
    """Binary format for storing compressed KV cache in TQK format."""

    def __init__(self, tensors: dict[str, torch.Tensor], metadata: TQKMetadata) -> None:
        """
        Initialize TQKFile.

        Args:
            tensors: Dictionary of tensors to store.
            metadata: Metadata associated with the tensors.
        """
        self.tensors = tensors
        self.metadata = metadata

    @classmethod
    def from_cache_entry(cls, entry: Any, metadata: TQKMetadata) -> TQKFile:
        """
        Create TQKFile from a TurboQuant CacheEntry.

        Args:
            entry: Object with compressed_keys and compressed_values.
            metadata: Metadata for the file.

        Returns:
            TQKFile instance.
        """
        tensors: dict[str, torch.Tensor] = {}

        # Safely extract tensors from CacheEntry or similar objects
        if hasattr(entry, "compressed_keys") and entry.compressed_keys is not None:
            tensors["keys_packed"] = entry.compressed_keys.packed
            tensors["keys_scales"] = entry.compressed_keys.scales
        if hasattr(entry, "compressed_values") and entry.compressed_values is not None:
            tensors["values_packed"] = entry.compressed_values.packed
            tensors["values_scales"] = entry.compressed_values.scales

        # Optional residuals
        if hasattr(entry, "residual_keys") and entry.residual_keys is not None:
            tensors["residual_keys"] = entry.residual_keys
        if hasattr(entry, "residual_values") and entry.residual_values is not None:
            tensors["residual_values"] = entry.residual_values

        return cls(tensors, metadata)

    def save(self, path: str | Path) -> None:
        """
        Save the TQKFile to a binary file.

        Args:
            path: Path to the destination file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata_json = json.dumps(self.metadata.__dict__).encode("utf-8")
        metadata_len = len(metadata_json)

        # Generate safetensors blob
        payload = safetensors.torch.save(self.tensors)

        with open(path, "wb") as f:
            # Header: magic (4), version (4), metadata_len (4)
            f.write(MAGIC)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<I", metadata_len))
            f.write(metadata_json)
            f.write(payload)

    @classmethod
    def load(cls, path: str | Path) -> TQKFile:
        """
        Load a TQKFile from disk.

        Args:
            path: Path to the .tqk file.

        Returns:
            TQKFile instance.

        Raises:
            ValueError: If magic bytes don't match or version is unsupported.
        """
        path = Path(path)
        with open(path, "rb") as f:
            # Magic check
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic bytes: expected {MAGIC!r}, got {magic!r}")

            # Version check
            version_bytes = f.read(4)
            if not version_bytes:
                raise ValueError("Unexpected EOF while reading version")
            version = struct.unpack("<I", version_bytes)[0]
            if version != VERSION:
                raise ValueError(f"Unsupported TQK version: {version}")

            # Metadata length
            metalen_bytes = f.read(4)
            if not metalen_bytes:
                raise ValueError("Unexpected EOF while reading metadata length")
            metadata_len = struct.unpack("<I", metalen_bytes)[0]

            # Metadata JSON
            metadata_json = f.read(metadata_len)
            if len(metadata_json) != metadata_len:
                raise ValueError("Unexpected EOF while reading metadata")

            metadata_dict = json.loads(metadata_json.decode("utf-8"))
            metadata = TQKMetadata(**metadata_dict)

            # Rest is safetensors payload
            payload = f.read()

        tensors = safetensors.torch.load(payload)
        return cls(tensors, metadata)

    def to_cache_entry(self) -> dict[str, torch.Tensor]:
        """
        Return the raw tensor dictionary for use without turboquant.

        Returns:
            Dictionary of stored tensors.
        """
        return self.tensors

    def compression_ratio(self) -> float:
        """
        Return the compression ratio from metadata.

        Returns:
            The stored compression ratio.
        """
        return self.metadata.compression_ratio

    def info(self) -> dict[str, Any]:
        """
        Return a human-readable dictionary of file information.

        Returns:
            Detailed information about the TQK file.
        """
        # Calculate total size in MB (approximate from tensors)
        total_bytes = sum(t.nbytes for t in self.tensors.values())
        total_size_mb = total_bytes / (1024 * 1024)

        return {
            "source_model": self.metadata.source_model,
            "created_at": time.ctime(self.metadata.created_at),
            "num_tensors": len(self.tensors),
            "total_size_mb": round(total_size_mb, 2),
            "compression_ratio": round(self.metadata.compression_ratio, 2),
            "num_layers": self.metadata.num_layers,
            "head_dim": self.metadata.head_dim,
            "num_heads": self.metadata.num_heads,
        }

    def __repr__(self) -> str:
        """Return a single-line summary of the TQKFile."""
        info = self.info()
        return f"TQKFile(model={info['source_model']}, size={info['total_size_mb']}MB)"
