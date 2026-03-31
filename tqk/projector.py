"""tqk/projector.py — Cross-model KV-cache transfer implementation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog
import safetensors.torch

logger = structlog.get_logger()

REGISTRY: dict[str, str] = {
    "llama3.2-3b->mistral-7b": "projectors/llama3_to_mistral.safetensors",
    "llama3.2-3b->qwen2.5-3b": "projectors/llama3_to_qwen.safetensors",
    "mistral-7b->qwen2.5-3b": "projectors/mistral_to_qwen.safetensors",
}


@dataclass
class ProjectorConfig:
    """Configuration for the CrossModelKVProjector."""

    source_model: str
    target_model: str
    source_dim: int
    target_dim: int
    source_heads: int
    target_heads: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class LinearProjector(nn.Module):
    """Multi-layer MLP for projecting between different vector spaces."""

    def __init__(self, config: ProjectorConfig) -> None:
        """
        Initialize LinearProjector MLP.

        Args:
            config: Projector configuration.
        """
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []

        curr_dim = config.source_dim
        for i in range(config.num_layers):
            next_dim = config.hidden_dim if i < config.num_layers - 1 else config.target_dim
            linear = nn.Linear(curr_dim, next_dim)

            # Xavier uniform initialization
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

            layers.append(linear)
            if i < config.num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
            curr_dim = next_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states from source dimension to target dimension.

        Args:
            x: Input tensor [..., source_dim].

        Returns:
            Projected tensor [..., target_dim].
        """
        return cast(torch.Tensor, self.net(x.to(torch.float32)))


class CrossModelKVProjector:
    """High-level interface for cross-model KV transfer."""

    def __init__(self, config: ProjectorConfig, device: str = "cpu") -> None:
        """
        Initialize CrossModelKVProjector.

        Args:
            config: Projector configuration.
            device: Computing device.
        """
        self.config = config
        self.device = device
        self.model = LinearProjector(config).to(device)

    @classmethod
    def from_pretrained(cls, pair: str, device: str = "cpu") -> CrossModelKVProjector:
        """
        Load a pretrained projector for a specific model pair.

        Args:
            pair: Format "source_model->target_model".
            device: Computing device.

        Returns:
            Projector instance.
        """
        if pair not in REGISTRY:
             raise ValueError(f"Unknown pair: {pair}. Registry: {list(REGISTRY.keys())}")

        weight_path = Path(REGISTRY[pair])

        # We need a proper config here. In production, we'd load this from a JSON file.
        # For the registry, we'll assume standard dimensions for these pairs.
        # Llama 3.2-3B: 3072, Mistral-7B: 4096.
        # This is a simplified fallback for demonstration.
        source_model, target_model = pair.split("->")
        config = ProjectorConfig(
            source_model=source_model,
            target_model=target_model,
            source_dim=0,  # Placeholder to be overridden by weights if possible
            target_dim=0,
            source_heads=0,
            target_heads=0,
        )

        if not weight_path.exists():
            print(f"Projector weights for {pair} not found locally.")
            print(f"Train them with: tqk train --pair {pair}")
            print("Or download from: huggingface.co/RemizovDenis/tqk-projectors")
            warnings.warn(f"Returning untrained projector for {pair}.", UserWarning)
            # Default dims for common pairs if missing
            if "llama3.2-3b" in source_model and "mistral-7b" in target_model:
                config.source_dim, config.target_dim = 3072, 4096
            return cls(config, device=device)

        # Loading logic
        weights = safetensors.torch.load_file(str(weight_path))
        # Inspect weights to determine dimensions
        proj_weight = weights["net.0.weight"]
        config.source_dim = proj_weight.shape[1]
        config.target_dim = weights[list(weights.keys())[-1]].shape[0]

        instance = cls(config, device=device)
        instance.model.load_state_dict(weights)
        return instance

    def transfer(self, kv_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Transfer KV tensors to the target model space.

        Args:
            kv_tensors: Source KV tensors.

        Returns:
            Transferred KV tensors.
        """
        self.model.eval()
        result: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key, tensor in kv_tensors.items():
                # Store original shape to restore after projection
                orig_shape = list(tensor.shape)
                # Group all dims except last one
                flattened = tensor.view(-1, self.config.source_dim).to(self.device)
                projected = self.model(flattened)
                # Restore shape with new target dimension
                new_shape = orig_shape[:-1] + [self.config.target_dim]
                result[key] = projected.view(*new_shape).cpu()
        return result

    def train_on_pairs(
        self,
        source_kvs: list[dict[str, torch.Tensor]],
        target_kvs: list[dict[str, torch.Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        callback: Any | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the projector on pairs of source and target KV caches.

        Args:
            source_kvs: List of source KV dicts.
            target_kvs: List of corresponding target KV dicts.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            callback: Optional function(epoch, loss, cosine_sim).

        Returns:
            Training history: {"train_loss": [...], "cosine_sim": [...]}.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: dict[str, list[float]] = {"train_loss": [], "cosine_sim": []}

        # Flatten all KV pairs into a single dataset of tensors for the projector
        # Keys to extract (typically layers)
        keys = sorted([k for k in source_kvs[0].keys() if k.startswith("layer_")])

        x_list: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []

        for src, tgt in zip(source_kvs, target_kvs):
            for k in keys:
                if k in tgt:
                    # Flatten spatial/head dims: [..., dim] -> [N, dim]
                    x_list.append(src[k].view(-1, self.config.source_dim))
                    y_list.append(tgt[k].view(-1, self.config.target_dim))

        X = torch.cat(x_list, dim=0).to(self.device).to(torch.float32)
        Y = torch.cat(y_list, dim=0).to(self.device).to(torch.float32)

        dataset_size = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_cos = 0.0
            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                idx = indices[i : i + batch_size]
                batch_x, batch_y = X[idx], Y[idx]

                optimizer.zero_grad()
                pred_y = self.model(batch_x)

                mse_loss = F.mse_loss(pred_y, batch_y)
                cos_sim = F.cosine_similarity(pred_y, batch_y).mean()
                loss = mse_loss + 0.1 * (1.0 - cos_sim)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(idx)
                epoch_cos += cos_sim.item() * len(idx)

            avg_loss = epoch_loss / dataset_size
            avg_cos = epoch_cos / dataset_size
            history["train_loss"].append(avg_loss)
            history["cosine_sim"].append(avg_cos)

            logger.info("epoch_end", epoch=epoch, loss=round(avg_loss, 6), cos_sim=round(avg_cos, 4))
            if callback:
                callback(epoch, avg_loss, avg_cos)

        return history

    def save(self, path: str | Path) -> None:
        """Save weights and config of the projector."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Weights
        safetensors.torch.save_file(self.model.state_dict(), str(path))
        # Config
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CrossModelKVProjector:
        """Load projector from saved directory/file."""
        path = Path(path)
        config_path = path.with_suffix(".json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            config = ProjectorConfig(**config_dict)

        instance = cls(config, device=device)
        weights = safetensors.torch.load_file(str(path))
        instance.model.load_state_dict(weights)
        return instance

    def quality(
        self,
        source_kvs: list[dict[str, torch.Tensor]],
        target_kvs: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Measure projector quality on a dataset."""
        self.model.eval()
        total_cos = 0.0
        total_mse = 0.0
        count = 0
        above_threshold = 0

        with torch.no_grad():
            for src, tgt in zip(source_kvs, target_kvs):
                transferred = self.transfer(src)
                for k, tensor in transferred.items():
                    if k in tgt:
                        t1 = tensor.flatten().to(torch.float32)
                        t2 = tgt[k].flatten().to(torch.float32)
                        cos = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
                        mse = F.mse_loss(t1, t2).item()

                        total_cos += cos
                        total_mse += mse
                        count += 1
                        if cos > 0.85:
                            above_threshold += 1

        return {
            "mean_cosine_similarity": total_cos / count if count > 0 else 0.0,
            "mean_mse": total_mse / count if count > 0 else 0.0,
            "percent_above_threshold": (above_threshold / count) if count > 0 else 0.0,
        }
