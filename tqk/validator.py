"""tqk/validator.py — Metric measurement for saved KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from tqk.format import TQKFile


@dataclass
class ValidationResult:
    """Result of a KV-cache validation check."""

    cosine_similarity: float
    mse: float
    passed: bool
    details: dict[str, float]


class TQKValidator:
    """Validator for measuring the quality of compressed and restored KV caches."""

    def __init__(self, threshold: float = 0.85) -> None:
        """
        Initialize TQKValidator.

        Args:
            threshold: Minimum acceptable cosine similarity for a 'passed' result.
        """
        self.threshold = threshold

    def validate(
        self,
        original: dict[str, torch.Tensor],
        restored: dict[str, torch.Tensor],
    ) -> ValidationResult:
        """
        Calculate cosine similarity and MSE between original and restored tensors.

        Args:
            original: Dictionary of ground truth KV tensors.
            restored: Dictionary of reconstructed KV tensors.

        Returns:
            ValidationResult with metrics and pass/fail status.
        """
        total_cos_sim = 0.0
        total_mse = 0.0
        count = 0
        details: dict[str, float] = {}
        all_passed = True

        # Consider only keys starting with "layer_" to avoid meta-tensors
        keys = sorted([k for k in original.keys() if k.startswith("layer_")])

        for key in keys:
            if key not in restored:
                continue

            orig_t = original[key].to(torch.float32).flatten()
            rest_t = restored[key].to(torch.float32).flatten()

            if orig_t.numel() == 0:
                continue

            # Cosine similarity: F.cosine_similarity needs at least 2 dims usually,
            # but for vectors we can do it manually or via unsqueeze.
            cos_sim = F.cosine_similarity(orig_t.unsqueeze(0), rest_t.unsqueeze(0)).item()
            mse = F.mse_loss(orig_t, rest_t).item()

            total_cos_sim += cos_sim
            total_mse += mse
            details[key] = cos_sim
            count += 1

            if cos_sim < self.threshold:
                all_passed = False

        avg_cos_sim = total_cos_sim / count if count > 0 else 0.0
        avg_mse = total_mse / count if count > 0 else 0.0

        if count == 0:
            all_passed = False

        return ValidationResult(
            cosine_similarity=avg_cos_sim,
            mse=avg_mse,
            passed=all_passed,
            details=details,
        )

    def validate_file(
        self, tqk_file: TQKFile, original: dict[str, torch.Tensor]
    ) -> ValidationResult:
        """
        Validate a TQKFile against original ground truth tensors.

        Args:
            tqk_file: Loaded TQKFile instance.
            original: Ground truth tensors.

        Returns:
            Validation result.
        """
        restored = tqk_file.to_cache_entry()
        return self.validate(original, restored)

    def summary(self, result: ValidationResult) -> str:
        """
        Return a human-readable summary string of the validation result.

        Args:
            result: The validation result to summarize.

        Returns:
            Formatted string describing the status.
        """
        status = "PASSED" if result.passed else "FAILED"
        return (
            f"{status} cosine_sim={result.cosine_similarity:.4f} "
            f"mse={result.mse:.6f}"
        )
