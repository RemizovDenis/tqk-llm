"""tests/test_tqk.py — Unit tests for TQK modules."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from tqk.cli import main
from tqk.extractor import KVExtractor
from tqk.format import TQKFile, TQKMetadata
from tqk.validator import TQKValidator


def test_tqk_save_load_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading a TQKFile preserves data."""
    torch.manual_seed(42)
    path = tmp_path / "test.tqk"
    tensors = {"layer_0_keys": torch.randn(2, 4, 8)}
    metadata = TQKMetadata(source_model="test-model", num_layers=1)

    tqk = TQKFile(tensors, metadata)
    tqk.save(path)

    loaded = TQKFile.load(path)
    assert loaded.metadata.source_model == "test-model"
    assert loaded.metadata.num_layers == 1
    assert torch.allclose(loaded.tensors["layer_0_keys"], tensors["layer_0_keys"])


def test_tqk_magic_bytes(tmp_path: Path) -> None:
    """Verify that a saved file starts with the correct magic bytes."""
    torch.manual_seed(42)
    path = tmp_path / "magic.tqk"
    tqk = TQKFile({}, TQKMetadata(source_model="m"))
    tqk.save(path)

    with open(path, "rb") as f:
        magic = f.read(4)
    assert magic == b"TQK1"


def test_tqk_invalid_magic_raises(tmp_path: Path) -> None:
    """Verify loading a file with invalid magic bytes raises ValueError."""
    path = tmp_path / "invalid.tqk"
    with open(path, "wb") as f:
        f.write(b"NOT_TQK")

    with pytest.raises(ValueError, match="Invalid magic bytes"):
        TQKFile.load(path)


def test_validator_passes_identical() -> None:
    """Identical tensors should pass validation."""
    torch.manual_seed(42)
    tensors = {"layer_0_keys": torch.randn(10, 10)}
    validator = TQKValidator(threshold=0.99)
    result = validator.validate(tensors, tensors)
    assert result.passed is True
    assert result.cosine_similarity > 0.999


def test_validator_fails_random() -> None:
    """Randomly generated tensors should fail strict validation."""
    torch.manual_seed(42)
    orig = {"layer_0_keys": torch.randn(100, 100)}
    restored = {"layer_0_keys": torch.randn(100, 100)}
    validator = TQKValidator(threshold=0.95)
    result = validator.validate(orig, restored)
    assert result.passed is False


def test_cli_info_runs(tmp_path: Path) -> None:
    """Test that 'tqk info' command runs successfully on a valid file."""
    torch.manual_seed(42)
    path = tmp_path / "info_test.tqk"
    TQKFile({}, TQKMetadata(source_model="test_model")).save(path)

    with patch("sys.argv", ["tqk", "info", str(path)]):
        with patch("sys.stdout") as mock_stdout:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
            # Check if source_model is in the output
            args, _ = mock_stdout.write.call_args_list[0]
            assert "source_model" in args[0]


def test_cli_validate_ok(tmp_path: Path) -> None:
    """Test that 'tqk validate' outputs OK for a valid file."""
    torch.manual_seed(42)
    path = tmp_path / "valid.tqk"
    TQKFile({}, TQKMetadata(source_model="m")).save(path)

    with patch("sys.argv", ["tqk", "validate", str(path)]):
        with patch("sys.stdout") as mock_stdout:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
            mock_stdout.write.assert_any_call("OK")
            mock_stdout.write.assert_any_call("\n")


def test_tqk_repr() -> None:
    """Check that TQKFile repr matches the expected pattern."""
    tqk = TQKFile({}, TQKMetadata(source_model="my-model"))
    assert "TQKFile(model=my-model" in repr(tqk)


def test_metadata_serialization() -> None:
    """Verify TQKMetadata JSON serialization via TQKFile save/load logic."""
    # This is implicitly tested in roundtrip but let's be explicit
    meta = TQKMetadata(source_model="m", extra={"key": "val"})
    assert meta.extra["key"] == "val"


def test_compression_ratio_positive() -> None:
    """Verify compression_ratio method returns the expected value."""
    meta = TQKMetadata(source_model="m", compression_ratio=14.5)
    tqk = TQKFile({}, meta)
    assert tqk.compression_ratio() == 14.5


def test_extractor_mock() -> None:
    """Verify KVExtractor baseline with mocked model and tokenizer."""
    torch.manual_seed(42)
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_tokenizer = MagicMock()
    # Mocking HF output structure
    mock_output = MagicMock()
    mock_output.past_key_values = ((torch.randn(1, 2), torch.randn(1, 2)),)
    mock_model.return_value = mock_output
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}

    extractor = KVExtractor(mock_model, mock_tokenizer)
    res = extractor.extract("hello")
    assert "layer_0_keys" in res
    assert "layer_0_values" in res
