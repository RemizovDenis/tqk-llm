"""tests/test_tqk.py — Unit tests for TQK modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tqk.cli import main
from tqk.extractor import KVExtractor
from tqk.format import TQKFile, TQKMetadata
from tqk.projector import CrossModelKVProjector, LinearProjector, ProjectorConfig
from tqk.turboquant_bridge import (
    HAS_TURBOQUANT,
    TQKPipeline,
    compress_to_tqk,
    decompress_from_tqk,
    patch_model_with_tqk,
)
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


class TestProjector:
    """Test suite for LinearProjector and CrossModelKVProjector."""

    def test_linear_projector_forward_shape(self) -> None:
        """Verify that LinearProjector produces the correct output shape."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=64, target_dim=128,
            source_heads=1, target_heads=1
        )
        model = LinearProjector(config)
        x = torch.randn(4, 32, 64)
        out = model(x)
        assert out.shape == (4, 32, 128)

    def test_linear_projector_xavier_init(self) -> None:
        """Verify that weights are initialized via Xavier (not zero)."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=100, target_dim=100,
            source_heads=1, target_heads=1
        )
        model = LinearProjector(config)
        for name, param in model.named_parameters():
            if "weight" in name:
                assert not torch.all(param == 0)

    def test_transfer_preserves_keys(self) -> None:
        """Verify that CrossModelKVProjector.transfer preserves dictionary keys."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=64, target_dim=128,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)
        kv = {"layer_0_keys": torch.randn(1, 64), "layer_0_values": torch.randn(1, 64)}
        res = projector.transfer(kv)
        assert set(res.keys()) == {"layer_0_keys", "layer_0_values"}

    def test_transfer_changes_dim(self) -> None:
        """Verify that transfer changes source_dim to target_dim."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=64, target_dim=128,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)
        kv = {"layer_0_keys": torch.randn(1, 4, 64)}
        res = projector.transfer(kv)
        assert res["layer_0_keys"].shape[-1] == 128

    def test_from_pretrained_missing_weights_warns(self) -> None:
        """Verify that from_pretrained warns and doesn't crash when weights missing."""
        with pytest.warns(UserWarning, match="untrained projector"):
            projector = CrossModelKVProjector.from_pretrained("llama3.2-3b->mistral-7b")
        assert isinstance(projector, CrossModelKVProjector)

    def test_train_on_pairs_reduces_loss(self) -> None:
        """Verify that weights are updated and loss decreases during training."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=16, target_dim=16,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)

        # Create dummy related data
        source_kv = [{"layer_0_keys": torch.randn(10, 16)}]
        target_kv = [{"layer_0_keys": source_kv[0]["layer_0_keys"] * 2.5 + 0.1}]

        history = projector.train_on_pairs(source_kv, target_kv, epochs=5, lr=0.1)
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_train_returns_history(self) -> None:
        """Verify training history dictionary structure."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=16, target_dim=16,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)
        source_kv = [{"layer_0_keys": torch.randn(2, 16)}]
        target_kv = [{"layer_0_keys": torch.randn(2, 16)}]
        history = projector.train_on_pairs(source_kv, target_kv, epochs=2)
        assert "train_loss" in history
        assert "cosine_sim" in history
        assert len(history["train_loss"]) == 2

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify that save and load restore identical projector weights."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=32, target_dim=64,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)
        path = tmp_path / "proj.safetensors"
        projector.save(path)

        loaded = CrossModelKVProjector.load(path)
        assert loaded.config.source_dim == 32
        assert loaded.config.target_dim == 64

        orig_weight = projector.model.net[0].weight # type: ignore[cast]
        loaded_weight = loaded.model.net[0].weight # type: ignore[cast]
        assert torch.allclose(orig_weight, loaded_weight)

    def test_quality_identical_perfect(self) -> None:
        """If source matches target, quality should be near perfect."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="s",
            source_dim=16, target_dim=16,
            source_heads=1, target_heads=1,
            num_layers=1
        )
        # Manually set to identity for perfect score
        projector = CrossModelKVProjector(config)
        with torch.no_grad():
            for m in projector.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.eye_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        kv = [{"layer_0_keys": torch.randn(1, 16)}]
        res = projector.quality(kv, kv)
        assert res["mean_cosine_similarity"] > 0.99
        assert res["percent_above_threshold"] == 1.0

    def test_quality_random_low(self) -> None:
        """Random unrelated tensors should result in low passing rate."""
        torch.manual_seed(42)
        config = ProjectorConfig(
            source_model="s", target_model="t",
            source_dim=16, target_dim=16,
            source_heads=1, target_heads=1
        )
        projector = CrossModelKVProjector(config)
        src = [{"layer_0_keys": torch.randn(1, 16)}]
        tgt = [{"layer_0_keys": torch.randn(1, 16)}]
        res = projector.quality(src, tgt)
        # Random vectors in 16D have low expected cosine sim
        assert res["percent_above_threshold"] < 0.5


class TestTurboquantBridge:
    """Test suite for TQK bridge and pipeline."""

    def test_compress_without_turboquant(self) -> None:
        """Verify fallback compression logic (no turboquant needed)."""
        torch.manual_seed(42)
        k = torch.randn(1, 16, 16)
        v = torch.randn(1, 16, 16)
        tqk = compress_to_tqk(k, v, source_model="m")
        assert isinstance(tqk, TQKFile)
        assert tqk.metadata.source_model == "m"

    def test_decompress_roundtrip(self) -> None:
        """Verify decompression accuracy (cosine_sim > 0.95)."""
        torch.manual_seed(42)
        k = torch.randn(1, 16, 16).half()
        v = torch.randn(1, 16, 16).half()
        tqk = compress_to_tqk(k, v, source_model="m")
        dk, dv = decompress_from_tqk(tqk)

        cos_k = torch.nn.functional.cosine_similarity(k.flatten(), dk.flatten(), dim=0)
        assert cos_k > 0.95

    def test_pipeline_save_context(self, tmp_path: Path) -> None:
        """Verify TQKPipeline can save context using a mock model."""
        class MockModel:
            config = type("obj", (object,), {"hidden_size": 16})
            def to(self, *args, **kwargs): return self
            def eval(self): return self
            def __call__(self, *args, **kwargs):
                return type("obj", (object,), {"past_key_values": ((torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)),)})()

        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

        pipeline = TQKPipeline(MockModel(), source_tokenizer=MockTokenizer())
        path = tmp_path / "context.tqk"
        tqk = pipeline.save_context("test", path)
        assert path.exists()
        assert tqk.metadata.source_model == "pipeline"

    def test_patch_model_unsupported_warns(self) -> None:
        """Verify warning when patching a non-HF model."""
        class NotAModel:
            pass
        with pytest.warns(UserWarning, match="not appear to be a standard HuggingFace"):
            tqk = TQKFile({}, TQKMetadata(source_model="m"))
            res = patch_model_with_tqk(NotAModel(), tqk)
            assert isinstance(res, NotAModel)

    def test_has_turboquant_flag(self) -> None:
        """Verify HAS_TURBOQUANT global flag exists and is boolean."""
        assert isinstance(HAS_TURBOQUANT, bool)
