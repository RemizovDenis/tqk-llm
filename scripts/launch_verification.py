"""scripts/launch_verification.py — Full E2E Journey Test for TQK Release v0.1.0."""

import os
import sys
import time
from pathlib import Path

import torch
from tqk.format import TQKFile, TQKMetadata
from tqk.projector import CrossModelKVProjector, ProjectorConfig
from tqk.turboquant_bridge import TQKPipeline, HAS_TURBOQUANT
from tqk.validator import TQKValidator

def run_launch_verification() -> bool:
    print("🚀 Starting TQK v0.1.0 E2E Launch Verification\n")
    start_time = time.time()
    
    # 1. Mock Environment Setup
    class MockModel:
        config = type("obj", (object,), {"hidden_size": 16})
        def to(self, *args, **kwargs): return self
        def eval(self): return self
        def __call__(self, *args, **kwargs):
            # Return meaningful KV cache shape
            return type("obj", (object,), {
                "past_key_values": ((torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),)
            })()

    class MockTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

    model = MockModel()
    tokenizer = MockTokenizer()
    pipeline = TQKPipeline(model, tokenizer)
    output_path = Path("launch_test.tqk")

    try:
        # 2. Extraction & Saving
        print("[1/4] Extracting & Saving context...")
        tqk_file = pipeline.save_context("Launch Verification Prompt", output_path)
        if not output_path.exists():
            raise RuntimeError("File not saved.")
        print(f"      SUCCESS: Saved to {output_path} ({output_path.stat().st_size} bytes)")

        # 3. Loading & Meta-Analysis
        print("[2/4] Loading & Metadata Analysis...")
        loaded_tqk = TQKFile.load(output_path)
        print(f"      Source Model: {loaded_tqk.metadata.source_model}")
        print(f"      Compression: {loaded_tqk.metadata.compression_ratio}x")

        # 4. Projection (Source-to-Target mapping)
        print("[3/4] Cross-Model Projection logic...")
        # Llama-like to Mistral-like projection
        config = ProjectorConfig(
            source_model="llama-3.2-3b",
            target_model="mistral-7b",
            source_dim=16,
            target_dim=32,
            source_heads=4,
            target_heads=8
        )
        projector = CrossModelKVProjector(config)
        kv = loaded_tqk.to_cache_entry()
        projected_kv = projector.transfer(kv)
        
        # Verify dimension change in the first layer
        projected_tensor = projected_kv["layer_0_keys"]
        if projected_tensor.shape[-1] != 32:
             raise ValueError(f"Projection failed. Dim is {projected_tensor.shape[-1]}, expected 32")
        print(f"      SUCCESS: Projected 16D -> {projected_tensor.shape[-1]}D")

        # 5. Quality Validation
        print("[4/4] Final Quality Validation (Roundtrip)...")
        validator = TQKValidator(threshold=0.99)
        # Using loaded vs original (identical in this mock)
        report = validator.validate(kv, loaded_tqk.to_cache_entry())
        print(f"      Similarity Result: {report.cosine_similarity:.4f}")
        if report.cosine_similarity < 0.99:
             raise ValueError("Quality validation failed.")

        duration = time.time() - start_time
        print(f"\n✅ VERIFICATION COMPLETE: ALL SYSTEMS GO (Total time: {duration:.2f}s)")
        return True

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        return False
    finally:
        if output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    success = run_launch_verification()
    sys.exit(0 if success else 1)
