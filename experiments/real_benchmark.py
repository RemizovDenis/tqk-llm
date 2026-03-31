"""experiments/real_benchmark.py — Real-world TQK Benchmark with Qwen2.5-0.5B."""

import os
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqk.turboquant_bridge import TQKPipeline
from tqk.validator import TQKValidator
from tqk.format import TQKFile

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cpu"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 3 technical texts (~500 words each) for benchmarking
TEXTS = [
    "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks. A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers; but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning. Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain. In its application across business problems, machine learning is also referred to as predictive analytics. " * 10,
    "The transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV). Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not need to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times. Transformers were introduced in 2017 by a team at Google Brain and are increasingly the model of choice for NLP problems, replacing RNN models such as long short-term memory (LSTM). " * 10,
    "Quantum computing is a type of computation whose operations can harness the phenomena of quantum mechanics, such as superposition, interference, and entanglement. Devices that perform quantum computations are known as quantum computers. Though current quantum computers are too small to outperform usual (classical) computers for practical applications, they are believed to be capable of solving certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers. The study of quantum computing is a subfield of quantum information science. There are several types of quantum computers (also known as quantum computing systems), including the quantum circuit model, quantum Turing machine, adiabatic quantum computer, one-way quantum computer, and various quantum cellular automata. The most widely used model is the quantum bit (qubit), which is based on the quantum bit, or qubit. " * 10
]

def run_benchmark():
    print(f"🚀 Initializing Real-World Benchmark with {MODEL_ID} on {DEVICE}...")
    
    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    pipeline = TQKPipeline(model, tokenizer)
    validator = TQKValidator()
    
    results = []
    
    for i, text in enumerate(TEXTS):
        print(f"\n[Test {i+1}/3] Processing {len(text.split())} words...")
        
        # Measure Save Latency
        save_path = OUTPUT_DIR / f"bench_{i}.tqk"
        start_save = time.time()
        pipeline.save_context(text, save_path)
        save_latency = (time.time() - start_save) * 1000 # ms
        
        # Measure Load Latency
        start_load = time.time()
        loaded_tqk = TQKFile.load(save_path)
        restored_kv = loaded_tqk.to_cache_entry()
        load_latency = (time.time() - start_load) * 1000 # ms
        
        # Measure Quality (against live extraction)
        original_kv = pipeline.extractor.extract(text)
        q_report = validator.validate(original_kv, restored_kv)
        
        # Size Metrics
        tqk_size_kb = save_path.stat().st_size / 1024
        # Estimate FP16 size: [2 layers * head_dim * heads * seq_len * 2 bytes]
        # Qwen2.5-0.5B has 24 layers, 14 heads, 64 dim.
        # But let's just use the actual tensor nbytes
        raw_size_kb = sum(t.nbytes for t in original_kv.values()) / 1024
        
        res = {
            "test_id": i + 1,
            "word_count": len(text.split()),
            "save_ms": round(save_latency, 2),
            "load_ms": round(load_latency, 2),
            "tqk_kb": round(tqk_size_kb, 2),
            "raw_kb": round(raw_size_kb, 2),
            "compression": round(raw_size_kb / tqk_size_kb, 2),
            "cosine_sim": round(q_report.cosine_similarity, 6)
        }
        results.append(res)
        print(f"      Fidelity: {res['cosine_sim']:.6f} | Size: {res['tqk_kb']} KB | Save: {res['save_ms']} ms")
        
        # Cleanup
        if save_path.exists():
            save_path.unlink()

    # 4. Final Output
    report_path = OUTPUT_DIR / "real_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print(f"{'ID':<4} | {'Words':<6} | {'Save (ms)':<10} | {'Load (ms)':<10} | {'TQK (KB)':<10} | {'CosSim':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['test_id']:<4} | {r['word_count']:<6} | {r['save_ms']:<10} | {r['load_ms']:<10} | {r['tqk_kb']:<10} | {r['cosine_sim']:.6f}")
    print("="*80)
    print(f"✅ Benchmark Complete! Saved to {report_path}")

if __name__ == "__main__":
    run_benchmark()
