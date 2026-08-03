"""Evaluates TinyLlama model performance with Quanto and PyTorch quantization."""
import contextlib
import gc
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from datasets import load_dataset
from optimum.quanto import freeze
from optimum.quanto import qint8
from optimum.quanto import quantize
from torch.ao.quantization import quantize_dynamic
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_SEED = 42

@dataclass
class GenMetrics:
    """Stores latency, throughput, and memory metrics for text generation.
    
    Attributes:
        latency_s: Average generation latency in seconds.
        tokens_per_sec: Average generation throughput in tokens per second.
        peak_gpu_mem_mb: Peak GPU memory allocated during generation in MB.
    """
    latency_s: float
    tokens_per_sec: float
    peak_gpu_mem_mb: float

def dir_size_mb(path: str) -> float:
    """Calculates the total size of a directory in MB.

    Args:
        path: The file path to the directory.

    Returns:
        The total size of the directory in megabytes.
    """
    total_size = 0.0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

@contextlib.contextmanager
def torch_cuda_monitor(device: str) -> Iterator[None]:
    """Monitors peak CUDA memory usage during the execution of a with-block.

    Args:
        device: The device string (e.g., "cuda" or "cpu").

    Yields:
        None.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_alloc = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            torch_cuda_monitor.peak_memory = peak / (1024 * 1024)
            torch_cuda_monitor.start_mb = start_alloc / (1024 * 1024)
    else:
        try:
            yield
        finally:
            torch_cuda_monitor.peak_memory = 0.0
            torch_cuda_monitor.start_mb = 0.0

def measure_generate(
    model: Any, 
    tokenizer: Any, 
    prompt: str, 
    max_new_tokens: int = 32, 
    num_runs: int = 10,
    device: str = "cpu"
) -> GenMetrics:
    """Measures latency and throughput of the model's generate function.

    Args:
        model: The causal language model to evaluate.
        tokenizer: The tokenizer corresponding to the model.
        prompt: The input text prompt to generate from.
        max_new_tokens: The maximum number of tokens to generate.
        num_runs: The number of inference passes to average over.
        device: The target device ("cpu" or "cuda").

    Returns:
        A GenMetrics dataclass containing performance statistics.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
   
    # Warm-up
    for _ in range(3):
        _ = model.generate(
            **inputs, max_new_tokens=8, do_sample=False, use_cache=True
        )

    latencies = []
    tps = []
    
    with torch_cuda_monitor(device):
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False, 
                    use_cache=True
                )   
            if device == "cuda":
                torch.cuda.synchronize()
            
            t1 = time.perf_counter()
            gen_len = output.shape[1] - input_len
            latency = t1 - t0
            latencies.append(latency)
            tps.append(gen_len / latency if latency > 0 else 0.0)
                            
    return GenMetrics(
        latency_s=sum(latencies) / len(latencies),
        tokens_per_sec=sum(tps) / len(tps),
        peak_gpu_mem_mb=getattr(torch_cuda_monitor, "peak_memory", 0.0)
    )
@torch.no_grad()
def compute_perplexity(
    model: Any, 
    tokenizer: Any, 
    seq_len: int = 128,
    device: str = "cpu"
) -> float:
    """Estimates perplexity using a self-contained evaluation text.

    Args:
        model: The causal language model to evaluate.
        tokenizer: The tokenizer corresponding to the model.
        seq_len: The sequence length chunk size to use for evaluation.
        device: The target device ("cpu" or "cuda").

    Returns:
        The calculated perplexity score.
    """
    eval_text = (
        "Quantization reduces the precision of neural network weights and "
        "activations. This process shrinks model size, lowers memory use, and "
        "can speed up inference. The tradeoff is a small drop in accuracy. "
        "Perplexity measures how well a language model predicts text: a lower "
        "perplexity means the model is more confident in its predictions. "
        "Large language models like LLaMA or TinyLlama are evaluated on "
        "benchmarks such as WikiText, where perplexity is calculated over "
        "thousands of tokens. In practice, we only need a small text sample "
        "to compare relative changes. By quantizing a model to 8-bit, we can "
        "observe whether perplexity increases significantly. If the rise is "
        "modest while speed and memory improve, quantization is usually a "
        "good trade-off. This evaluation text is deliberately extended to "
        "ensure enough tokens for testing."
    )

    enc = tokenizer(eval_text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    usable = (len(input_ids) // seq_len) * seq_len
    input_ids = input_ids[: usable + 1]
    
    if len(input_ids) <= seq_len:
        raise ValueError("Evaluation text is too short for the sequence length.")

    nll_sum = 0.0
    tok_count = 0
    model.eval()
    
    for start in range(0, len(input_ids) - 1 - seq_len, seq_len):
        chunk = input_ids[start : start + seq_len + 1]
        inputs = chunk[:-1].unsqueeze(0).to(device)
        labels = chunk[1:].unsqueeze(0).to(device)
        
        output = model(inputs, labels=labels)
        nll_sum += output.loss.item() * labels.numel()
        tok_count += labels.numel()

    return math.exp(nll_sum / max(1, tok_count))

def save_and_size(model: Any, tokenizer: Any, out_dir: str) -> float:
    """Saves the model and tokenizer to disk and returns the size.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        out_dir: The directory where the artifacts will be saved.

    Returns:
        The total size of the output directory in megabytes (MB).
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return dir_size_mb(out_dir)

def print_row(
    title: str, 
    size_mb: float, 
    lat_s: float, 
    tps: float, 
    gpu_mb: float, 
    ppl: float
) -> None:
    """Prints a formatted row of benchmarking metrics."""
    print(
        f"{title:18s} | Size: {size_mb:8.1f} MB | Latency: {lat_s:7.3f} s | "
        f"Throughput: {tps:7.2f} tok/s | Peak VRAM: {gpu_mb:7.1f} MB | "
        f"PPL: {ppl:7.2f}"
    )
    
def main() -> None:
    """Executes model quantization benchmarking for GPU and CPU."""
    torch.manual_seed(_SEED)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if default_device == "cuda":
        torch.cuda.manual_seed(_SEED)

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------------
    # 1) Baseline: FP32 (Default Device)
    # ----------------------------------------------------------------------
    print(f"\n== Baseline FP32 ({default_device.upper()}) ==")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )  

    base_size_mb = save_and_size(base_model, tokenizer, "baseline_fp32")
    base_metrics = measure_generate(
        base_model, 
        tokenizer, 
        prompt="Hello, how are you",
        max_new_tokens=128, 
        num_runs=3,
        device=default_device
    )
    base_ppl = compute_perplexity(
        base_model, tokenizer, seq_len=128, device=default_device
    )

    print_row(
        "FP32 (baseline)", 
        base_size_mb, 
        base_metrics.latency_s, 
        base_metrics.tokens_per_sec,
        base_metrics.peak_gpu_mem_mb, 
        base_ppl
    )

    # ----------------------------------------------------------------------
    # 2) Post-Training Quantization: 8-bit (Optimum-Quanto, weight-only)
    # ----------------------------------------------------------------------
    print("\n== PTQ INT8 (Optimum-Quanto, weight-only) ==")
    
    q_model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    quantize(q_model, weights=qint8)
    freeze(q_model)
    q_model.to(default_device)

    q_size_mb = save_and_size(q_model, tokenizer, out_dir="tinyllama_quanto")
    q_metrics = measure_generate(
        q_model, 
        tokenizer, 
        prompt="Hello, how are you",
        max_new_tokens=128, 
        num_runs=3,
        device=default_device
    )
    q_ppl = compute_perplexity(
        q_model, tokenizer, seq_len=128, device=default_device
    )

    print_row(
        "INT8 (Quanto)", 
        q_size_mb, 
        q_metrics.latency_s, 
        q_metrics.tokens_per_sec,
        q_metrics.peak_gpu_mem_mb, 
        q_ppl
    )

    # Free up memory before CPU benchmarking
    del base_model
    del q_model
    if default_device == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # 3) CPU Quantization: INT8 Dynamic Quantization (PyTorch Native)
    # ----------------------------------------------------------------------
    cpu_prompt = (
        "Quantization test: explain why int8 dynamic quantization "
        "can be faster on CPU."
    )
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print("\n== CPU FP32 Baseline ==")
    model_cpu_fp32 = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, torch_dtype=torch.float32
    ).to("cpu").eval()
    model_cpu_fp32.config.use_cache = True

    metrics_cpu_fp32 = measure_generate(
        model_cpu_fp32, 
        tokenizer, 
        prompt=cpu_prompt, 
        max_new_tokens=128, 
        num_runs=3,
        device="cpu"
    )
    
    print(
        f"FP32 (CPU) | Latency {metrics_cpu_fp32.latency_s:.3f} s | "
        f"Throughput {metrics_cpu_fp32.tokens_per_sec:.2f} tok/s"
    )

    print("\n== CPU INT8 Dynamic (PyTorch Native) ==")
    model_cpu_int8 = quantize_dynamic(
        model_cpu_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    del model_cpu_fp32  # Free FP32 version

    metrics_cpu_int8 = measure_generate(
        model_cpu_int8, 
        tokenizer, 
        prompt=cpu_prompt, 
        max_new_tokens=128, 
        num_runs=3, 
        device="cpu"
    )
    
    latency_int8 = metrics_cpu_int8.latency_s
    speedup = (
        metrics_cpu_fp32.latency_s / latency_int8 
        if latency_int8 > 0 else float("inf")
    )

    print("\n== Summary (CPU) ==")
    print(
        f"CPU FP32 latency: {metrics_cpu_fp32.latency_s:.3f} s | "
        f"CPU INT8 latency: {latency_int8:.3f} s | Speedup: {speedup:.2f}x"
    )
    print(
        f"CPU FP32 tput:   {metrics_cpu_fp32.tokens_per_sec:.2f} tok/s | "
        f"CPU INT8 tput:   {metrics_cpu_int8.tokens_per_sec:.2f} tok/s"
    )


if __name__ == "__main__":
    main()