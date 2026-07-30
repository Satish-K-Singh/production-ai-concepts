"""Evaluates TinyLlama model for peak memory, perplexity, and latency."""
import math
import os 
import time
from typing import Dict, Tuple

# Set environment variables before importing transformers to disable TF/Flax.
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizerBase
)

_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    """Loads the model and tokenizer from Hugging Face.

    Args:
        model_name: The Hugging Face model identifier.

    Returns:
        A tuple containing the initialized tokenizer, the pre-trained model, 
        and the torch device (CUDA if available, else CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device).eval()
    return tokenizer, model, device

def select_text() -> str:
    """Selects a text prompt for evaluation.
    
    Returns:
        A string containing the evaluation prompt.
    """
    return (
        "I am working on model optimization and checking peak memory and perplexity:\n"
    )

def tokenize_with_labels(tokenizer: PreTrainedTokenizerBase, text: str) -> Dict[str, torch.Tensor]:
    """Tokenizes the input text and prepares labels for perplexity calculation.

    Args:
        tokenizer: The tokenizer associated with the model.
        text: The input text to be tokenized.

    Returns:
        A tuple containing a dictionary of tokenized inputs (tensors) 
        and the integer length of the input.
    """ 
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()  
    input_len = input_ids.shape[1]
    return inputs, input_len

def compute_peak_memory_and_loss(model: PreTrainedModel, inputs: Dict[str, torch.Tensor], device: torch.device) -> tuple[float, float]:
    """Computes the peak memory usage and loss for the given inputs.

    Args:
        model: The pre-trained model to be evaluated.
        inputs: A dictionary containing the tokenized input tensors.
        device: The torch device (CPU or CUDA) on which the model is running.

    Returns:
        A tuple containing the peak memory allocated in MiB (or NaN if on CPU) 
        and the computed loss as a float.
    """
    
    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs.get("attention_mask", None).to(device) if "attention_mask" in inputs else None,
            labels=inputs["input_ids"].to(device) 
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mib = peak_bytes / (1024 ** 2)

    else:
        peak_mib = float('nan')  

    return peak_mib, outputs.loss.item()

def compute_perplexity(loss: float) -> float:
    """Computes the perplexity from the loss.

    Args:
        loss: The loss value obtained from the model's output.

    Returns:
        The computed perplexity value.
    """
    return math.exp(loss)

def time_forward(model, inputs, device, num_warmups=1, num_runs=3):
    """Times the forward pass of the model.

    Args:
        model: The pre-trained model to be evaluated.
        inputs: A dictionary containing the tokenized input tensors.
        device: The torch device (CPU or CUDA) on which the model is running.
        num_warmups: Number of warm-up runs before timing.
        num_runs: Number of runs to time for averaging.
    Returns:
        The average latency of the forward pass in seconds.
    """
    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(**{k:v.to(device) for k,v in inputs.items()})

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(**{k:v.to(device) for k,v in inputs.items()})
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append(end_time - start_time)

    return sum(latencies) / max(len(latencies),1)  

def main() -> None:
    """Executes the model loading, evaluation, and reporting process."""
    tokenizer, model, device = load_model_and_tokenizer(_MODEL_NAME)
    
    text = select_text()
    inputs, input_len = tokenize_with_labels(tokenizer, text)
    
    peak_mib, loss = compute_peak_memory_and_loss(model, inputs, device)
    perplexity = compute_perplexity(loss)
    avg_latency = time_forward(model, inputs, device)

    print(f"Peak Memory (MiB): {peak_mib:.2f}")
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Latency (s): {avg_latency:.4f}")

if __name__ == "__main__":
    main() 