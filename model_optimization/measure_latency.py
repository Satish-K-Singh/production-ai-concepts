"""Benchmarks latency and throughput for a Hugging Face causal language model."""
import time
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_MAX_NEW_TOKENS = 50

def load_model_and_tokenizer(
    model_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    """Loads the model and tokenizer from Hugging Face.

    Args:
        model_name: The Hugging Face model identifier.

    Returns:
        A tuple containing the tokenizer, the loaded model, and the torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.to(device).eval()
    return tokenizer, model, device

def prepare_prompt(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, torch.Tensor], int]:
    """Prepares and tokenizes the input prompt.

    Args:
        tokenizer: The tokenizer associated with the model.

    Returns:
        A tuple containing the tokenized inputs (as a dictionary of tensors) 
        and the length of the input sequence.
    """
    prompt = "I am working on model optimization and checking latency and throughput:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    inputs_dict = dict(inputs)
    input_len = inputs_dict["input_ids"].shape[1]
    
    return inputs_dict, input_len

def warmup_model(model: PreTrainedModel, inputs: Mapping[str, torch.Tensor], device: torch.device) -> None:
    """Warms up the model by running a few inference passes.

    Args:
        model: The pre-trained model to be warmed up.
        inputs: A dictionary containing the tokenized input tensors.
        device: The torch device (CPU or CUDA) on which the model is running.
    """
    _ = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
    if device.type == "cuda":
        torch.cuda.synchronize()

def measure_generation(
    model: PreTrainedModel,
    inputs: Mapping[str, torch.Tensor],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[float, int]:
    """Measures the time taken and tokens generated for model generation.

    Args:
        model: The causal language model.
        inputs: A dictionary of input tensors already on the target device.
        max_new_tokens: The maximum number of tokens to generate.
        device: The device the model is running on.

    Returns:
        A tuple containing the latency (in seconds) and the total number of 
        newly generated tokens.
    """
    input_len = inputs["input_ids"].shape[1]
    
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    latency = end_time - start_time
    generated_tokens = outputs.shape[1] - input_len
    
    return latency, generated_tokens

def main() -> None:
    """Executes the model loading, warmup, and benchmarking process."""
    tokenizer, model, device = load_model_and_tokenizer(_MODEL_NAME)
    
    inputs, _ = prepare_prompt(tokenizer)
    
    # Move inputs to the correct device once
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    warmup_model(model, inputs, device)
    
    latency, generated_tokens = measure_generation(
        model, inputs, _MAX_NEW_TOKENS, device
    )
    
    throughput = generated_tokens / latency
    
    print(f"Generated tokens: {generated_tokens}")
    print(f"Latency         : {latency:.3f} s")
    print(f"Throughput      : {throughput:.1f} tokens/s")


if __name__ == "__main__":
    main()