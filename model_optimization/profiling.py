"""Profiles and optimizes a large language model using PyTorch and Transformers.

This script loads a Hugging Face model, generates a batch of inputs, profiles
the baseline model inference, and demonstrates applying magnitude pruning and
dynamic quantization.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as tq
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PRUNE_AMOUNT = 0.30  # 30% magnitude pruning
BATCH_SIZE = 5
MAX_NEW_TOKENS = 50
PROMPT = (
    "In a world increasingly driven by artificial intelligence, the ability to interpret "
    "large language models efficiently is crucial for both research and deployment."
)
LOGDIR_BASELINE = "./profiler_logs/baseline"


def load_model(
    model_name: str, device: torch.device
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Loads a pre-trained model and tokenizer from Hugging Face.

    Args:
        model_name: The identifier of the model on the Hugging Face Hub.
        device: The PyTorch device to load the model onto.

    Returns:
        A tuple containing the initialized tokenizer and model.
    """
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Ensure pad_token is set for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    
    return tokenizer, model


def make_batch(
    tokenizer: PreTrainedTokenizerBase, prompt: str, device: torch.device, batch_size: int
) -> Dict[str, torch.Tensor]:
    """Creates a batch of tokenized inputs from a given prompt.

    Args:
        tokenizer: The initialized Hugging Face tokenizer.
        prompt: The text prompt to tokenize.
        device: The PyTorch device to place tensors on.
        batch_size: The number of identical prompts in the batch.

    Returns:
        A dictionary of tensor inputs ready for model inference.
    """
    texts = [prompt] * batch_size
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    return {key: value.to(device) for key, value in inputs.items()}


def profile_inference(
    model: PreTrainedModel, inputs: Dict[str, torch.Tensor], logdir: str, label: str
) -> None:
    """Profiles the inference of the model using torch.profiler.

    Args:
        model: The PyTorch model to profile.
        inputs: The dictionary of tokenized inputs.
        logdir: The directory path where trace files will be saved.
        label: A descriptive string label for the profiling block.
    """
    log_path = Path(logdir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting profiling for %s...", label)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_path)),
    ) as prof:
        with record_function(label):
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    logger.info("\n=== %s Top-3 ops by CPU self time ===", label)
    logger.info("\n%s", prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=3))

    if torch.cuda.is_available():
        logger.info("\n=== %s Top-3 ops by CUDA self time ===", label)
        logger.info("\n%s", prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))

    # Summarize total CPU vs CUDA self times safely
    events = prof.key_averages()
    
    # Use getattr to default to 0 if the attribute is missing (e.g., on CPU-only machines)
    total_cpu = sum(getattr(event, "self_cpu_time_total", 0) for event in events)
    total_cuda = sum(getattr(event, "self_cuda_time_total", 0) for event in events)
    
    logger.info("\n=== %s Total self-time ===", label)
    logger.info("CPU  : %.2f ms", total_cpu / 1e3)
    logger.info("CUDA : %.2f ms", total_cuda / 1e3)
    logger.info("Trace files for '%s' written to: %s", label, logdir)


def apply_pruning_and_quant(model: PreTrainedModel) -> nn.Module:
    """Applies unstructured L1 pruning and dynamic quantization to a model.

    Args:
        model: The PyTorch model to optimize.

    Returns:
        The optimized, quantized PyTorch model.
    """
    logger.info("Applying magnitude pruning (%.2f) and dynamic quantization...", PRUNE_AMOUNT)
    model.cpu()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=PRUNE_AMOUNT)
            prune.remove(module, "weight")

    quantized_model = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model


def main() -> None:
    """Main execution entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer, model = load_model(MODEL_NAME, device)
    batch_inputs = make_batch(tokenizer, PROMPT, device, BATCH_SIZE)
    
    profile_inference(model, batch_inputs, LOGDIR_BASELINE, "Baseline Model")
  
if __name__ == "__main__":
    main()