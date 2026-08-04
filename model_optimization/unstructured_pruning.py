"""Global magnitude pruning utility for large language models.

This script loads a pretrained Causal LM, applies global magnitude pruning
to its Linear layers based on a threshold approximated via histograms, 
and saves the resulting sparse weights in CSR format.
"""

import gc
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CANDIDATES: List[str] = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def load_fallback_model(
    candidates: List[str], device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Attempts to load a model and tokenizer from a list of candidate names.

    Args:
        candidates: A list of Hugging Face hub model identifiers.
        device: The torch device to load the model onto.

    Returns:
        A tuple containing the loaded model and tokenizer.

    Raises:
        RuntimeError: If all candidate models fail to load.
    """
    last_err: Exception | None = None

    for name in candidates:
        try:
            logging.info(f"Attempting to load {name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True, use_fast=False
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            
            logging.info(f"Successfully loaded {name}.")
            return model, tokenizer

        except Exception as e:
            logging.warning(f"Failed to load {name}. Error: {e}")
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to load any model from candidates. Last error: {last_err}"
    )


def _get_linear_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Gathers all nn.Linear modules in the model that have a weight attribute.

    Args:
        model: The PyTorch module to inspect.

    Returns:
        A list of tuples containing the module reference and the string "weight".
    """
    linear_weights = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None:
            linear_weights.append((module, "weight"))
    return linear_weights


@torch.no_grad()
def _compute_global_threshold(
    params_to_prune: List[Tuple[nn.Module, str]], 
    amount: float, 
    bins: int = 2048
) -> Tuple[float, int]:
    """Computes a global magnitude threshold via histogram approximation.

    Args:
        params_to_prune: List of (module, parameter_name) tuples.
        amount: Fraction of weights to prune (0.0 to 1.0).
        bins: Number of histogram bins.

    Returns:
        A tuple containing the computed threshold and the total number of elements.
    """
    gmin, gmax = float("inf"), 0.0
    total_elems = 0
    
    # Find global min/max absolute values
    for mod, pname in params_to_prune:
        weight = getattr(mod, pname).detach()
        total_elems += weight.numel()
        a = weight.abs().float().cpu()
        gmin = min(gmin, a.min().item())
        gmax = max(gmax, a.max().item())
        del a

    if not math.isfinite(gmin):
        gmin = 0.0

    if gmax <= gmin:
        return 0.0, total_elems

    # Populate histogram
    hist = torch.zeros(bins, dtype=torch.int64)
    for mod, pname in params_to_prune:
        a = getattr(mod, pname).detach().abs().float().cpu()
        hist += torch.histc(a, bins=bins, min=gmin, max=gmax)
        del a

    cum = torch.cumsum(hist, dim=0)
    k = int(amount * total_elems)
    k = max(0, min(k, total_elems - 1))
    idx = int(torch.searchsorted(cum, torch.tensor(k, dtype=torch.long)))

    bin_width = (gmax - gmin) / bins
    threshold = gmin + bin_width * (idx + 1)
    
    return float(threshold), total_elems


@torch.no_grad()
def global_prune_linear_weights_streamed(
    model: nn.Module, amount: float = 0.2, bins: int = 2048
) -> None:
    """Applies global magnitude pruning to all linear layers in the model in place.

    Args:
        model: The model to prune.
        amount: The target sparsity ratio (0.0 to 1.0).
        bins: Resolution of the histogram approximation.
    """
    logging.info(f"Computing global threshold for target sparsity: {amount:.2%}")
    params_to_prune = _get_linear_modules(model)
    threshold, total_elems = _compute_global_threshold(params_to_prune, amount, bins=bins)
    logging.info(f"Computed threshold: {threshold:.6f}")

    pruned = 0
    for mod, pname in params_to_prune:
        weight = getattr(mod, pname)
        mask = weight.abs() > threshold
        pruned += weight.numel() - mask.sum().item()
        weight.data.mul_(mask)
        del mask

    actual_sparsity = pruned / max(1, total_elems)
    logging.info(f"Pruning complete. Actual global sparsity: {actual_sparsity:.2%}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def save_pruned_model(model: nn.Module, save_dir: Path) -> None:
    """Saves the pruned model weights in both CSR format and dense state_dict.

    Args:
        model: The pruned PyTorch model.
        save_dir: Path to the directory where weights should be saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    sparse_dump: Dict[str, Dict[str, Any]] = {}
    nonzero_total = 0
    element_total = 0

    logging.info("Extracting sparse weights for serialization...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None:
                weight = module.weight.detach().to("cpu")
                element_total += weight.numel()
                nonzero_total += (weight != 0).sum().item()
                
                weight_csr = weight.to_sparse_csr()
                sparse_dump[name] = {
                    "weight_csr": weight_csr,
                    "shape": weight.shape,
                    "dtype": str(weight.dtype),
                }

    csr_path = save_dir / "linear_weights_csr.pt"
    torch.save(sparse_dump, csr_path)
    logging.info(f"Saved CSR sparse weights to: {csr_path}")
    
    if element_total > 0:
        density = nonzero_total / element_total
        sparsity = 1.0 - density
        logging.info(f"Overall density: {density:.2%} | Sparsity: {sparsity:.2%}")

    dense_path = save_dir / "pruned_dense_state_dict.pt"
    torch.save(model.state_dict(), dense_path)
    logging.info(f"Saved dense pruned state_dict to: {dense_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Model
    model, _ = load_fallback_model(CANDIDATES, device)

    # 2. Prune Model (20% Global Magnitude)
    global_prune_linear_weights_streamed(model, amount=0.2, bins=2048)

    # 3. Save Artifacts
    save_dir = Path("pruned_models")
    save_pruned_model(model, save_dir)


if __name__ == "__main__":
    main()