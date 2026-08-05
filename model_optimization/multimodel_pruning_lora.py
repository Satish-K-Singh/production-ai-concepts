"""Initializes and loads the Qwen2-VL model for pruning and LoRA fine-tuning."""
import gc
import os
import platform
import random
import math
from typing import Any, Dict, List
import torch
import torch.nn as nn
from typing import Iterator, Tuple

# Set environment variables before heavy imports to ensure they take effect
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

# Configuration and Constants
SEED = 42
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
PRUNE_MODE = "ffn-channels"
PRUNE_RATIO = 0.15
USE_FP32 = False

LR = 1e-4
EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 1
MAX_LENGTH = 512
WARMUP_STEPS = 20
N_TRAIN = 800
N_VAL = 200

OUTPUT_DIR = "./demo_pruned_lora_qwen2vl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DTYPE = torch.float32 if USE_FP32 else torch.float16

# Initialization
def set_seed(seed: int) -> None:
    """Sets the random seed for deterministic execution."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda} | Py: {platform.python_version()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Model: {MODEL_NAME} | Prune: {PRUNE_MODE} | Ratio: {PRUNE_RATIO} | Output: {OUTPUT_DIR}")

# Load Processor and Config
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Model Loading and Architecture Inspection

def load_model(
    model_name: str, 
    target_dtype: torch.dtype, 
    device: torch.device
) -> Tuple[AutoModelForVision2Seq, torch.dtype]:
    """Loads the Vision2Seq model with specified dtype, falling back to FP16 on OOM.

    Args:
        model_name: The Hugging Face model identifier.
        target_dtype: The desired PyTorch data type for the weights.
        device: The target device (CPU/GPU) to load the model onto.

    Returns:
        A tuple containing the loaded PyTorch model and the actual dtype used.
    """
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        return model, target_dtype

    except torch.cuda.OutOfMemoryError:
        print("OOM at requested dtype; falling back to FP16.")
        torch.cuda.empty_cache()
        gc.collect()
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        return model, torch.float16


# Load the model and enable gradient checkpointing for memory efficiency
model, actual_dtype = load_model(MODEL_NAME, TARGET_DTYPE, DEVICE)
model.grad_checkpointing_enable()

# Safely extract architectural dimensions (accounting for different config structures)
hidden_size = getattr(model.config, "hidden_size", getattr(config, "hidden_dim", None))
num_heads = getattr(model.config, "num_attention_heads", getattr(config, "num_heads", None))
intermediate_size = getattr(model.config, "intermediate_size", getattr(config, "ffn_hidden_size", None))

if not all([hidden_size, num_heads, intermediate_size]):
    raise ValueError("Failed to extract one or more key architectural dimensions from the config.")

head_dim = hidden_size // num_heads
total_params = sum(p.numel() for p in model.parameters()) / 1e6

print(
    f"Model loaded with {total_params:.2f}M parameters | "
    f"hidden_size: {hidden_size} | num_heads: {num_heads} | "
    f"intermediate_size: {intermediate_size} | head_dim: {head_dim}"
)

# Architecture Search Utilities
def find_attention_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yields all attention modules within the model.

    Args:
        model: The PyTorch model to inspect.

    Yields:
        A tuple of (module_name, module_instance).
    """
    for name, module in model.named_modules():
        if all(hasattr(module, proj) for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            yield name, module


def find_mlp_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yields all Multilayer Perceptron (MLP) modules within the model.

    Args:
        model: The PyTorch model to inspect.

    Yields:
        A tuple of (module_name, module_instance).
    """
    for name, module in model.named_modules():
        if all(hasattr(module, proj) for proj in ["gate_proj", "up_proj", "down_proj"]):
            yield name, module

# Pruning Utilities (Logical Masking)
@torch.no_grad()
def prune_attention_heads_logical_gqa(attn_mod: nn.Module, ratio: float) -> None:
    """Prunes attention heads in a module using logical masking (zeroing weights).

    Supports Grouped-Query Attention (GQA) and standard Multi-Head Attention.
    Instead of removing dimensions (which requires reallocation), this sets the
    corresponding weight rows/columns to zero.

    Args:
        attn_mod: The attention module to prune.
        ratio: The fraction of attention heads to prune (0.0 to 1.0).

    Raises:
        ValueError: If head dimensions cannot be inferred or GQA math is invalid.
    """
    # Safely extract the number of query heads
    n_q = (
        getattr(attn_mod, "num_heads", None)
        or getattr(attn_mod, "n_heads", None)
        or getattr(getattr(attn_mod, "config", None), "num_attention_heads", None)
    )

    # Safely extract the number of key/value heads (defaults to n_q for standard MHA)
    n_kv = (
        getattr(attn_mod, "num_key_value_heads", None)
        or getattr(attn_mod, "n_kv_heads", None)
        or getattr(getattr(attn_mod, "config", None), "num_key_value_heads", None)
        or n_q
    )

    # Infer or extract head dimension
    hd = getattr(attn_mod, "head_dim", None)
    if hd is None:
        q_rows = attn_mod.q_proj.weight.shape[0]
        if not (n_q and q_rows % n_q == 0):
            raise ValueError("Cannot infer head_dim: q_proj rows not divisible by n_q.")
        hd = q_rows // n_q

    # Validate architectural invariants
    if not (n_q and n_kv and n_q >= 1 and n_kv >= 1):
        raise ValueError(f"Invalid head counts: n_q={n_q}, n_kv={n_kv}")
    if n_q % n_kv != 0:
        raise ValueError("Expected n_q to be perfectly divisible by n_kv for GQA.")

    # Determine heads to keep vs. prune
    n_keep_q = max(1, int(n_q * (1.0 - ratio)))
    prune_q = list(range(n_keep_q, n_q))
    
    if not prune_q:
        return  # Nothing to prune

    def get_rows_for_heads(head_ids: list[int], per_head: int) -> list[int]:
        """Calculates flat row indices for a given list of head indices."""
        rows = []
        for h in head_ids:
            start = h * per_head
            rows.extend(range(start, start + per_head))
        return rows

    # Calculate affected rows for Query heads
    rows_query = get_rows_for_heads(prune_q, hd)
    
    # Calculate affected rows for Key/Value heads (mapping Q heads to KV heads for GQA)
    group_size = max(1, n_q // n_kv)
    prune_kv = sorted(set(h // group_size for h in prune_q))
    rows_kv = get_rows_for_heads(prune_kv, hd)

    
    # Masking Query Projection (q_proj)
   
    weight_query = attn_mod.q_proj.weight
    mask_query = torch.ones(weight_query.shape[0], dtype=weight_query.dtype, device=weight_query.device)
    
    if rows_query:
        valid_rows_q = [r for r in rows_query if 0 <= r < weight_query.shape[0]]
        index_query = torch.tensor(valid_rows_q, dtype=torch.long, device=weight_query.device)
        
        if index_query.numel() > 0:
            mask_query.index_fill_(0, index_query, 0.0)
            weight_query.mul_(mask_query[:, None])
            
            bias_query = getattr(attn_mod.q_proj, "bias", None)
            if bias_query is not None:
                bias_query.mul_(mask_query.to(bias_query.dtype))

  
    # Masking Key and Value Projections (k_proj, v_proj)

    for projection_name, rows in (("k_proj", rows_kv), ("v_proj", rows_kv)):
        projection = getattr(attn_mod, projection_name, None)
        if projection is None or not hasattr(projection, "weight"):
            continue

        weight = projection.weight
        mask = torch.ones(weight.shape[0], dtype=weight.dtype, device=weight.device)
        
        if rows:
            valid_rows = [r for r in rows if 0 <= r < weight.shape[0]]
            index = torch.tensor(valid_rows, dtype=torch.long, device=weight.device)
            
            if index.numel() > 0:
                mask.index_fill_(0, index, 0.0)
                weight.mul_(mask[:, None])
                
                bias = getattr(projection, "bias", None)
                if bias is not None:
                    bias.mul_(mask.to(bias.dtype))

   
    # Masking Output Projection (o_proj)
  
    o_proj = getattr(attn_mod, "o_proj", None)
    if o_proj is not None and hasattr(o_proj, "weight"):
        weight_out = o_proj.weight
        mask_out = torch.ones(weight_out.shape[1], dtype=weight_out.dtype, device=weight_out.device)
        
        if rows_query:
            valid_rows_out = [r for r in rows_query if 0 <= r < weight_out.shape[1]]
            index_out = torch.tensor(valid_rows_out, dtype=torch.long, device=weight_out.device)
            
            if index_out.numel() > 0:
                mask_out.index_fill_(0, index_out, 0.0)
                weight_out.mul_(mask_out[None, :])


# FFN Pruning Utilities

@torch.no_grad()
def prune_ffn_channels_logical_mask(
    mlp_mod: nn.Module, 
    ratio: float, 
    intermediate_size: int
) -> None:
    """Prunes FFN channels using L1-norm magnitude pruning (logical masking).

    Calculates the L1 norm of the down_proj weights (exit valves) to identify
    the least active channels, and zeroes out the corresponding dimensions 
    across down_proj, up_proj, and gate_proj.

    Args:
        mlp_mod: The Multilayer Perceptron (MLP) module to prune.
        ratio: The fraction of channels to prune (0.0 to 1.0).
        intermediate_size: The total number of intermediate FFN channels.
    """
    n_prune = max(1, int(intermediate_size * ratio))
    if n_prune <= 0:
        return

    # 1. Identify the weakest channels using L1 norm on the down_proj weights
    down_weight = mlp_mod.down_proj.weight
    col_norms = torch.norm(down_weight, p=1, dim=0)
    prune_indices = torch.topk(col_norms, k=n_prune, largest=False).indices

    inter_dim = down_weight.shape[1]
    
    # Use the weight's native dtype (e.g., float16) to avoid casting issues during mul_
    keep_mask = torch.ones(inter_dim, dtype=down_weight.dtype, device=down_weight.device)

    if prune_indices.numel() > 0:
        keep_mask.index_fill_(0, prune_indices, 0.0)

    # 2. Apply mask to the exit projection (zeroing specific columns)
    mlp_mod.down_proj.weight.mul_(keep_mask[None, :])

    # 3. Apply mask to the entry projections (zeroing specific rows)
    for name in ["up_proj", "gate_proj"]:
        proj_layer = getattr(mlp_mod, name, None)
        
        if proj_layer is not None and hasattr(proj_layer, "weight"):
            proj_layer.weight.mul_(keep_mask[:, None])
            
            bias = getattr(proj_layer, "bias", None)
            if bias is not None:
                bias.mul_(keep_mask.to(bias.dtype))


# Tokenizer Configuration

# Ensure the tokenizer has a padding token defined; fallback to eos_token if missing
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"Set tokenizer.pad_token to eos_token: {processor.tokenizer.pad_token}")

# Standardized constants for data processing
PAD_TOKEN_ID = processor.tokenizer.pad_token_id
EOS_TOKEN_ID = processor.tokenizer.eos_token_id
IGNORE_INDEX = -100  # Standard PyTorch CrossEntropyLoss ignore index


# Dataset Preparation
DATA_ROOT = "./data_cifar10"
RESIZE_TO = 448

print(f"Loading CIFAR10 dataset from {DATA_ROOT}...")

# Download and load raw datasets
train_raw = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True)
test_raw = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

label_names = train_raw.classes
print(f"CIFAR10 Classes: {label_names}")

# Initialize image transformations
resize_tf = T.Resize((RESIZE_TO, RESIZE_TO))

# Dataset Generation and Formatting
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset

def make_examples(dataset: Dataset, n_take: int) -> List[Dict[str, Any]]:
    """Extracts and formats examples from the raw dataset.

    Args:
        dataset: The raw PyTorch vision dataset (e.g., CIFAR10).
        n_take: Number of examples to extract.

    Returns:
        A list of dictionaries containing the image, question, and formatted answers.
    """
    items = []
    # Ensure we use the string version of the EOS token for concatenation
    eos_token_str = processor.tokenizer.eos_token or ""
    
    for i in range(n_take):
        img_pil, y = dataset[i]
        img = resize_tf(img_pil)
        ans = label_names[y]
        
        items.append({
            "image": img,
            "question": "What is in this image?",
            "answer": ans,
            "answer_with_eos": ans + eos_token_str,
        })
    return items


print(f"Preparing {N_TRAIN} training examples and {N_VAL} validation examples...")
train_items = make_examples(train_raw, N_TRAIN)
val_items = make_examples(test_raw, N_VAL)

# Data Tokenization and Masking
def find_subsequence(full_seq: List[int], sub_seq: List[int]) -> int:
    """Finds the starting index of a subsequence within a larger sequence.
    
    Returns:
        The starting index, or -1 if the subsequence is not found.
    """
    len_full, len_sub = len(full_seq), len(sub_seq)
    if len_sub == 0 or len_sub > len_full:
        return -1
        
    for i in range(len_full - len_sub + 1):
        if full_seq[i:i + len_sub] == sub_seq:
            return i
    return -1


def mask_answer_only(input_ids_2d: torch.Tensor, answer_text: str) -> torch.Tensor:
    """Creates a label tensor that ignores the prompt and only supervises the answer.

    Args:
        input_ids_2d: The 2D tensor of input IDs from the processor.
        answer_text: The raw string of the expected answer.

    Returns:
        A tensor of labels where prompt tokens are set to IGNORE_INDEX (-100).
    """
    full_ids = input_ids_2d[0].tolist()
    
    # Tokenize the answer text in isolation
    ans_encoded = processor.tokenizer(
        answer_text, 
        add_special_tokens=False, 
        return_tensors="pt"
    )
    ans_ids = ans_encoded["input_ids"][0].tolist()

    # Initialize labels with the ignore index
    labels = input_ids_2d.clone()
    labels[:] = IGNORE_INDEX
    
    start_idx = find_subsequence(full_ids, ans_ids)
    
    if start_idx >= 0:
        end_idx = start_idx + len(ans_ids)
        labels[:, start_idx:end_idx] = input_ids_2d[:, start_idx:end_idx]
    else:
        # Fallback: supervise the last few tokens if exact span matching fails
        keep = min(8, input_ids_2d.shape[1])
        labels[:, -keep:] = input_ids_2d[:, -keep:]
        
    return labels


def encode_example_vqa(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Formats, templates, and tokenizes a single VQA example for the model.
    
    Args:
        ex: A dictionary containing the image, question, and answer.
        
    Returns:
        A dictionary of tensors ready for the PyTorch DataLoader.
    """
    messages_train = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "content": ex["image"]},
                {"type": "text", "content": ex["question"]}
            ]
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "content": ex["answer_with_eos"]}
            ]
        }
    ]
    
    messages_gen = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "content": ex["image"]},
                {"type": "text", "content": ex["question"]}
            ]
        }
    ]
    
    # Apply chat templates
    train_text = processor.apply_chat_template(
        messages_train, tokenize=False, add_generation_prompt=False
    )
    gen_text = processor.apply_chat_template(
        messages_gen, tokenize=False, add_generation_prompt=True
    )

    # Process texts and images into model inputs
    out = processor(
        text=[train_text],
        images=[ex["image"]],  # standard kwarg for vision processors is usually 'images'
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="longest",
        truncation=True,
    )

    input_ids = out["input_ids"]
    labels = mask_answer_only(input_ids, ex["answer_with_eos"])

    # Extract dynamic grid variables safely
    image_grid_thw = out.get("image_grid_thw", None)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.squeeze(0)

    return {
        "pixel_values": out["pixel_values"].squeeze(0),
        "image_grid_thw": image_grid_thw,
        "input_ids": input_ids.squeeze(0),
        "attention_mask": out["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0),
        # Extra metadata for evaluation
        "answer_text": ex["answer"],
        "raw_image": ex["image"],
        "gen_prompt": gen_text,
    }

print("Tokenizing training and validation examples...")
train_encoded = [encode_example_vqa(ex) for ex in train_items]
val_encoded = [encode_example_vqa(ex) for ex in val_items]

# PyTorch Dataset Wrapper
class CIFARVQADataset(Dataset):
    """PyTorch Dataset wrapper for pre-encoded CIFAR-10 VQA examples."""
    
    def __init__(self, encoded_data: List[Dict[str, Any]]):
        self.encoded = encoded_data
        
    def __len__(self) -> int:
        return len(self.encoded)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.encoded[idx]

train_ds = CIFARVQADataset(train_encoded)
val_ds = CIFARVQADataset(val_encoded)

def pad_1d(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pads a list of 1D tensors to the maximum length in the batch.

    Args:
        sequences: A list of 1D PyTorch tensors.
        pad_value: The value used to pad shorter sequences.

    Returns:
        A 2D tensor of shape (batch_size, max_length).
    """
    max_length = max(seq.size(0) for seq in sequences)
    
    # Pre-allocate padded tensor with the correct shape and data type
    out = torch.full((len(sequences), max_length), pad_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        out[i, :seq.size(0)] = seq
    return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of dataset examples into a batched dictionary."""
    out = {}
    
    # Image inputs
    out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)
    
    grids = [b["image_grid_thw"] for b in batch]
    out["image_grid_thw"] = torch.stack(grids, dim=0) if all(g is not None for g in grids) else None

    # Text inputs (padded to max sequence length in batch)
    out["input_ids"] = pad_1d([b["input_ids"] for b in batch], PAD_TOKEN_ID)
    out["attention_mask"] = pad_1d([b["attention_mask"] for b in batch], 0)
    out["labels"] = pad_1d([b["labels"] for b in batch], IGNORE_INDEX)

    # Raw metadata used for generation evaluation
    out["answer_text"] = [b["answer_text"] for b in batch]
    out["raw_images"] = [b["raw_image"] for b in batch]
    out["gen_prompts"] = [b["gen_prompt"] for b in batch]
    
    return out

print("Initializing DataLoaders...")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Evaluation Metrics
@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates the model's perplexity on a given validation loader.

    Args:
        model: The PyTorch model to evaluate.
        loader: The DataLoader containing validation data.

    Returns:
        The calculated perplexity (exponentiated token-level cross entropy).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in loader:
        moved_batch = {}
        
        # Move tensors to appropriate device and precision
        for key, val in batch.items():
            if key == "pixel_values" and val is not None:
                moved_batch[key] = val.to(DEVICE, dtype=TARGET_DTYPE)
            elif key in ("input_ids", "attention_mask", "labels") and val is not None:
                moved_batch[key] = val.to(DEVICE)
                
        if batch.get("image_grid_thw") is not None:
            moved_batch["image_grid_thw"] = batch["image_grid_thw"].to(DEVICE)
            
        outputs = model(**moved_batch)
        
        # Calculate loss weighted by the actual number of supervised tokens
        num_valid_tokens = (moved_batch["labels"] != IGNORE_INDEX).sum().item()
        num_valid_tokens = max(1, num_valid_tokens)  # Avoid division by zero
        
        total_loss += outputs.loss.item() * num_valid_tokens
        total_tokens += num_valid_tokens
        
    model.train()
    
    return math.exp(total_loss / max(1, total_tokens))


@torch.no_grad()
def eval_gen_accuracy(
    model: nn.Module, 
    processor: AutoProcessor, 
    loader: DataLoader, 
    k_samples: int = 50, 
    max_new_tokens: int = 3
) -> float:
    """Evaluates zero-shot generation accuracy on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        processor: The Hugging Face processor for formatting.
        loader: The DataLoader containing evaluation data.
        k_samples: Maximum number of samples to process before stopping.
        max_new_tokens: Maximum tokens to generate per answer.

    Returns:
        The accuracy score (0.0 to 1.0) based on substring matching.
    """
    model.eval()
    correct = 0
    seen = 0
    
    for batch in loader:
        for img, prompt, gold_answer in zip(batch["raw_images"], batch["gen_prompts"], batch["answer_text"]):
            if seen >= k_samples:
                break

            # Encode individual prompt/image pair
            inputs = processor(text=[prompt], images=[img], return_tensors="pt")
            
            # Cast inputs to device (and image to specific dtype to avoid fp32/fp16 conflicts)
            for key, val in inputs.items():
                if key == "pixel_values":
                    inputs[key] = val.to(DEVICE, dtype=TARGET_DTYPE)
                else:
                    inputs[key] = val.to(DEVICE)
                    
            # Generate response
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,  # Greedy search prevents visual beam expansion issues
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                length_penalty=2.0,
                use_cache=True,
            )
            
            # Slice out the prompt to isolate the newly generated tokens
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, prompt_length:]
            
            # Decode and format string
            decoded_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            cleaned_text = decoded_text.strip().lower()
            
            # Extract the first predicted word for robust classification matching
            prediction = cleaned_text.split()[0] if cleaned_text else ""
            
            if gold_answer.lower() in prediction:
                correct += 1
            seen += 1
            
        if seen >= k_samples:
            break
            
    model.train()
    return correct / max(1, seen)

# Baseline Evaluation & Pruning Execution
print("\n--- Running quick baseline eval (pre-prune) ---")
ppl_train = eval_loss(model, train_loader)
acc_val0 = eval_gen_accuracy(model, processor, val_loader, k_samples=40)
print(f"Train PPL (pre-prune): {ppl_train:.2f} | Val Gen@1 Acc (pre-prune): {acc_val0:.2%}")

print("\n--- Executing Pruning ---")
pruned_module_count = 0

if PRUNE_MODE == "attn_heads":
    for name, attn_module in find_attention_modules(model):
        prune_attention_heads_logical_gqa(attn_module, PRUNE_RATIO)
        pruned_module_count += 1
    print(f"Pruned heads in {pruned_module_count} attention modules (GQA-safe, mask-based).")
    
elif PRUNE_MODE == "ffn_channels":
    for name, mlp_module in find_mlp_modules(model):
        prune_ffn_channels_logical_mask(mlp_module, PRUNE_RATIO, intermediate_size)
        pruned_module_count += 1
    print(f"Pruned channels in {pruned_module_count} MLP modules (mask-based).")
    
else:
    raise ValueError("PRUNE_MODE must be either 'attn_heads' or 'ffn_channels'.")

ppl_after_prune = eval_loss(model, train_loader)
print(f"Train PPL (post-prune, pre-LoRA): {ppl_after_prune:.2f}")

# LoRA Configuration
def collect_lora_targets(model: nn.Module) -> List[str]:
    """Dynamically identifies linear layers suitable for LoRA adaptation.
    
    Args:
        model: The PyTorch model to inspect.
        
    Returns:
        A list of target module names for PEFT configuration.
    """
    target_names = set()
    valid_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for suffix in valid_suffixes:
                if name.endswith(suffix):
                    # Extract just the layer name (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
                    target_names.add(name.split(".")[-1])

    return sorted(list(target_names)) or valid_suffixes


targets = collect_lora_targets(model)
print(f"\nApplying LoRA to targets: {targets}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=targets,    
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optimizer and Scheduler Setup
# Filter out frozen parameters to save memory in the optimizer
trainable_params = (p for p in model.parameters() if p.requires_grad)
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
num_training_steps = EPOCHS * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, num_training_steps)

# Initialize mixed precision scaler
use_amp = (TARGET_DTYPE == torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Training Loop
print("\n--- Starting Training ---")
model.train()
global_step = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    for step, batch in enumerate(train_loader):
        moved_batch = {}
        
        # Move tensors to appropriate device and precision
        for key, val in batch.items():
            if key == "pixel_values" and val is not None:
                moved_batch[key] = val.to(DEVICE, dtype=TARGET_DTYPE)
            elif key in ("input_ids", "attention_mask", "labels") and val is not None:
                moved_batch[key] = val.to(DEVICE)

        if batch.get("image_grid_thw") is not None:
            moved_batch["image_grid_thw"] = batch["image_grid_thw"].to(DEVICE)

        # Forward pass with mixed precision
        with torch.autocast(device_type="cuda", dtype=TARGET_DTYPE, enabled=use_amp):
            outputs = model(**moved_batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation and optimization step
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            
            # set_to_none=True is slightly more memory efficient than zero_grad
            optimizer.zero_grad(set_to_none=True) 
            scheduler.step()
            global_step += 1
            
            # Simple progress logging
            if global_step % 10 == 0 or step == len(train_loader) - 1:
                print(f"  Step {global_step}/{num_training_steps} | Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")

# Clean up memory after training
torch.cuda.empty_cache()
print("\nTraining complete!")

# Final Evaluation and Model Saving
from PIL import Image

print("\n--- Running final evaluation (post-prune, post-LoRA) ---")
ppl_after_lora = eval_loss(model, train_loader)
acc_val1 = eval_gen_accuracy(model, processor, val_loader, k_samples=40)

print(
    f"Train PPL (post-prune, post-LoRA): {ppl_after_lora:.2f} | "
    f"Val Gen@1 Acc (post-prune, post-LoRA): {acc_val1:.2%}"
)

print("\n--- Saving Models ---")
lora_save_path = os.path.join(OUTPUT_DIR, "lora_pruned_qwen2vl")
merged_save_path = os.path.join(OUTPUT_DIR, "merged_pruned_qwen2vl")

# Save the LoRA adapter weights and processor
model.save_pretrained(lora_save_path)
processor.save_pretrained(lora_save_path)
print(f"Saved LoRA adapter to: {lora_save_path}")

# Merge LoRA weights back into the base model and save the full standalone model
print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_save_path)
processor.save_pretrained(merged_save_path)
print(f"Saved merged model to: {merged_save_path}")


# Inference and Demonstration Utilities
@torch.no_grad()
def qwen_vl_infer(
    infer_model: nn.Module, 
    infer_processor: AutoProcessor, 
    img_pil: Image.Image, 
    question: str, 
    max_new_tokens: int = 3
) -> str:
    """Runs a single zero-shot inference pass through the Qwen-VL model.

    Args:
        infer_model: The PyTorch model to use for inference.
        infer_processor: The Hugging Face processor.
        img_pil: The input image as a PIL Image.
        question: The text prompt/question.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated text string.
    """
    infer_model.eval()
    img_resized = img_pil.resize((RESIZE_TO, RESIZE_TO))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_resized},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    prompt = infer_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = infer_processor(text=[prompt], images=[img_resized], return_tensors="pt")

    # Move tensors to appropriate device and precision
    for key, val in inputs.items():
        if key == "pixel_values":
            inputs[key] = val.to(DEVICE, dtype=TARGET_DTYPE)
        elif isinstance(val, torch.Tensor):
            inputs[key] = val.to(DEVICE)

    generated_ids = infer_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=infer_processor.tokenizer.eos_token_id,
        eos_token_id=infer_processor.tokenizer.eos_token_id,
        repetition_penalty=1.5,
        length_penalty=2.0,
        use_cache=True
    )
    
    # Slice out the prompt to isolate the newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    
    text = infer_processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return text


def demo_val_predictions(model_to_use: nn.Module, k: int = 5) -> None:
    """Runs and prints predictions for a few samples from the raw validation set.

    Args:
        model_to_use: The trained/merged PyTorch model.
        k: The number of examples to evaluate.
    """
    print(f"\n--- Running Demo Inference on {k} Validation Samples ---")
    for i in range(k):
        img_pil, y = test_raw[i]
        gold_answer = label_names[y]
        
        prediction = qwen_vl_infer(
            model_to_use, 
            processor, 
            img_pil, 
            "What is in this image?"
        )
        
        print(f"GT: {gold_answer:<10s} | PRED: {prediction}")


# Trigger the demo function on the newly merged model
demo_val_predictions(merged_model, k=5)