"""Performs Quantization-Aware Training (QAT) on a TinyLlama model."""

import math
import os
import shutil
import time
from typing import Any, Callable, Dict, Tuple

from datasets import Dataset, load_dataset
import evaluate
from neural_compressor import QuantizationAwareTrainingConfig
import numpy
from optimum.intel import INCModelForCausalLM, INCTrainer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    EvalPrediction,
    TrainingArguments,
)

_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_PROMPT = (
    "Deep-sea exploration technologies are unlocking the mysteries of the "
    "abyssal plains, revealing undiscovered ecosystems and vital insights "
    "into the global carbon cycle's regulatory mechanisms."
)

_PERP_TEXT = (
    "Coral reefs are diverse underwater ecosystems held together by calcium "
    "carbonate structures secreted by corals. Reefs are built by colonies of "
    "tiny animals found in marine water that contain few nutrients. Most coral "
    "reefs are built from stony corals, whose polyps cluster in groups."
)

_MAX_NEW_TOKENS = 50
_OUTPUT_DIR = "qat_tinyllama"

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """Loads the causal language model and its tokenizer.

    Args:
        model_name: The Hugging Face hub path or local path to the model.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def prepare_datasets(
    tokenizer: Any, 
    block_size: int = 8, 
    train_size: int = 1000,
    eval_size: int = 500
) -> Tuple[Dataset, Dataset]:
    """Prepares and tokenizes the train and evaluation datasets.

    Args:
        tokenizer: The tokenizer to process the text.
        block_size: The maximum sequence length.
        train_size: The number of samples for the training set.
        eval_size: The number of samples for the evaluation set.

    Returns:
        A tuple containing the training and evaluation Dataset objects.
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def _tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        out = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=block_size,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = raw.map(
        _tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    train_ids = tokenized["train"].select(range(train_size))
    eval_ids = tokenized["validation"].select(range(eval_size))
    
    return train_ids, eval_ids

def create_inc_trainer(
    model: Any, 
    tokenizer: Any, 
    train_dataset: Dataset, 
    quant_config: QuantizationAwareTrainingConfig, 
    output_dir: str
) -> INCTrainer:
    """Creates the optimum-intel INC Trainer for Quantization Aware Training.

    Args:
        model: The model to train.
        tokenizer: The tokenizer corresponding to the model.
        train_dataset: The dataset used for training.
        quant_config: The quantization configuration.
        output_dir: The directory to save training outputs.

    Returns:
        An instantiated INCTrainer object.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        optim="adamw_bnb_8bit",
        evaluation_strategy="no",
        save_strategy="no", 
        num_train_epochs=1,
        logging_steps=50,
        save_total_limit=1,
    )
    
    trainer = INCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        quantization_config=quant_config
    )
    
    return trainer

def make_compute_ppl_function(
    pad_token_id: int
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """Creates a metrics computation function for perplexity.

    Args:
        pad_token_id: The token ID used for padding to ignore in loss calc.

    Returns:
        A callable function that computes perplexity from EvalPredictions.
    """
    def compute_ppl(pred: EvalPrediction) -> Dict[str, float]:
        logits = pred.predictions
        labels = pred.label_ids
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.reshape(-1)
        
        logits_tensor = torch.from_numpy(flat_logits)
        labels_tensor = torch.from_numpy(flat_labels)
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fct(logits_tensor, labels_tensor)
        
        ppl = torch.exp(loss)
        return {"perplexity": ppl.item()}
        
    return compute_ppl

def run_qat_and_evaluate(
    trainer: INCTrainer, 
    eval_ds: Dataset, 
    tokenizer: Any
) -> Dict[str, float]:
    """Runs the training loop and evaluates the final model.

    Args:
        trainer: The instantiated INCTrainer object.
        eval_ds: The evaluation dataset.
        tokenizer: The model's tokenizer.

    Returns:
        A dictionary containing the evaluation metrics.
    """
    trainer.train()
    small_eval = eval_ds.select(range(min(len(eval_ds), 100)))

    ppl_fn = make_compute_ppl_function(tokenizer.pad_token_id)

    eval_args = TrainingArguments(
        output_dir=trainer.args.output_dir,
        per_device_eval_batch_size=1,
        fp16=True,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
    )
    
    eval_trainer = INCTrainer(
        model=trainer.model,
        args=eval_args,
        eval_dataset=small_eval,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=ppl_fn,
    )
    
    return eval_trainer.evaluate()

def save_and_load_qat_model(trainer: INCTrainer, output_dir: str) -> Any:
    """Saves the QAT model to disk and reloads it using INCModelForCausalLM.

    Args:
        trainer: The trainer containing the optimized model.
        output_dir: The directory to save the model weights.

    Returns:
        The loaded quantized causal LM model.
    """
    trainer.save_model(output_dir)
    model = INCModelForCausalLM.from_pretrained(output_dir)
    return model

def measure_latency_and_throughput(
    model: Any, 
    tokenizer: Any, 
    prompt: str, 
    max_new_tokens: int = 50, 
    num_runs: int = 10
) -> Tuple[float, float]:
    """Measures model inference latency and throughput.

    Args:
        model: The language model to benchmark.
        tokenizer: The associated tokenizer.
        prompt: The text prompt to generate from.
        max_new_tokens: The maximum number of tokens to generate.
        num_runs: The number of benchmark iterations to run.

    Returns:
        A tuple containing average latency (seconds) and throughput (tok/sec).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Warm-up phase
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Benchmark phase
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
            end_time = time.time()
            latencies.append(end_time - start_time)

    avg_latency = sum(latencies) / len(latencies)
    throughput = 1.0 / avg_latency if avg_latency > 0 else float("inf")

    return avg_latency, throughput

def measure_peak_mb_and_perplexity(
    model: Any, 
    tokenizer: Any, 
    text: str, 
    device: torch.device
) -> Tuple[float, float]:
    """Measures peak GPU memory consumption and perplexity for a given text.

    Args:
        model: The language model to benchmark.
        tokenizer: The associated tokenizer.
        text: The text to evaluate.
        device: The device to perform the evaluation on.

    Returns:
        A tuple containing peak memory in MB and the text's perplexity score.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_bytes / (1024 ** 2)
    else:
        peak_memory_mb = 0.0
        
    perplexity = math.exp(outputs.loss.item())
    
    return peak_memory_mb, perplexity

def main() -> None:
    """Executes dataset preparation, QAT, evaluation, and benchmarking."""
    model, tokenizer = load_model_and_tokenizer(_MODEL_NAME)
    train_ds, eval_ds = prepare_datasets(tokenizer)

    quant_config = QuantizationAwareTrainingConfig()  

    qat_trainer = create_inc_trainer(
        model, 
        tokenizer, 
        train_ds, 
        quant_config, 
        _OUTPUT_DIR
    )
    
    metrics = run_qat_and_evaluate(qat_trainer, eval_ds, tokenizer)
    print(f"Final perplexity: {metrics['eval_perplexity']:.2f}")

    qat_model = save_and_load_qat_model(qat_trainer, _OUTPUT_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qat_model.to(device)
    
    latency, throughput = measure_latency_and_throughput(
        qat_model, tokenizer, _PROMPT, _MAX_NEW_TOKENS
    )
    peak_mem_mb, perp = measure_peak_mb_and_perplexity(
        qat_model, tokenizer, _PERP_TEXT, device
    )
    
    print("\n--- Benchmark Results ---")
    print(f"Latency:        {latency:.4f} seconds")
    print(f"Throughput:     {throughput:.2f} tokens/second")
    print(f"Peak VRAM:      {peak_mem_mb:.2f} MB")
    print(f"Text PPL:       {perp:.2f}")


if __name__ == "__main__":
    main()