"""Trains and profiles a ResNet18 model on the CIFAR-10 dataset.

This script demonstrates model setup, data loading, training/evaluation loops, 
and performance profiling using torch.profiler for both CPU and CUDA devices.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch import optim
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.profiler import schedule
from torch.profiler import tensorboard_trace_handler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms

# --- Constants ---
BATCH_SIZE = 128
NUM_WORKERS = 2
LOG_DIR = Path("logs/profiler")
PROFILE_STEPS = 15


def setup_log_directory(log_dir: Path) -> None:
    """Cleans up and recreates the logging directory.

    Args:
        log_dir: The path to the directory used for storing profiler logs.
    """
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Builds a modified ResNet18 model suited for 32x32 CIFAR-10 images.

    Args:
        num_classes: The number of output classes.

    Returns:
        A PyTorch nn.Module representing the modified ResNet18.
    """
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Modify the first convolutional layer for smaller 32x32 images
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove the maxpool layer as it reduces spatial dimensions too quickly
    model.maxpool = nn.Identity()
    return model


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Downloads and prepares CIFAR-10 train and test DataLoaders.

    Returns:
        A tuple containing the training and testing DataLoaders.
    """
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tfms
    )
    test_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tfms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    limit_steps: Optional[int] = None
) -> Tuple[float, float]:
    """Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        loader: The training data loader.
        optimizer: The optimizer used to update model weights.
        criterion: The loss function.
        device: The device to run computations on (CPU/CUDA).
        limit_steps: Optional maximum number of steps to run.

    Returns:
        A tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for step, (images, labels) in enumerate(loader):
        if limit_steps is not None and step >= limit_steps:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_steps: Optional[int] = None
) -> float:
    """Evaluates the model on the test dataset.

    Args:
        model: The neural network model to evaluate.
        loader: The test data loader.
        device: The device to run computations on (CPU/CUDA).
        limit_steps: Optional maximum number of steps to run.

    Returns:
        The accuracy of the model on the evaluated subset as a float (0 to 1).
    """
    model.eval()
    correct, total = 0, 0

    for step, (images, labels) in enumerate(loader):
        if limit_steps is not None and step >= limit_steps:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return correct / total


def profile_training(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    activities: list,
    log_dir: Path
) -> None:
    """Profiles the training loop and saves the trace to TensorBoard.

    Args:
        model: The neural network model to profile.
        loader: The data loader.
        optimizer: The optimizer used for weight updates.
        criterion: The loss function.
        device: The target execution device.
        activities: A list of ProfilerActivity to record.
        log_dir: Base directory to save TensorBoard logs.
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_logdir = str(log_dir / f"train_{run_timestamp}")

    capture_schedule = schedule(wait=1, warmup=1, active=5, repeat=1)

    model.train()
    step_iter = iter(loader)

    with profile(
        activities=activities,
        schedule=capture_schedule,
        on_trace_ready=tensorboard_trace_handler(train_logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        for _ in range(PROFILE_STEPS):
            try:
                images, labels = next(step_iter)
            except StopIteration:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prof.step()

    print(f"Training trace saved to: {train_logdir}")

    tb_cpu = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(f"\nTop ops by CPU self time:\n{tb_cpu}")

    if torch.cuda.is_available():
        tb_cuda = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(f"\nTop ops by CUDA self time:\n{tb_cuda}")

    cpu_total_us = sum(e.self_cpu_time_total for e in prof.key_averages())
    gpu_total_us = sum(getattr(e, "self_cuda_time_total", 0) for e in prof.key_averages())
    
    print(f"\nApprox CPU time (self): {cpu_total_us / 1e6:.3f}s")
    if torch.cuda.is_available():
        print(f"Approx CUDA time (self): {gpu_total_us / 1e6:.3f}s")


def profile_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    activities: list,
    log_dir: Path
) -> None:
    """Profiles the inference loop and saves the trace to TensorBoard.

    Args:
        model: The neural network model to profile.
        loader: The data loader.
        device: The target execution device.
        activities: A list of ProfilerActivity to record.
        log_dir: Base directory to save TensorBoard logs.
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    infer_logdir = str(log_dir / f"infer_{run_timestamp}")

    model.eval()
    step_iter = iter(loader)

    with torch.no_grad():
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(infer_logdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof_inf:
            for _ in range(12):
                try:
                    x, _ = next(step_iter)
                except StopIteration:
                    break

                x = x.to(device, non_blocking=True)
                _ = model(x)
                prof_inf.step()

    print(f"\nInference trace saved to: {infer_logdir}")

    tb_cpu_inf = prof_inf.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=10
    )
    print(f"\n[Inference] Top ops by CPU self time:\n{tb_cpu_inf}")

    if torch.cuda.is_available():
        tb_cuda_inf = prof_inf.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=10
        )
        print(f"\n[Inference] Top ops by CUDA self time:\n{tb_cuda_inf}")

    cpu_total_inf = sum(e.self_cpu_time_total for e in prof_inf.key_averages())
    gpu_total_inf = sum(getattr(e, "self_cuda_time_total", 0) for e in prof_inf.key_averages())
    
    print(f"\n[Inference] Approx CPU time (self): {cpu_total_inf / 1e6:.3f}s")
    if torch.cuda.is_available():
        print(f"[Inference] Approx CUDA time (self): {gpu_total_inf / 1e6:.3f}s")


def main() -> None:
    """Executes the setup, profiling, and training routines."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        torch.backends.cudnn.benchmark = True

    setup_log_directory(LOG_DIR)

    print("Preparing datasets...")
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = build_resnet18(num_classes=10).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model instantiated with {param_count:.2f} M params.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    print("\n--- Profiling Training ---")
    profile_training(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        activities=activities,
        log_dir=LOG_DIR
    )

    print("\n--- Profiling Inference ---")
    profile_inference(
        model=model,
        loader=test_loader,
        device=device,
        activities=activities,
        log_dir=LOG_DIR
    )

    print("\n--- Running Mini Training Loop ---")
    loss, acc = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        limit_steps=200
    )
    scheduler.step()

    val_acc = evaluate(
        model=model, 
        loader=test_loader, 
        device=device, 
        limit_steps=80
    )
    
    print(f"Train loss ~{loss:.3f} | Train acc ~{acc * 100:.1f}% | Val acc (subset) ~{val_acc * 100:.1f}%")


if __name__ == "__main__":
    main()