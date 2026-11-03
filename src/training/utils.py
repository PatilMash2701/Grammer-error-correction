"""
Training utilities: seed, scheduling, checkpointing, early stopping.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seed for: random, numpy, torch (CPU + CUDA).

    Args:
        seed: Integer seed value.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "linear",
) -> LambdaLR:
    """
    Create a learning rate scheduler with warmup.

    Supports linear warmup followed by linear/cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        scheduler_type: Type of decay ('linear' or 'cosine').

    Returns:
        LambdaLR scheduler instance.

    Example:
        >>> scheduler = get_lr_scheduler(
        ...     optimizer, num_warmup_steps=500, num_training_steps=10000
        ... )
        >>> # In training loop:
        >>> scheduler.step()
    """
    import math

    def lr_lambda(current_step: int) -> float:
        """Compute learning rate multiplier for current step."""
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # After warmup: linear or cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        if scheduler_type == "linear":
            # Linear decay from 1.0 to 0.0
            return max(0.0, 1.0 - progress)

        elif scheduler_type == "cosine":
            # Cosine decay from 1.0 to 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Created {scheduler_type} LR scheduler: "
        f"{num_warmup_steps} warmup steps, "
        f"{num_training_steps} total steps"
    )

    return scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    output_dir: str = "./checkpoints",
    filename: Optional[str] = None,
) -> Path:
    """
    Save model checkpoint to disk.

    Saves model state dict, optimizer state, epoch, and loss.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch number.
        loss: Current loss value.
        output_dir: Directory to save checkpoint.
        filename: Checkpoint filename (auto-generated if None).

    Returns:
        Path to the saved checkpoint file.

    Example:
        >>> path = save_checkpoint(model, optimizer, epoch=5, loss=0.123)
        >>> print(f"Checkpoint saved to: {path}")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"

    filepath = output_path / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )

    logger.info(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a model checkpoint from disk.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to the checkpoint file.
        optimizer: Optimizer to restore state (optional).
        device: Device to map checkpoint tensors to.

    Returns:
        Dictionary with 'epoch' and 'loss' from checkpoint.

    Example:
        >>> info = load_checkpoint(model, "./checkpoints/best_model.pt")
        >>> print(f"Loaded from epoch {info['epoch']}, loss={info['loss']:.4f}")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
        f"loss={checkpoint['loss']:.4f}"
    )

    return {"epoch": checkpoint["epoch"], "loss": checkpoint["loss"]}


def save_step_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    batch_idx: int,
    optimizer_step: int,
    loss: float,
    checkpoint_dir: str,
) -> Path:
    """
    Save a mid-epoch step checkpoint for crash recovery.

    Saves LoRA adapter weights (via PEFT save_pretrained), optimizer state,
    scheduler state, and a meta.json with training position.

    Args:
        model: The model (expected to be a PEFT model with save_pretrained).
        optimizer: Optimizer to save state for.
        scheduler: LR scheduler to save state for.
        epoch: Current epoch (1-indexed).
        batch_idx: Current batch index within the epoch.
        optimizer_step: Global optimizer step count.
        loss: Current running loss.
        checkpoint_dir: Root directory for step checkpoints.

    Returns:
        Path to the step checkpoint directory.
    """
    step_dir = Path(checkpoint_dir) / f"step_{optimizer_step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights (PEFT format, ~50MB)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(step_dir / "lora_adapter"))
    else:
        torch.save(model.state_dict(), step_dir / "model_state.pt")

    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")

    # Save meta information for resume
    meta = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "optimizer_step": optimizer_step,
        "loss": loss,
    }
    with open(step_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Step checkpoint saved: step={optimizer_step}, "
        f"epoch={epoch}, batch={batch_idx}, loss={loss:.4f} → {step_dir}"
    )
    return step_dir


def find_latest_step_checkpoint(checkpoint_dir: str) -> Optional[Tuple[Path, Dict]]:
    """
    Find the most recent step checkpoint in a directory.

    Scans for subdirectories named 'step_XXXXXXX', picks the highest
    optimizer step number, and returns its path with metadata.

    Args:
        checkpoint_dir: Root directory containing step_XXXXXXX subdirs.

    Returns:
        (checkpoint_path, meta_dict) if found, else None.
    """
    root = Path(checkpoint_dir)
    if not root.exists():
        return None

    step_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not step_dirs:
        return None

    latest = step_dirs[-1]
    meta_file = latest / "meta.json"
    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    logger.info(
        f"Found step checkpoint: {latest.name} "
        f"(epoch={meta['epoch']}, batch={meta['batch_idx']}, "
        f"optimizer_step={meta['optimizer_step']})"
    )
    return latest, meta


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Monitors validation loss and stops training if it doesn't improve
    for a specified number of consecutive epochs (patience).

    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        best_loss: Best validation loss seen so far.
        counter: Number of epochs without improvement.
        should_stop: Whether training should be stopped.

    Example:
        >>> early_stop = EarlyStopping(patience=3, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = evaluate(model, val_loader)['loss']
        ...     early_stop(val_loss)
        ...     if early_stop.should_stop:
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping.
            min_delta: Minimum loss decrease to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        """
        Check whether training should stop.

        Args:
            val_loss: Current validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best loss: {self.best_loss:.4f}"
                )
