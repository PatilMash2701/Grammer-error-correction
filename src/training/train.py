"""
Training loop for grammar correction models.

Supports:
- BART and GECToR model training
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (FP16) via torch.cuda.amp
- Wandb experiment tracking
- Checkpoint saving and early stopping
- Learning rate scheduling with warmup

Designed to run on Google Colab T4 GPU (16GB VRAM).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data.augmentation import augment_dataset
from src.data.preprocess import GECDataset, build_dataloader, create_data_splits, make_collate_fn
from src.models.llama_gec import LlamaGEC
from src.training.utils import (
    EarlyStopping,
    find_latest_step_checkpoint,
    get_lr_scheduler,
    save_checkpoint,
    save_step_checkpoint,
    set_seed,
)


def setup_training(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    model_type: str = "llama",
    dataset_path: str = "./data/raw/bea2019",
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    max_length: int = 256,
    gradient_accumulation_steps: int = 4,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    augmentation_factor: int = 2,
    seed: int = 42,
    output_dir: str = "./checkpoints",
    wandb_project: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set up all components needed for training Llama 3.2-3B.

    Loads model with LoRA, downloads and preprocesses data with augmentation,
    creates dataloaders, and initializes optimizer and scheduler.

    Args:
        model_name: Hugging Face model name or path.
        model_type: Model type ('llama', 't5', or 'bart').
        dataset_path: Path to dataset directory (CSV files: train.csv, dev.csv).
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate for AdamW.
        epochs: Number of training epochs.
        max_length: Maximum sequence length for generation.
        gradient_accumulation_steps: Steps to accumulate before update.
        use_lora: Whether to use LoRA adapters (highly recommended).
        lora_r: LoRA rank (16 is good balance of capacity/efficiency).
        lora_alpha: LoRA scaling factor.
        augmentation_factor: How many synthetic copies per sample (0 to disable).
        seed: Random seed for reproducibility.
        output_dir: Directory for saving checkpoints.
        wandb_project: Wandb project name (None to disable).

    Returns:
        Dictionary with all training components:
            - model: LlamaGEC with LoRA
            - tokenizer: Llama tokenizer
            - optimizer: AdamW optimizer
            - scheduler: Learning rate scheduler
            - train_loader: Training DataLoader
            - val_loader: Validation DataLoader
            - scaler: GradScaler for mixed precision
            - config: Training configuration dict

    Example:
        >>> components = setup_training(
        ...     model_name="meta-llama/Llama-3.2-3B-Instruct",
        ...     dataset_path="./data/raw/bea2019",
        ...     batch_size=8,
        ...     epochs=3,
        ... )
        >>> model = components['model']
        >>> train_loader = components['train_loader']
    """
    set_seed(seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if wandb_project is not None:
        try:
            import wandb

            wandb.init(
                project=wandb_project,
                config={
                    "model_name": model_name,
                    "model_type": model_type,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "max_length": max_length,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "use_lora": use_lora,
                    "lora_r": lora_r,
                    "augmentation_factor": augmentation_factor,
                    "seed": seed,
                },
            )
            logger.info(f"Wandb tracking enabled: {wandb_project}")
        except ImportError:
            logger.warning("Wandb not installed, skipping tracking")
            wandb_project = None

    # Load model
    logger.info(f"Loading model: {model_name}")
    if model_type.lower() == "llama":
        model = LlamaGEC.from_pretrained(
            model_name_or_path=model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            load_in_8bit=False,
            device=device,
            max_length=max_length,
            use_gradient_checkpointing=True,
        )
        tokenizer = model.tokenizer
    elif model_type.lower() == "t5":
        from src.models.t5_gec import T5GEC

        model = T5GEC.from_pretrained(
            model_name_or_path=model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            load_in_8bit=False,
            device=device,
            max_length=max_length,
        )
        tokenizer = model.tokenizer
    else:
        raise NotImplementedError(f"Model type '{model_type}' not yet supported in setup_training")

    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset_dir = Path(dataset_path)
    train_file = dataset_dir / "train.csv"
    dev_file = dataset_dir / "dev.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            f"Please run: python -m src.data.download --dataset bea2019"
        )

    train_df = pd.read_csv(train_file)
    train_data = train_df.to_dict("records")
    logger.info(f"Loaded {len(train_data)} training samples")

    # Apply data augmentation
    if augmentation_factor > 0:
        logger.info(f"Augmenting data with factor {augmentation_factor}...")
        train_data = augment_dataset(
            data=train_data,
            augmentation_factor=augmentation_factor,
            error_rate=0.15,
            seed=seed,
        )
        logger.info(f"After augmentation: {len(train_data)} training samples")

    # Load validation data
    if dev_file.exists():
        val_df = pd.read_csv(dev_file)
        val_data = val_df.to_dict("records")
        logger.info(f"Loaded {len(val_data)} validation samples")
    else:
        # If no dev file, split from training data
        logger.warning("No dev.csv found, splitting from training data")
        train_data, val_data, _ = create_data_splits(
            data=train_data,
            train_ratio=0.9,
            val_ratio=0.1,
            test_ratio=0.0,
            seed=seed,
        )
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")

    # Create datasets
    train_dataset = GECDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        model_type=model_type,
    )

    val_dataset = GECDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        model_type=model_type,
    )

    # Create dataloaders
    # For Llama, use dynamic padding: each batch is padded only to its own max
    # length, not the global max_length. This cuts attention compute ~2-3x for
    # BEA 2019 (avg sentence ~150 tokens vs 256 fixed padding).
    if model_type.lower() in ("llama", "llama32"):
        collate_fn = make_collate_fn(tokenizer.pad_token_id)
    else:
        collate_fn = None  # T5/BART use fixed-length tensors (already padded)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for Windows compatibility
        collate_fn=collate_fn,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Set up optimizer (only train LoRA parameters if use_lora=True)
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Training {len(trainable_params)} LoRA parameter groups")
    else:
        trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Set up learning rate scheduler
    num_training_steps = len(train_loader) * epochs // gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_type="linear",
    )

    # Mixed precision scaler
    # T5 is unstable with float16 → use bfloat16 (no scaler) for T5.
    # Llama is also designed for bfloat16 and should NOT use float16 scaler.
    # Only BART-type models benefit from the float16 GradScaler path.
    use_scaler = device == "cuda" and model_type not in ("t5", "llama")
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    # Training configuration
    config = {
        "model_name": model_name,
        "model_type": model_type,
        "device": device,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "max_length": max_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_lora": use_lora,
        "lora_r": lora_r,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "output_dir": output_dir,
        "wandb_project": wandb_project,
    }

    logger.info("✓ Training setup complete")
    logger.info(f"  Train samples: {len(train_data):,}")
    logger.info(f"  Val samples: {len(val_data):,}")
    logger.info(f"  Batch size: {batch_size} × {gradient_accumulation_steps} accumulation = {batch_size * gradient_accumulation_steps} effective")
    logger.info(f"  Training steps: {num_training_steps:,} ({num_warmup_steps} warmup)")
    logger.info(f"  Checkpoints: {output_dir}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "scaler": scaler,
        "config": config,
    }


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    save_steps: int = 0,
    step_checkpoint_dir: str = "",
    epoch: int = 1,
    start_batch: int = 0,
    optimizer_step_start: int = 0,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Implements gradient accumulation and optional mixed precision with bfloat16.

    Args:
        model: GEC model (LlamaGEC with LoRA).
        train_loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision (None for CPU/bfloat16).
        gradient_accumulation_steps: Steps before optimizer update.
        max_grad_norm: Maximum gradient norm for clipping.
        device: Device to train on.
        save_steps: Save a step checkpoint every N optimizer steps (0 = disabled).
        step_checkpoint_dir: Directory for step checkpoints.
        epoch: Current epoch number (1-indexed), used for checkpoint metadata.
        start_batch: Skip this many batches at the start (for mid-epoch resume).
        optimizer_step_start: Initial optimizer step count (for global step tracking).

    Returns:
        Dictionary with training metrics:
            - 'loss': Average training loss for the epoch.
            - 'lr': Current learning rate.
            - 'optimizer_step': Global optimizer step count at end of epoch.

    Example:
        >>> metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        >>> print(f"Train Loss: {metrics['loss']:.4f}")
    """
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer_step = optimizer_step_start

    optimizer.zero_grad()

    # Progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Skip batches already processed (mid-epoch resume)
        if batch_idx < start_batch:
            continue

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with mixed precision
        if scaler is not None:
            # Use float16 automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
        elif device == "cuda":
            # Use bfloat16 for Llama (better than fp16)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
        else:
            # CPU training (no mixed precision)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate loss
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Optimizer step every N batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

            # Mid-epoch step checkpoint
            if save_steps > 0 and step_checkpoint_dir and optimizer_step % save_steps == 0:
                current_loss = total_loss / max(num_batches, 1)
                save_step_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    optimizer_step=optimizer_step,
                    loss=current_loss,
                    checkpoint_dir=step_checkpoint_dir,
                )

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{total_loss / max(num_batches, 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": optimizer_step,
            }
        )

    avg_loss = total_loss / max(num_batches, 1)
    current_lr = scheduler.get_last_lr()[0]

    return {"loss": avg_loss, "lr": current_lr, "optimizer_step": optimizer_step}


def train(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    model_type: str = "llama",
    dataset_path: str = "./data/raw/bea2019",
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    save_steps: int = 2000,
    **kwargs,
) -> Path:
    """
    Full training pipeline: setup, train, validate, save.

    This is the main entry point for training a GEC model with LoRA.
    Automatically resumes from the latest step checkpoint if one exists.

    Args:
        model_name: Hugging Face model name.
        model_type: Model type ('llama', 't5', 'bart').
        dataset_path: Path to dataset directory.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        epochs: Number of training epochs.
        save_steps: Save a step checkpoint every N optimizer steps (default 2000,
            ~1.5 hours at the current training speed). Set 0 to disable.
        **kwargs: Additional arguments passed to setup_training.

    Returns:
        Path to the best model checkpoint.

    Example:
        >>> best_checkpoint = train(
        ...     model_name="meta-llama/Llama-3.2-3B-Instruct",
        ...     dataset_path="./data/raw/bea2019",
        ...     epochs=3,
        ... )
        >>> print(f"Best model saved to: {best_checkpoint}")
    """
    from src.training.evaluate import evaluate

    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Setup
    components = setup_training(
        model_name=model_name,
        model_type=model_type,
        dataset_path=dataset_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        **kwargs,
    )

    model = components["model"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]
    train_loader = components["train_loader"]
    val_loader = components["val_loader"]
    scaler = components["scaler"]
    config = components["config"]

    device = config["device"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    output_dir = config["output_dir"]
    wandb_project = config["wandb_project"]

    step_checkpoint_dir = str(Path(output_dir) / "step_checkpoints")

    # Resume from latest step checkpoint if available
    resume_epoch = 0      # 0-indexed; which epoch to start at
    resume_batch = 0      # which batch_idx to skip to within that epoch
    resume_step = 0       # global optimizer step count
    best_val_loss = float("inf")

    result = find_latest_step_checkpoint(step_checkpoint_dir)
    if result is not None:
        ckpt_path, meta = result
        resume_epoch = meta["epoch"] - 1   # convert 1-indexed → 0-indexed
        resume_batch = meta["batch_idx"] + 1  # start from the next batch
        resume_step = meta["optimizer_step"]
        logger.info(
            f"Resuming from step checkpoint: epoch={meta['epoch']}, "
            f"batch={meta['batch_idx']}, optimizer_step={resume_step}"
        )

        # Restore LoRA adapter weights
        lora_path = ckpt_path / "lora_adapter"
        adapter_file = lora_path / "adapter_model.safetensors"
        if adapter_file.exists():
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_file))
            model.load_state_dict(adapter_weights, strict=False)
            logger.info(f"LoRA adapter weights restored from {lora_path}")

        # Restore optimizer state
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(
                torch.load(str(opt_path), map_location=device)
            )
            logger.info("Optimizer state restored")

        # Restore scheduler state
        sched_path = ckpt_path / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(
                torch.load(str(sched_path), map_location=device)
            )
            logger.info("Scheduler state restored")

    # Early stopping
    early_stop = EarlyStopping(patience=2, min_delta=0.001)

    best_checkpoint_path = None
    optimizer_step = resume_step

    # Training loop
    for epoch in range(epochs):
        # Skip epochs already completed
        if epoch < resume_epoch:
            logger.info(f"Skipping epoch {epoch + 1} (already completed)")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"EPOCH {epoch + 1}/{epochs}")
        logger.info(f"{'=' * 60}")

        # For the resume epoch, skip already-processed batches
        start_batch = resume_batch if epoch == resume_epoch else 0
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch}")

        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=1.0,
            device=device,
            save_steps=save_steps,
            step_checkpoint_dir=step_checkpoint_dir,
            epoch=epoch + 1,
            start_batch=start_batch,
            optimizer_step_start=optimizer_step,
        )

        optimizer_step = train_metrics["optimizer_step"]

        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, LR: {train_metrics['lr']:.2e}"
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}")

        # Log to wandb
        if wandb_project is not None:
            try:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "learning_rate": train_metrics["lr"],
                    }
                )
            except (ImportError, Exception):
                pass

        # Save checkpoint if best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_metrics["loss"],
                output_dir=output_dir,
                filename="best_model.pt",
            )
            logger.info(f"✓ New best model saved: {best_checkpoint_path}")

        # Save regular checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=val_metrics["loss"],
            output_dir=output_dir,
            filename=f"checkpoint_epoch_{epoch + 1}.pt",
        )

        # Early stopping check
        early_stop(val_metrics["loss"])
        if early_stop.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Training complete
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    # Save final model (LoRA adapters only if using LoRA)
    if config.get("use_lora", True):
        # Save only LoRA weights
        final_path = Path(output_dir) / "llama_gec_lora"
        model.save_pretrained(str(final_path))
        logger.info(f"✓ Final LoRA adapters saved to: {final_path}")

    if wandb_project is not None:
        try:
            import wandb

            wandb.finish()
        except (ImportError, Exception):
            pass

    return best_checkpoint_path if best_checkpoint_path else Path(output_dir) / "best_model.pt"


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train GEC model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-base",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bea2019",
        choices=["bea2019", "lang8", "c4_200m"],
        help="Training dataset",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        dataset=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
