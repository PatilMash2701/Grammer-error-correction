"""Fine-tune Grammarly CoEdIT-Large on BEA 2019 for focused GEC.

OPTIMIZED for maximum GPU utilization on RTX 4080 SUPER (16GB):
- Length-sorted batching (no wasted padding, consistent batch times)
- Dynamic padding to longest in batch
- MAX_LENGTH=128 (covers 95% of data, halves compute vs 256)
- BATCH_SIZE=64 (short seqs = small memory footprint)
- bfloat16 mixed precision
- LoRA on q,v,k only (~15M params, fast convergence)
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─── Configuration ───────────────────────────────────────────────────
MODEL_NAME = "grammarly/coedit-large"
TASK_PREFIX = "Fix grammatical errors in this sentence: "
DATASET_PATH = "./data/bea2019_hf"
OUTPUT_DIR = "./checkpoints/coedit_large_bea2019"

BATCH_SIZE = 8              # Keep activations within 16GB VRAM
GRADIENT_ACCUMULATION = 8   # Effective batch = 8*8 = 64
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 128            # 95% of data fits; halves compute vs 256
SEED = 42

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "v", "k"]  # Attention only, fast + effective

MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10
# ─────────────────────────────────────────────────────────────────────


class GECDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.data = data
        self.max_length = max_length
        # Pre-compute approximate lengths for sorting (char-based, fast)
        self.lengths = [len(d["source"]) for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "source": f"{TASK_PREFIX}{item['source']}",
            "target": item["target"],
        }


class LengthSortedSampler(Sampler):
    """Sort by length with slight randomization for training variety.

    Groups similar-length sentences into batches. Within each mega-batch
    (pool_size batches), samples are sorted by length. Mega-batches
    themselves are shuffled for randomness.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Sort indices by length
        indices = np.argsort(self.lengths)

        # Split into mega-batches (groups of 8 batches)
        pool_size = self.batch_size * 8
        pools = [indices[i:i + pool_size] for i in range(0, len(indices), pool_size)]

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(pools)
            self.epoch += 1

        for pool in pools:
            yield from pool

    def __len__(self):
        return len(self.lengths)


class DynamicPadCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sources = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]

        source_enc = self.tokenizer(
            sources, max_length=self.max_length, truncation=True,
            padding="longest", return_tensors="pt",
        )
        target_enc = self.tokenizer(
            targets, max_length=self.max_length, truncation=True,
            padding="longest", return_tensors="pt",
        )

        labels = target_enc["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": labels,
        }


def get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, grad_accum, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Training", leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum

        loss.backward()
        total_loss += loss.item() * grad_accum
        num_batches += 1

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if num_batches % 50 == 0:
            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                toks=f"{input_ids.shape[1]}",
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="  Validating", leave=False, dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += outputs.loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution kernels
    device = "cuda"

    print("\n" + "=" * 70)
    print("FINE-TUNING: Grammarly CoEdIT-Large on BEA 2019")
    print("=" * 70)
    print(f"  Model:          {MODEL_NAME} (770M params)")
    print(f"  LoRA:           r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
    print(f"  Batch:          {BATCH_SIZE} x {GRADIENT_ACCUMULATION} accum = {BATCH_SIZE*GRADIENT_ACCUMULATION} effective")
    print(f"  Max length:     {MAX_LENGTH} tokens")
    print(f"  LR:             {LEARNING_RATE} (cosine, {WARMUP_RATIO*100:.0f}% warmup)")
    print(f"  Precision:      bfloat16")
    print(f"  Batching:       Length-sorted (minimal padding waste)")
    print(f"  Epochs:         {EPOCHS}")
    print("=" * 70 + "\n")

    # ── Load model ──────────────────────────────────────────────────
    logger.info("Loading CoEdIT-Large...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
    ).to(device)

    # ── Apply LoRA ──────────────────────────────────────────────────
    logger.info(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to reduce VRAM (trades compute for memory)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # ── Load data ───────────────────────────────────────────────────
    logger.info("Loading BEA 2019 dataset...")
    train_df = pd.read_csv(Path(DATASET_PATH) / "train.csv")
    val_df = pd.read_csv(Path(DATASET_PATH) / "dev.csv")
    train_data = train_df.to_dict("records")
    val_data = val_df.to_dict("records")
    logger.info(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    train_dataset = GECDataset(train_data, MAX_LENGTH)
    val_dataset = GECDataset(val_data, MAX_LENGTH)
    collator = DynamicPadCollator(tokenizer, MAX_LENGTH)

    # Length-sorted sampler for efficient batching
    train_sampler = LengthSortedSampler(
        train_dataset.lengths, BATCH_SIZE, shuffle=True, seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0, pin_memory=True,
        collate_fn=collator, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=collator,
    )

    # ── Optimizer + Scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01,
    )

    num_training_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
    scheduler = get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps)

    logger.info(f"Training steps: {num_training_steps:,} ({num_warmup_steps} warmup)")
    logger.info(f"Batches per epoch: {len(train_loader):,}")

    # ── CUDA warmup ─────────────────────────────────────────────────
    logger.info("CUDA warmup (first forward pass)...")
    dummy = next(iter(train_loader))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(
            input_ids=dummy["input_ids"].to(device),
            attention_mask=dummy["attention_mask"].to(device),
            labels=dummy["labels"].to(device),
        )
    torch.cuda.synchronize()
    logger.info(f"GPU memory after warmup: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # ── Training loop ───────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\n{'-' * 70}")
        print(f"  EPOCH {epoch + 1}/{EPOCHS}")
        print(f"{'-' * 70}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            GRADIENT_ACCUMULATION, device,
        )

        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        samples_per_sec = len(train_data) / epoch_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time:       {epoch_time / 60:.1f} min ({samples_per_sec:.0f} samples/sec)")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB peak")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(OUTPUT_DIR) / "coedit_gec_lora"
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            print(f"  >> NEW BEST MODEL saved to {save_path}")
        else:
            print(f"  (no improvement, best={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time:     {total_time / 60:.1f} minutes")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Saved to:       {OUTPUT_DIR}/coedit_gec_lora")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
