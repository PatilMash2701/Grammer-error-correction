"""Train Llama 3.2-3B-Instruct on BEA 2019 dataset with LoRA.

Configuration:
- Model: meta-llama/Llama-3.2-3B-Instruct (3B params) with LoRA (~24M trainable)
- Dataset: BEA 2019 (data/bea2019_hf) - 75K train, 4K dev
- Epochs: 3
- Batch size: 8 (effective: 32 with gradient accumulation x4)
- Learning rate: 2e-4
- Mixed precision: bfloat16  (Llama was designed for bfloat16, NOT float16)
- Max sequence length: 256   (BEA 2019 sentences avg ~30 tokens; 256 is ample)
- device_map: explicit cuda  (NOT "auto" — avoids LoRA-on-CPU slowdown)

Optimizations applied:
- SDPA (Scaled Dot-Product Attention) — PyTorch built-in FlashAttention kernels
- Dynamic padding — each batch padded to its own max length, not global max_length
- Gradient checkpointing — recompute activations to save ~40-50% VRAM
- Step-based checkpointing — crash recovery every 500 optimizer steps

Hardware requirements:
- GPU VRAM: 12-16 GB  (RTX 3080/4080/4090 etc.)
- System RAM: 32+ GB recommended

Checkpoint saved to: checkpoints/llama32_bea2019/llama_gec_lora/
"""

import os

from src.training.train import train

MODEL_PATH = os.getenv("LLAMA32_MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", None)

print("=" * 60)
print("TRAINING LLAMA 3.2-3B ON BEA 2019 DATASET")
print("=" * 60)
print("\nConfiguration:")
print(f"  Model:             {MODEL_PATH}")
print("  Training samples:  ~75,695")
print("  Validation samples: ~4,205")
print("  Epochs:            3")
print("  Batch size:        8 (effective: 32 with grad accumulation x4)")
print("  LoRA:              Enabled (r=16, alpha=32, 7 target modules)")
print("  Learning rate:     2e-4")
print("  Max seq length:    256")
print("  Mixed Precision:   bfloat16")
print("  Optimizations:     SDPA + dynamic padding + gradient checkpointing")
print("  Step checkpoints:  every 500 optimizer steps")
print("  Output dir:        checkpoints/llama32_bea2019/")
print("=" * 60)

best_checkpoint = train(
    model_name=MODEL_PATH,
    model_type="llama",
    dataset_path="./data/bea2019_hf",
    batch_size=8,
    learning_rate=2e-4,
    epochs=3,
    max_length=256,
    gradient_accumulation_steps=4,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    augmentation_factor=0,
    seed=42,
    output_dir="./checkpoints/llama32_bea2019",
    wandb_project=WANDB_PROJECT,
    save_steps=500,
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Best checkpoint:  {best_checkpoint}")
print("LoRA adapters:    checkpoints/llama32_bea2019/llama_gec_lora/")
print("\nNext step - run evaluation:")
print("  python evaluate_llama32.py")
print("=" * 60)
