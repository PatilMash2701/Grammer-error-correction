"""Train FLAN-T5-Large on real BEA 2019 dataset (84K samples).

Configuration:
- Model: FLAN-T5-Large (780M params) with LoRA (4.7M trainable)
- Dataset: BEA 2019 (juancavallotti/bea-19-corruption) - 75K train, 4K dev
- Epochs: 3
- Learning rate: 1e-5 (safe for T5 with bfloat16)
- Mixed precision: bfloat16 (stable for T5)
"""

from src.training.train import train

print("=" * 60)
print("TRAINING FLAN-T5-LARGE ON BEA 2019 DATASET")
print("=" * 60)
print("\nConfiguration:")
print("  Model: FLAN-T5-Large (780M parameters)")
print("  Training samples: 75,695")
print("  Validation samples: 4,205")
print("  Epochs: 3")
print("  Batch size: 4 (effective: 8 with grad accumulation)")
print("  LoRA: Enabled (r=16, alpha=32)")
print("  Learning rate: 1e-5")
print("  Mixed Precision: bfloat16")
print("=" * 60)

best_checkpoint = train(
    model_name="google/flan-t5-large",
    model_type="t5",
    dataset_path="./data/bea2019_hf",
    batch_size=4,
    learning_rate=1e-5,
    epochs=3,
    max_length=256,
    gradient_accumulation_steps=2,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    augmentation_factor=0,
    seed=42,
    output_dir="./checkpoints/flan_t5_large_bea2019",
    wandb_project=None
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Best checkpoint: {best_checkpoint}")
print("=" * 60)
