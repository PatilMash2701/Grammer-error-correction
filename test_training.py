"""Quick training test with synthetic dataset."""

from src.training.train import train

print("=" * 60)
print("STARTING QUICK TRAINING TEST")
print("=" * 60)
print("Configuration:")
print("  - Model: FLAN-T5-Large (780M params)")
print("  - Dataset: 200 train samples, 15 val samples")
print("  - Epochs: 1")
print("  - Batch size: 4")
print("  - No augmentation")
print("  - Note: Using T5 while waiting for Llama access")
print("=" * 60)
print()

# Run quick test with FLAN-T5 (no auth needed)
best_checkpoint = train(
    model_name="google/flan-t5-large",  # 780M parameters, open access
    model_type="t5",
    dataset_path="./data/raw/bea2019_test",
    batch_size=4,
    learning_rate=5e-5,  # Slightly lower LR for T5
    epochs=1,
    max_length=256,
    gradient_accumulation_steps=2,  # Effective batch = 8
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    augmentation_factor=0,  # No augmentation for quick test
    seed=42,
    output_dir="./checkpoints/test_run_t5",
    wandb_project=None  # Disable wandb
)

print()
print("=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
print(f"Best checkpoint: {best_checkpoint}")
print()
print("Next steps:")
print("  1. Download BEA-2019 dataset")
print("  2. Run full training on ~27K samples")
print("  3. Evaluate on CoNLL-2014 and JFLEG")
