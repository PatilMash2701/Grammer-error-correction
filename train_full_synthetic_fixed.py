"""Full training run on synthetic GEC dataset - FIXED for NaN issue.

This trains FLAN-T5-Large on the synthetic dataset with:
- 6,998 training samples
- 712 validation samples
- 5 epochs
- LoRA for efficient training
- **FIXED**: Lower learning rate (1e-5) to prevent NaN with T5
- **FIXED**: Uses bfloat16 instead of float16 for stability
"""

from src.training.train import train

print("=" * 60)
print("FULL TRAINING RUN - SYNTHETIC GEC DATASET (FIXED)")
print("=" * 60)
print("\nConfiguration:")
print("  Model: FLAN-T5-Large (780M parameters)")
print("  Training samples: 6,998")
print("  Validation samples: 712")
print("  Epochs: 5")
print("  Batch size: 4 (effective: 8 with grad accumulation)")
print("  LoRA: Enabled (r=16, alpha=32)")
print("  Learning rate: 1e-5 (REDUCED from 5e-5 to prevent NaN)")
print("  Mixed Precision: bfloat16 (more stable than float16 for T5)")
print("=" * 60)

# Train the model with FIXED settings
best_checkpoint = train(
    model_name="google/flan-t5-large",
    model_type="t5",
    dataset_path="./data/synthetic_gec_csv",
    batch_size=4,
    learning_rate=1e-5,  # REDUCED: 5e-5 -> 1e-5 to prevent NaN
    epochs=5,
    max_length=256,
    gradient_accumulation_steps=2,  # Effective batch = 8
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    augmentation_factor=0,  # No augmentation - dataset already has variety
    seed=42,
    output_dir="./checkpoints/full_training_synthetic_fixed",
    wandb_project=None  # Set to your wandb project name if you want logging
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Best checkpoint: {best_checkpoint}")
print("\nNext steps:")
print("  1. Check that loss converged (not NaN)")
print("  2. Test inference with: python test_inference_full.py")
print("  3. Evaluate on test set: python test_evaluation.py")
print("=" * 60)
