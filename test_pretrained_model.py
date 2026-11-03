"""Evaluate pre-trained pszemraj/flan-t5-large-grammar-synthesis on BEA 2019"""
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.training.evaluate import calculate_f05_gleu

def main():
    print("\n" + "="*80)
    print("EVALUATING PRE-TRAINED MODEL: pszemraj/flan-t5-large-grammar-synthesis")
    print("="*80 + "\n")

    # Load model
    print("Loading model...")
    model_name = "pszemraj/flan-t5-large-grammar-synthesis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded on GPU\n")

    # Load BEA 2019 test set
    print("Loading BEA 2019 test set...")
    test_df = pd.read_csv("data/processed/bea19_dev.csv")
    print(f"✓ Loaded {len(test_df)} samples\n")

    # Run inference
    print("Running inference on all samples...")
    print("(This will take ~30-45 minutes)\n")

    start_time = time.time()
    predictions = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        source = row["source"]

        # Generate correction
        inputs = tokenizer(
            source,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)

    elapsed = time.time() - start_time
    print(f"\n✓ Inference completed in {elapsed/60:.1f} minutes\n")

    # Calculate metrics
    print("Calculating metrics...")
    references = test_df["target"].tolist()
    sources = test_df["source"].tolist()

    results = calculate_f05_gleu(
        predictions=predictions,
        references=references,
        sources=sources
    )

    # Save results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"F0.5 Score:       {results['f05']:.4f}")
    print(f"Precision:        {results['precision']:.4f}")
    print(f"Recall:           {results['recall']:.4f}")
    print(f"GLEU:             {results['gleu']:.4f}")
    print(f"TP:               {results['tp']}")
    print(f"FP:               {results['fp']}")
    print(f"FN:               {results['fn']}")
    print(f"Correction Rate:  {results['correction_rate']:.2%}")
    print("="*80 + "\n")

    # Save predictions
    output_df = test_df.copy()
    output_df["prediction"] = predictions
    output_path = "checkpoints/pretrained_pszemraj_evaluation.csv"
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")

    # Save metrics
    metrics_path = "checkpoints/pretrained_pszemraj_results.txt"
    with open(metrics_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("PRE-TRAINED MODEL: pszemraj/flan-t5-large-grammar-synthesis\n")
        f.write("Dataset: BEA 2019 Test Set\n")
        f.write("="*80 + "\n\n")
        f.write(f"F0.5 Score:       {results['f05']:.4f}\n")
        f.write(f"Precision:        {results['precision']:.4f}\n")
        f.write(f"Recall:           {results['recall']:.4f}\n")
        f.write(f"GLEU:             {results['gleu']:.4f}\n")
        f.write(f"TP:               {results['tp']}\n")
        f.write(f"FP:               {results['fp']}\n")
        f.write(f"FN:               {results['fn']}\n")
        f.write(f"Correction Rate:  {results['correction_rate']:.2%}\n")
        f.write(f"\nEvaluation time:  {elapsed/60:.1f} minutes\n")
        f.write(f"Timestamp:        {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✓ Metrics saved to: {metrics_path}\n")

    # Comparison
    print("="*80)
    print("COMPARISON TO YOUR FINE-TUNED MODEL")
    print("="*80)
    print(f"{'Model':<40} {'F0.5':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)
    print(f"{'Your FLAN-T5-Large (fine-tuned)':<40} {0.3201:<10.4f} {0.2744:<12.4f} {0.9588:<10.4f}")
    print(f"{'Pre-trained (pszemraj)':<40} {results['f05']:<10.4f} {results['precision']:<12.4f} {results['recall']:<10.4f}")
    print("="*80 + "\n")

    if results['f05'] > 0.32:
        improvement = ((results['f05'] - 0.3201) / 0.3201) * 100
        print(f"✅ PRE-TRAINED MODEL IS BETTER by {improvement:.1f}%!")
        print("   → You can use this model directly without retraining\n")
    else:
        print("❌ Pre-trained model is worse than your fine-tuned model")
        print("   → Recommended: Expand LoRA config and retrain your model\n")

if __name__ == "__main__":
    main()
