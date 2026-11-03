"""Evaluate Grammarly CoEdIT-large on BEA 2019 with BATCHED inference for GPU optimization"""
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from src.training.evaluate import compute_f05, compute_gleu

class GrammarDataset(Dataset):
    """Simple dataset for batched inference"""
    def __init__(self, sources):
        self.sources = sources

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx]

def collate_fn(batch, tokenizer):
    """Collate function to batch and pad inputs"""
    # Add instruction prefix to each sentence
    prompts = [f"Fix grammatical errors in this sentence: {src}" for src in batch]

    # Tokenize and pad
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    return encoded

def main():
    print("\n" + "="*80)
    print("EVALUATING: Grammarly CoEdIT-large (BATCHED INFERENCE - OPTIMIZED)")
    print("="*80 + "\n")

    # Load model
    print("Loading Grammarly CoEdIT-large model...")
    model_name = "grammarly/coedit-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"[OK] Model loaded on GPU (770M parameters)\n")

    # Load BEA 2019 test set
    print("Loading BEA 2019 test set...")
    test_df = pd.read_csv("data/bea2019_hf/test.csv")
    print(f"[OK] Loaded {len(test_df)} samples\n")

    # Create dataset and dataloader for batched inference
    sources = test_df["source"].tolist()
    dataset = GrammarDataset(sources)

    # Batch size 16 for optimal GPU utilization
    batch_size = 16
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0  # Windows compatibility
    )

    # Run batched inference
    print(f"Running BATCHED inference (batch_size={batch_size})...")
    print(f"Total batches: {len(dataloader)}")
    print("(This will take ~25-40 minutes with 80-95% GPU utilization)\n")

    start_time = time.time()
    predictions = []

    with torch.no_grad():
        for batch_inputs in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            # Move batch to GPU
            batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}

            # Generate corrections for entire batch
            outputs = model.generate(
                **batch_inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

            # Decode all predictions in batch
            batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_predictions)

    elapsed = time.time() - start_time
    print(f"\n[OK] Inference completed in {elapsed/60:.1f} minutes")
    print(f"[OK] Average speed: {len(predictions) / elapsed:.1f} samples/sec\n")

    # Calculate metrics
    print("Calculating metrics...")
    references = test_df["target"].tolist()
    refs_for_f05 = [[r] for r in references]

    print("  Computing F0.5...")
    f05_scores = compute_f05(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )

    print("  Computing GLEU...")
    gleu_score = compute_gleu(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )

    # Combine results
    results = {**f05_scores, 'gleu': gleu_score}

    # Calculate correction rate
    num_corrected = sum(1 for p, s in zip(predictions, sources) if p != s)
    results['correction_rate'] = num_corrected / len(predictions)

    # Save results
    print("\n" + "="*80)
    print("RESULTS - Grammarly CoEdIT-large")
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
    output_path = "checkpoints/grammarly_coedit_evaluation.csv"
    output_df.to_csv(output_path, index=False)
    print(f"[OK] Predictions saved to: {output_path}")

    # Save metrics
    metrics_path = "checkpoints/grammarly_coedit_results.txt"
    with open(metrics_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("MODEL: Grammarly CoEdIT-large (Batched Inference)\n")
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

    print(f"[OK] Metrics saved to: {metrics_path}\n")

    # Comparison
    print("="*80)
    print("COMPARISON TO YOUR FINE-TUNED MODEL")
    print("="*80)
    print(f"{'Model':<45} {'F0.5':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*80)
    print(f"{'Your FLAN-T5-Large (fine-tuned)':<45} {0.3201:<10.4f} {0.2744:<12.4f} {0.9588:<10.4f}")
    print(f"{'Grammarly CoEdIT-large (pre-trained)':<45} {results['f05']:<10.4f} {results['precision']:<12.4f} {results['recall']:<10.4f}")
    print("="*80 + "\n")

    if results['f05'] > 0.40:
        improvement = ((results['f05'] - 0.3201) / 0.3201) * 100
        print(f"[GREAT] GRAMMARLY MODEL IS SIGNIFICANTLY BETTER by {improvement:.1f}%!")
        print("   -> Use this model directly without retraining\n")
    elif results['f05'] > 0.32:
        improvement = ((results['f05'] - 0.3201) / 0.3201) * 100
        print(f"[GOOD] Grammarly model is better by {improvement:.1f}%")
        print("   -> Consider using this model or fine-tuning it further\n")
    else:
        print("[INFO] Grammarly model is slightly worse")
        print("   -> Recommended: Expand LoRA config and retrain your model\n")

if __name__ == "__main__":
    main()
