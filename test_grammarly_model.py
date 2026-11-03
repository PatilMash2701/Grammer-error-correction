"""Evaluate Grammarly CoEdIT-large on BEA 2019 test set (batched inference)"""
import math
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from src.training.evaluate import compute_f05, compute_gleu

BATCH_SIZE = 32
MAX_LENGTH = 256
NUM_BEAMS = 4


def batched_generate(model, tokenizer, prompts, batch_size=BATCH_SIZE):
    """Run batched inference on a list of prompts."""
    all_predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating", unit="batch"):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        ).to("cuda")

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded)

    return all_predictions


def main():
    print("\n" + "=" * 80)
    print("EVALUATING: Grammarly CoEdIT-large (Official Grammarly Model)")
    print("=" * 80 + "\n")

    # Load model
    print("Loading Grammarly CoEdIT-large model...")
    model_name = "grammarly/coedit-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[OK] Model loaded on GPU (770M parameters)\n")

    # Load BEA 2019 test set
    print("Loading BEA 2019 test set...")
    test_df = pd.read_csv("data/bea2019_hf/test.csv")
    print(f"[OK] Loaded {len(test_df)} samples\n")

    # Prepare prompts with CoEdIT instruction prefix
    sources = test_df["source"].tolist()
    prompts = [f"Fix grammatical errors in this sentence: {s}" for s in sources]

    # Run batched inference
    print(f"Running batched inference (batch_size={BATCH_SIZE}, num_beams={NUM_BEAMS})...")
    num_batches = math.ceil(len(prompts) / BATCH_SIZE)
    print(f"Total: {len(prompts)} samples in {num_batches} batches\n")

    start_time = time.time()
    predictions = batched_generate(model, tokenizer, prompts, batch_size=BATCH_SIZE)
    elapsed = time.time() - start_time

    throughput = len(predictions) / elapsed
    print(f"\n[OK] Inference completed in {elapsed / 60:.1f} minutes ({throughput:.1f} samples/sec)\n")

    # Calculate metrics
    print("Calculating metrics...")
    references = test_df["target"].tolist()

    # Convert references to List[List[str]] format for ERRANT
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
    results = {**f05_scores, "gleu": gleu_score}

    # Calculate correction rate
    num_corrected = sum(1 for p, s in zip(predictions, sources) if p != s)
    results["correction_rate"] = num_corrected / len(predictions)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - Grammarly CoEdIT-large")
    print("=" * 80)
    print(f"F0.5 Score:       {results['f05']:.4f}")
    print(f"Precision:        {results['precision']:.4f}")
    print(f"Recall:           {results['recall']:.4f}")
    print(f"GLEU:             {results['gleu']:.4f}")
    print(f"TP:               {results['tp']}")
    print(f"FP:               {results['fp']}")
    print(f"FN:               {results['fn']}")
    print(f"Correction Rate:  {results['correction_rate']:.2%}")
    print(f"Throughput:       {throughput:.1f} samples/sec")
    print("=" * 80 + "\n")

    # Save predictions
    output_df = test_df.copy()
    output_df["prediction"] = predictions
    output_path = "checkpoints/grammarly_coedit_evaluation.csv"
    output_df.to_csv(output_path, index=False)
    print(f"[OK] Predictions saved to: {output_path}")

    # Save metrics
    metrics_path = "checkpoints/grammarly_coedit_results.txt"
    with open(metrics_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL: Grammarly CoEdIT-large\n")
        f.write("Dataset: BEA 2019 Test Set\n")
        f.write(f"Batch Size: {BATCH_SIZE}, Num Beams: {NUM_BEAMS}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"F0.5 Score:       {results['f05']:.4f}\n")
        f.write(f"Precision:        {results['precision']:.4f}\n")
        f.write(f"Recall:           {results['recall']:.4f}\n")
        f.write(f"GLEU:             {results['gleu']:.4f}\n")
        f.write(f"TP:               {results['tp']}\n")
        f.write(f"FP:               {results['fp']}\n")
        f.write(f"FN:               {results['fn']}\n")
        f.write(f"Correction Rate:  {results['correction_rate']:.2%}\n")
        f.write(f"\nThroughput:       {throughput:.1f} samples/sec\n")
        f.write(f"Evaluation time:  {elapsed / 60:.1f} minutes\n")
        f.write(f"Timestamp:        {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[OK] Metrics saved to: {metrics_path}\n")

    # Comparison
    print("=" * 80)
    print("COMPARISON TO YOUR FINE-TUNED MODEL")
    print("=" * 80)
    print(f"{'Model':<45} {'F0.5':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)
    print(f"{'Your FLAN-T5-Large (fine-tuned)':<45} {0.3201:<10.4f} {0.2744:<12.4f} {0.9588:<10.4f}")
    print(f"{'Grammarly CoEdIT-large (pre-trained)':<45} {results['f05']:<10.4f} {results['precision']:<12.4f} {results['recall']:<10.4f}")
    print("=" * 80 + "\n")

    if results["f05"] > 0.40:
        improvement = ((results["f05"] - 0.3201) / 0.3201) * 100
        print(f"[GREAT] GRAMMARLY MODEL IS SIGNIFICANTLY BETTER by {improvement:.1f}%!")
        print("   -> Use this model directly without retraining\n")
    elif results["f05"] > 0.32:
        improvement = ((results["f05"] - 0.3201) / 0.3201) * 100
        print(f"[GOOD] Grammarly model is better by {improvement:.1f}%")
        print("   -> Consider using this model or fine-tuning it further\n")
    else:
        print("[INFO] Grammarly model is slightly worse")
        print("   -> Recommended: Expand LoRA config and retrain your model\n")


if __name__ == "__main__":
    main()
