"""Evaluate trained Llama 3.2-3B-Instruct + LoRA on BEA 2019 test set.

Runs inference on the test split and computes:
- F0.5 score (ERRANT-based, CoNLL-2014 standard)
- GLEU score (JFLEG standard)
- Per-error-type analysis
- Correction rate and exact match statistics

Compares results against the FLAN-T5-Large baseline.

Usage:
    python evaluate_llama32.py

    # Use a custom checkpoint path:
    LLAMA32_MODEL_PATH=./checkpoints/llama32_bea2019/llama_gec_lora python evaluate_llama32.py
"""

import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from src.models.llama_gec import LlamaGEC
from src.training.evaluate import compute_f05, compute_gleu

# T5 baseline for comparison
T5_BASELINE = {
    "f05": 0.3201,
    "precision": 0.2744,
    "recall": 0.9588,
    "gleu": 0.9245,
}

CHECKPOINT_PATH = os.getenv(
    "LLAMA32_MODEL_PATH", "checkpoints/llama32_bea2019/llama_gec_lora"
)
OUTPUT_DIR = os.path.dirname(CHECKPOINT_PATH)


def main():
    print("=" * 70)
    print("EVALUATION: LLAMA 3.2-3B + LoRA (BEA 2019 TEST SET)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load Model ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading model from: {CHECKPOINT_PATH}")
    start = time.time()
    model = LlamaGEC.from_pretrained(
        model_name_or_path=CHECKPOINT_PATH,
        device=device,
    )
    model.model.eval()
    print(f"  Model loaded in {time.time() - start:.1f}s on {model.device}")

    # ── 2. Load Test Data ─────────────────────────────────────────────
    print("\n[2/5] Loading test data...")
    test_df = pd.read_csv("data/bea2019_hf/test.csv")
    test_df = test_df.dropna(subset=["source", "target"])
    sources = test_df["source"].tolist()
    references = test_df["target"].tolist()
    print(f"  Test samples: {len(sources)}")

    # ── 3. Run Inference ──────────────────────────────────────────────
    print(f"\n[3/5] Running inference on {len(sources)} samples...")
    predictions = []
    total_time = 0.0

    for text in tqdm(sources, desc="Generating corrections"):
        t0 = time.time()
        result = model.correct_text(text, num_beams=1, max_new_tokens=128)
        total_time += time.time() - t0
        predictions.append(result.corrected_text)

    avg_ms = (total_time / len(sources)) * 1000
    print(f"  Total inference time: {total_time:.1f}s")
    print(f"  Average latency:      {avg_ms:.0f}ms per sentence")

    # ── 4. Basic Statistics ───────────────────────────────────────────
    print("\n[4/5] Computing basic statistics...")
    num_changed = sum(
        1 for s, p in zip(sources, predictions) if s.strip() != p.strip()
    )
    num_exact = sum(
        1 for p, r in zip(predictions, references) if p.strip() == r.strip()
    )
    num_identity = sum(
        1 for s, r in zip(sources, references) if s.strip() == r.strip()
    )
    num_identity_correct = sum(
        1
        for s, p, r in zip(sources, predictions, references)
        if s.strip() == r.strip() and p.strip() == s.strip()
    )

    print(f"  Sentences changed by model:  {num_changed}/{len(sources)} ({100*num_changed/len(sources):.1f}%)")
    print(f"  Exact match with reference:  {num_exact}/{len(sources)} ({100*num_exact/len(sources):.1f}%)")
    print(f"  Identity samples (src==ref): {num_identity}/{len(sources)} ({100*num_identity/len(sources):.1f}%)")
    if num_identity > 0:
        print(f"  Identity preserved by model: {num_identity_correct}/{num_identity} ({100*num_identity_correct/num_identity:.1f}%)")

    # ── 5. Compute F0.5 and GLEU ──────────────────────────────────────
    print("\n[5/5] Computing evaluation metrics...")
    refs_for_f05 = [[r] for r in references]

    print("  Computing F0.5 (uses ERRANT — may take a few minutes)...")
    t0 = time.time()
    f05_scores = compute_f05(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )
    print(f"  F0.5 computed in {time.time() - t0:.1f}s")

    print("  Computing GLEU...")
    t0 = time.time()
    gleu_score = compute_gleu(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )
    print(f"  GLEU computed in {time.time() - t0:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Model:         Llama 3.2-3B-Instruct + LoRA (BEA 2019)")
    print(f"  Checkpoint:    {CHECKPOINT_PATH}")
    print(f"  Test samples:  {len(sources)}")
    print(f"  Device:        {model.device}")
    print("-" * 70)
    print(f"  {'Metric':<18} {'Llama 3.2-3B':>14}  {'T5-Large (baseline)':>20}")
    print(f"  {'-'*54}")
    print(f"  {'F0.5 Score':<18} {f05_scores['f05']:>14.4f}  {T5_BASELINE['f05']:>20.4f}")
    print(f"  {'Precision':<18} {f05_scores['precision']:>14.4f}  {T5_BASELINE['precision']:>20.4f}")
    print(f"  {'Recall':<18} {f05_scores['recall']:>14.4f}  {T5_BASELINE['recall']:>20.4f}")
    print(f"  {'GLEU Score':<18} {gleu_score:>14.4f}  {T5_BASELINE['gleu']:>20.4f}")
    print("-" * 70)
    print(f"  Avg Latency:     {avg_ms:.0f}ms per sentence")
    print(f"  Correction Rate: {100*num_changed/len(sources):.1f}%")
    print(f"  Exact Match:     {100*num_exact/len(sources):.1f}%")
    print(f"  TP/FP/FN:        {f05_scores['tp']}/{f05_scores['fp']}/{f05_scores['fn']}")
    print("=" * 70)

    # ── Save predictions and results ──────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_path = os.path.join(OUTPUT_DIR, "evaluation_predictions.csv")
    pd.DataFrame({
        "source": sources,
        "prediction": predictions,
        "reference": references,
    }).to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("EVALUATION RESULTS - Llama 3.2-3B-Instruct + LoRA (BEA 2019)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint:      {CHECKPOINT_PATH}\n")
        f.write(f"Test samples:    {len(sources)}\n")
        f.write(f"F0.5 Score:      {f05_scores['f05']:.4f}\n")
        f.write(f"Precision:       {f05_scores['precision']:.4f}\n")
        f.write(f"Recall:          {f05_scores['recall']:.4f}\n")
        f.write(f"TP/FP/FN:        {f05_scores['tp']}/{f05_scores['fp']}/{f05_scores['fn']}\n")
        f.write(f"GLEU Score:      {gleu_score:.4f}\n")
        f.write(f"Avg Latency:     {avg_ms:.0f}ms\n")
        f.write(f"Correction Rate: {100*num_changed/len(sources):.1f}%\n")
        f.write(f"Exact Match:     {100*num_exact/len(sources):.1f}%\n")
    print(f"Results saved to:  {results_path}")

    return f05_scores, gleu_score


if __name__ == "__main__":
    main()
