"""Full evaluation of trained FLAN-T5-Large on BEA 2019 test set.

Runs inference on 4,206 test samples and computes:
- F0.5 score (ERRANT-based, CoNLL-2014 standard)
- GLEU score (JFLEG standard)
- Per-error-type analysis
- Correction rate and exact match statistics
"""

import time
import pandas as pd
import torch
from tqdm import tqdm

from src.models.t5_gec import T5GEC
from src.training.evaluate import compute_f05, compute_gleu, per_error_analysis


def main():
    print("=" * 70)
    print("PHASE 4: FULL EVALUATION - FLAN-T5-LARGE (BEA 2019 TEST SET)")
    print("=" * 70)

    # ── 1. Load Model ─────────────────────────────────────────────────
    print("\n[1/5] Loading trained model...")
    start = time.time()
    model = T5GEC.from_pretrained(
        model_name_or_path="checkpoints/flan_t5_large_bea2019/llama_gec_lora",
        use_lora=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
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
    total_time = 0

    for i, text in enumerate(tqdm(sources, desc="Generating corrections")):
        start = time.time()
        result = model.correct_text(text, max_new_tokens=128, num_beams=4)
        elapsed = time.time() - start
        total_time += elapsed
        predictions.append(result.corrected_text)

    avg_ms = (total_time / len(sources)) * 1000
    print(f"  Total inference time: {total_time:.1f}s")
    print(f"  Average latency: {avg_ms:.0f}ms per sentence")

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

    print(f"  Sentences changed by model: {num_changed}/{len(sources)} ({100*num_changed/len(sources):.1f}%)")
    print(f"  Exact match with reference: {num_exact}/{len(sources)} ({100*num_exact/len(sources):.1f}%)")
    print(f"  Identity samples (src==ref): {num_identity}/{len(sources)} ({100*num_identity/len(sources):.1f}%)")
    if num_identity > 0:
        print(f"  Identity preserved by model: {num_identity_correct}/{num_identity} ({100*num_identity_correct/num_identity:.1f}%)")

    # ── 5. Compute F0.5 and GLEU ──────────────────────────────────────
    print("\n[5/5] Computing evaluation metrics...")
    print("  Computing F0.5 (this uses ERRANT and may take a while)...")

    # F0.5 expects references as list-of-lists (multiple refs per source)
    refs_for_f05 = [[r] for r in references]

    start = time.time()
    f05_scores = compute_f05(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )
    f05_time = time.time() - start
    print(f"  F0.5 computed in {f05_time:.1f}s")

    print("  Computing GLEU...")
    start = time.time()
    gleu_score = compute_gleu(
        predictions=predictions,
        sources=sources,
        references=refs_for_f05,
    )
    gleu_time = time.time() - start
    print(f"  GLEU computed in {gleu_time:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Model: FLAN-T5-Large + LoRA (BEA 2019)")
    print(f"  Test samples: {len(sources)}")
    print(f"  Device: {model.device}")
    print("-" * 70)
    print(f"  F0.5 Score:    {f05_scores['f05']:.4f}")
    print(f"  Precision:     {f05_scores['precision']:.4f}")
    print(f"  Recall:        {f05_scores['recall']:.4f}")
    print(f"  TP/FP/FN:      {f05_scores['tp']}/{f05_scores['fp']}/{f05_scores['fn']}")
    print(f"  GLEU Score:    {gleu_score:.4f}")
    print("-" * 70)
    print(f"  Avg Latency:   {avg_ms:.0f}ms per sentence")
    print(f"  Correction Rate: {100*num_changed/len(sources):.1f}%")
    print(f"  Exact Match:   {100*num_exact/len(sources):.1f}%")
    print("=" * 70)

    # ── Save predictions for analysis ─────────────────────────────────
    output_df = pd.DataFrame({
        "source": sources,
        "prediction": predictions,
        "reference": references,
    })
    output_path = "checkpoints/flan_t5_large_bea2019/evaluation_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Save results summary
    results_path = "checkpoints/flan_t5_large_bea2019/evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write("EVALUATION RESULTS - FLAN-T5-Large + LoRA (BEA 2019)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test samples: {len(sources)}\n")
        f.write(f"F0.5 Score:    {f05_scores['f05']:.4f}\n")
        f.write(f"Precision:     {f05_scores['precision']:.4f}\n")
        f.write(f"Recall:        {f05_scores['recall']:.4f}\n")
        f.write(f"TP/FP/FN:      {f05_scores['tp']}/{f05_scores['fp']}/{f05_scores['fn']}\n")
        f.write(f"GLEU Score:    {gleu_score:.4f}\n")
        f.write(f"Avg Latency:   {avg_ms:.0f}ms\n")
        f.write(f"Correction Rate: {100*num_changed/len(sources):.1f}%\n")
        f.write(f"Exact Match:   {100*num_exact/len(sources):.1f}%\n")
    print(f"Results saved to: {results_path}")

    return f05_scores, gleu_score


if __name__ == "__main__":
    main()
