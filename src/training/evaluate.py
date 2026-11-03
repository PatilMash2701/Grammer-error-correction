"""
Evaluation functions for grammar correction models.

Supports multiple evaluation metrics:
- F0.5 score via ERRANT (CoNLL-2014 standard)
- GLEU score (JFLEG standard)
- Precision, Recall, F1
- Per-error-type analysis using ERRANT
"""

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from loguru import logger
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute loss.

    Args:
        model: GEC model to evaluate.
        dataloader: Evaluation DataLoader.
        device: Device to evaluate on.

    Returns:
        Dictionary with:
            - 'loss': Average evaluation loss.
            - 'perplexity': Perplexity (exp of loss).
    """
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {"loss": avg_loss, "perplexity": perplexity}


def _edit_to_key(edit) -> Tuple[int, int, str]:
    """Convert an ERRANT edit to a hashable key (start, end, correction)."""
    return (edit.o_start, edit.o_end, edit.c_str)


def _compute_f_beta(tp: int, fp: int, fn: int, beta: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, and F-beta score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f_score = 0.0
    else:
        beta_sq = beta ** 2
        f_score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return {"precision": precision, "recall": recall, "f05": f_score}


def compute_f05(
    predictions: List[str],
    sources: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute F0.5 score using the ERRANT framework.

    F0.5 weights precision twice as much as recall, reflecting the
    preference in GEC for high-precision corrections.

    Args:
        predictions: List of model-corrected sentences.
        sources: List of original (erroneous) sentences.
        references: List of reference corrections (can have multiple refs per source).

    Returns:
        Dictionary with 'precision', 'recall', 'f05', 'tp', 'fp', 'fn'.
    """
    import errant

    annotator = errant.load("en")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for src_text, pred_text, ref_texts in zip(sources, predictions, references):
        src_parsed = annotator.parse(src_text)
        pred_parsed = annotator.parse(pred_text)

        # Get predicted edits
        pred_edits = annotator.annotate(src_parsed, pred_parsed)
        pred_edit_keys: Set[Tuple[int, int, str]] = set()
        for e in pred_edits:
            if e.type != "noop":
                pred_edit_keys.add(_edit_to_key(e))

        # Get best-matching reference edits (take the ref that gives highest F0.5)
        best_tp, best_fp, best_fn = 0, len(pred_edit_keys), 0

        if isinstance(ref_texts, str):
            ref_texts = [ref_texts]

        for ref_text in ref_texts:
            ref_parsed = annotator.parse(ref_text)
            ref_edits = annotator.annotate(src_parsed, ref_parsed)
            ref_edit_keys: Set[Tuple[int, int, str]] = set()
            for e in ref_edits:
                if e.type != "noop":
                    ref_edit_keys.add(_edit_to_key(e))

            tp = len(pred_edit_keys & ref_edit_keys)
            fp = len(pred_edit_keys - ref_edit_keys)
            fn = len(ref_edit_keys - pred_edit_keys)

            # Pick the reference that maximizes F0.5
            current_scores = _compute_f_beta(tp, fp, fn, beta=0.5)
            best_scores = _compute_f_beta(best_tp, best_fp, best_fn, beta=0.5)

            if current_scores["f05"] > best_scores["f05"]:
                best_tp, best_fp, best_fn = tp, fp, fn

        total_tp += best_tp
        total_fp += best_fp
        total_fn += best_fn

    result = _compute_f_beta(total_tp, total_fp, total_fn, beta=0.5)
    result["tp"] = total_tp
    result["fp"] = total_fp
    result["fn"] = total_fn
    return result


def compute_gleu(
    predictions: List[str],
    sources: List[str],
    references: List[List[str]],
    max_order: int = 4,
) -> float:
    """
    Compute GLEU score for fluency evaluation.

    GLEU (Generalized Language Evaluation Understanding) is adapted
    from BLEU for GEC. It rewards corrections that match references
    and penalizes changes that don't.

    For each sentence:
    1. Collect n-grams from source, prediction, and reference.
    2. Count n-grams in prediction that are in reference but not in source (good changes).
    3. Count n-grams in prediction that are not in reference (bad changes).
    4. GLEU = max(0, correct_ngrams) / total_prediction_ngrams

    Args:
        predictions: List of model-corrected sentences.
        sources: List of original sentences.
        references: List of reference corrections.
        max_order: Maximum n-gram order (default 4).

    Returns:
        GLEU score (0-1, higher is better).
    """
    from nltk.util import ngrams as nltk_ngrams

    sentence_gleu_scores = []

    for src_text, pred_text, ref_texts in zip(sources, predictions, references):
        if isinstance(ref_texts, str):
            ref_texts = [ref_texts]

        src_tokens = src_text.lower().split()
        pred_tokens = pred_text.lower().split()

        best_gleu = 0.0

        for ref_text in ref_texts:
            ref_tokens = ref_text.lower().split()

            # Collect n-gram counts
            all_src_ngrams: Counter = Counter()
            all_pred_ngrams: Counter = Counter()
            all_ref_ngrams: Counter = Counter()

            for n in range(1, max_order + 1):
                all_src_ngrams.update(nltk_ngrams(src_tokens, n))
                all_pred_ngrams.update(nltk_ngrams(pred_tokens, n))
                all_ref_ngrams.update(nltk_ngrams(ref_tokens, n))

            # GLEU counts:
            # - n-grams in prediction that match reference
            pred_ref_overlap = Counter()
            for ng, count in all_pred_ngrams.items():
                pred_ref_overlap[ng] = min(count, all_ref_ngrams.get(ng, 0))

            # - n-grams in source that are NOT in reference (errors in source)
            src_not_ref = Counter()
            for ng, count in all_src_ngrams.items():
                diff = count - all_ref_ngrams.get(ng, 0)
                if diff > 0:
                    src_not_ref[ng] = diff

            # - n-grams in prediction that are NOT in reference (errors in prediction)
            pred_not_ref = Counter()
            for ng, count in all_pred_ngrams.items():
                diff = count - all_ref_ngrams.get(ng, 0)
                if diff > 0:
                    pred_not_ref[ng] = diff

            # GLEU = (matched n-grams - penalty) / total prediction n-grams
            matched = sum(pred_ref_overlap.values())
            total_pred = sum(all_pred_ngrams.values())

            if total_pred == 0:
                gleu = 0.0
            else:
                gleu = matched / total_pred

            best_gleu = max(best_gleu, gleu)

        sentence_gleu_scores.append(best_gleu)

    return sum(sentence_gleu_scores) / len(sentence_gleu_scores) if sentence_gleu_scores else 0.0


def per_error_analysis(
    predictions: List[str],
    sources: List[str],
    references: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Analyze model performance per error type using ERRANT.

    Categorizes errors (e.g., R:VERB:SVA, M:DET, R:PREP) and
    computes precision/recall/F0.5 for each category.

    Args:
        predictions: Model-corrected sentences.
        sources: Original erroneous sentences.
        references: Gold-standard corrections.

    Returns:
        Dictionary mapping error types to their metrics.
    """
    import errant

    annotator = errant.load("en")

    # Track TP/FP/FN per error type
    type_tp: Dict[str, int] = defaultdict(int)
    type_fp: Dict[str, int] = defaultdict(int)
    type_fn: Dict[str, int] = defaultdict(int)

    for src_text, pred_text, ref_text in zip(sources, predictions, references):
        src_parsed = annotator.parse(src_text)
        pred_parsed = annotator.parse(pred_text)
        ref_parsed = annotator.parse(ref_text)

        pred_edits = annotator.annotate(src_parsed, pred_parsed)
        ref_edits = annotator.annotate(src_parsed, ref_parsed)

        # Build edit key -> type mappings
        pred_edit_map: Dict[Tuple[int, int, str], str] = {}
        for e in pred_edits:
            if e.type != "noop":
                pred_edit_map[_edit_to_key(e)] = e.type

        ref_edit_map: Dict[Tuple[int, int, str], str] = {}
        for e in ref_edits:
            if e.type != "noop":
                ref_edit_map[_edit_to_key(e)] = e.type

        pred_keys = set(pred_edit_map.keys())
        ref_keys = set(ref_edit_map.keys())

        # True positives: edits in both pred and ref
        for key in pred_keys & ref_keys:
            error_type = ref_edit_map[key]
            type_tp[error_type] += 1

        # False positives: edits in pred but not ref
        for key in pred_keys - ref_keys:
            error_type = pred_edit_map[key]
            type_fp[error_type] += 1

        # False negatives: edits in ref but not pred
        for key in ref_keys - pred_keys:
            error_type = ref_edit_map[key]
            type_fn[error_type] += 1

    # Compute metrics per error type
    all_types = set(type_tp.keys()) | set(type_fp.keys()) | set(type_fn.keys())
    results: Dict[str, Dict[str, float]] = {}

    for error_type in sorted(all_types):
        tp = type_tp[error_type]
        fp = type_fp[error_type]
        fn = type_fn[error_type]
        metrics = _compute_f_beta(tp, fp, fn, beta=0.5)
        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["total_ref"] = tp + fn
        results[error_type] = metrics

    return results
