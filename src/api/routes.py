"""
API route handlers for grammar correction endpoints.

Endpoints:
    POST /correct        - Correct a single text
    POST /correct/batch  - Correct multiple texts
    GET  /health         - Health check
    GET  /model/info     - Model information
"""

import difflib
import re
import time
from typing import List

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from src.api.models import (
    BatchCorrectionRequest,
    BatchCorrectionResponse,
    CorrectionDiff,
    CorrectionRequest,
    CorrectionResponse,
    HealthResponse,
    ModelInfoResponse,
    PositionSpan,
)

router = APIRouter()


def split_into_sentences(text: str) -> list[tuple[str, str]]:
    """Split text into sentences, preserving trailing whitespace.

    Returns a list of (sentence, trailing_whitespace) tuples so the
    paragraph can be reassembled exactly after per-sentence correction.
    """
    # Match sentences ending with .!? followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    results = []
    remaining = text
    for part in parts:
        idx = remaining.find(part)
        if idx > 0:
            # Capture any leading whitespace as trailing of previous sentence
            if results:
                prev_sent, _ = results[-1]
                results[-1] = (prev_sent, remaining[:idx])
        results.append((part, ""))
        remaining = remaining[idx + len(part) :]
    # Last piece gets whatever trailing whitespace remains
    if results:
        results[-1] = (results[-1][0], remaining)
    return results


def clean_t5_output(text: str) -> str:
    """Fix T5 tokenization artifacts like spaces before punctuation."""
    # Remove spaces before punctuation: "yesterday ." -> "yesterday."
    text = re.sub(r'\s+([.,!?;:\'"\)\]}])', r'\1', text)
    # Remove spaces after opening brackets/quotes
    text = re.sub(r'([\(\[{"\'])\s+', r'\1', text)
    return text.strip()


def extract_corrections(original: str, corrected: str) -> List[CorrectionDiff]:
    """Extract word-level corrections by comparing original and corrected text."""
    if original == corrected:
        return []

    orig_words = original.split()
    corr_words = corrected.split()

    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    corrections = []

    # Build character position map for original words
    char_positions = []
    pos = 0
    for word in orig_words:
        idx = original.index(word, pos)
        char_positions.append(idx)
        pos = idx + len(word)

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue

        orig_span = " ".join(orig_words[i1:i2])
        corr_span = " ".join(corr_words[j1:j2])

        if i1 < len(char_positions):
            start_pos = char_positions[i1]
        elif char_positions:
            start_pos = char_positions[-1] + len(orig_words[-1])
        else:
            start_pos = 0

        end_pos = start_pos + len(orig_span)

        corrections.append(
            CorrectionDiff(
                original=orig_span if orig_span else "(missing)",
                corrected=corr_span if corr_span else "(removed)",
                error_type="grammar",
                position=PositionSpan(start=start_pos, end=end_pos),
            )
        )

    return corrections


@router.post(
    "/correct",
    response_model=CorrectionResponse,
    summary="Correct grammar in text",
)
async def correct_text(request: CorrectionRequest) -> CorrectionResponse:
    """Correct grammatical errors in the input text."""
    from src.api.main import models

    selected = models.get(request.model)
    if selected is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model}' not loaded. Available: {list(models.keys())}",
        )

    try:
        start = time.time()

        # Split into sentences so the model gets single-sentence inputs
        sentence_pairs = split_into_sentences(request.text)
        sentences = [s for s, _ in sentence_pairs]

        if len(sentences) <= 1:
            # Single sentence – use correct_text directly
            result = selected.correct_text(
                text=request.text,
                num_beams=request.num_beams,
            )
            elapsed_ms = (time.time() - start) * 1000
            corrected_text = clean_t5_output(result.corrected_text)
            corrections = extract_corrections(request.text, corrected_text)
            return CorrectionResponse(
                original_text=request.text,
                corrected_text=corrected_text,
                corrections=corrections,
                confidence_score=result.confidence,
                model_used=request.model,
                processing_time_ms=round(elapsed_ms, 1),
            )

        # Multiple sentences – correct each individually via batch
        results = selected.correct_batch(
            texts=sentences,
            num_beams=request.num_beams,
        )
        elapsed_ms = (time.time() - start) * 1000

        # Reassemble the paragraph with original whitespace
        corrected_parts = []
        for (orig_sent, trailing_ws), result in zip(sentence_pairs, results):
            corrected_sent = clean_t5_output(result.corrected_text)
            corrected_parts.append(corrected_sent + trailing_ws)
        corrected_text = "".join(corrected_parts).strip()

        # Average confidence across sentences
        avg_confidence = round(
            sum(r.confidence for r in results) / len(results), 4
        )

        corrections = extract_corrections(request.text, corrected_text)

        return CorrectionResponse(
            original_text=request.text,
            corrected_text=corrected_text,
            corrections=corrections,
            confidence_score=avg_confidence,
            model_used=request.model,
            processing_time_ms=round(elapsed_ms, 1),
        )
    except Exception as e:
        logger.error(f"Correction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")


@router.post(
    "/correct/batch",
    response_model=BatchCorrectionResponse,
    summary="Correct grammar in multiple texts",
)
async def correct_batch(
    request: BatchCorrectionRequest,
) -> BatchCorrectionResponse:
    """Correct grammatical errors in a batch of texts."""
    from src.api.main import models

    selected = models.get(request.model)
    if selected is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model}' not loaded. Available: {list(models.keys())}",
        )

    try:
        start = time.time()
        results = selected.correct_batch(
            texts=request.texts,
            num_beams=request.num_beams,
        )
        elapsed_ms = (time.time() - start) * 1000

        responses = []
        for text, result in zip(request.texts, results):
            corrected_text = clean_t5_output(result.corrected_text)
            corrections = extract_corrections(text, corrected_text)
            resp = CorrectionResponse(
                original_text=text,
                corrected_text=corrected_text,
                corrections=corrections,
                confidence_score=result.confidence,
                model_used=request.model,
                processing_time_ms=round(elapsed_ms / len(request.texts), 1),
            )
            responses.append(resp)

        return BatchCorrectionResponse(
            results=responses,
            total_processing_time_ms=round(elapsed_ms, 1),
        )
    except Exception as e:
        logger.error(f"Batch correction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch correction failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check service health and model status."""
    from src.api.main import models, startup_time

    uptime = time.time() - startup_time if startup_time else 0.0

    models_status = {name: True for name in models}
    # Mark expected models as False if missing
    for name in ("t5", "coedit", "llama"):
        if name not in models_status:
            models_status[name] = False

    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=models_status,
        version="0.1.0",
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
)
async def model_info(
    model: str = Query(default="coedit", pattern="^(t5|coedit|llama)$"),
) -> ModelInfoResponse:
    """Get information about a loaded model."""
    from src.api.main import model_names, models

    selected = models.get(model)
    if selected is None:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded.")

    info = {
        "t5": {
            "model_name": model_names.get("t5", "FLAN-T5-Large + LoRA"),
            "model_type": "T5ForConditionalGeneration (LoRA fine-tuned)",
            "parameters": 780_000_000,
        },
        "coedit": {
            "model_name": model_names.get("coedit", "Grammarly CoEdIT-Large"),
            "model_type": "T5ForConditionalGeneration (Grammarly)",
            "parameters": 770_000_000,
        },
        "llama": {
            "model_name": model_names.get("llama", "Llama 3.2-3B-Instruct + LoRA"),
            "model_type": "LlamaForCausalLM (LoRA fine-tuned, chat format)",
            "parameters": 3_000_000_000,
        },
    }

    m = info.get(model, info["coedit"])

    return ModelInfoResponse(
        model_name=m["model_name"],
        model_type=m["model_type"],
        parameters=m["parameters"],
        max_length=selected.max_length,
        device=str(selected.device),
    )
