"""
Pydantic models for API request/response validation.

Defines the data schemas for the grammar correction API endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """Request body for single text correction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text with potential grammatical errors",
        examples=["She go to school yesterday."],
    )
    model: str = Field(
        default="coedit",
        pattern="^(t5|coedit|llama)$",
        description="Model to use: 't5' (fine-tuned FLAN-T5), 'coedit' (Grammarly CoEdIT), or 'llama' (Llama 3.2-3B)",
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of beams for beam search (higher = better but slower)",
    )


class PositionSpan(BaseModel):
    """Character position span in the original text."""

    start: int
    end: int


class CorrectionDiff(BaseModel):
    """A single correction made to the text."""

    original: str = Field(..., description="Original text span")
    corrected: str = Field(..., description="Corrected text span")
    error_type: str = Field(default="grammar", description="Error category")
    position: PositionSpan


class CorrectionResponse(BaseModel):
    """Response body for text correction."""

    original_text: str
    corrected_text: str
    corrections: List[CorrectionDiff] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_used: str = Field(default="coedit", description="Which model produced this correction")
    processing_time_ms: float = Field(ge=0.0)


class BatchCorrectionRequest(BaseModel):
    """Request body for batch text correction."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of texts to correct (max 50)",
    )
    model: str = Field(
        default="coedit",
        pattern="^(t5|coedit|llama)$",
        description="Model to use: 't5', 'coedit', or 'llama'",
    )
    num_beams: int = Field(default=4, ge=1, le=10)


class BatchCorrectionResponse(BaseModel):
    """Response body for batch correction."""

    results: List[CorrectionResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    models_loaded: Dict[str, bool] = Field(description="Which models are loaded")
    version: str = Field(default="0.1.0")
    uptime_seconds: float = Field(default=0.0)


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    parameters: Optional[int] = None
    max_length: int
    device: str
