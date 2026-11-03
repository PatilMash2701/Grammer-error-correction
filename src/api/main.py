"""
FastAPI application for the Grammar Correction System.

Provides REST API endpoints for grammar correction with:
- Single text correction
- Batch correction
- Health check
- Model info

Loads the FLAN-T5-Large model with LoRA adapters on startup.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router

# Global model references (loaded during startup)
models: dict = {}
model_names: dict = {}
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads GEC models into memory on startup and cleans up on shutdown.
    """
    global models, model_names, startup_time
    startup_time = time.time()

    from src.models.t5_gec import T5GEC

    # Load fine-tuned FLAN-T5-Large + LoRA
    t5_path = os.getenv(
        "MODEL_PATH", "checkpoints/flan_t5_large_bea2019/llama_gec_lora"
    )
    logger.info(f"Loading T5 model from: {t5_path}")
    try:
        models["t5"] = T5GEC.from_pretrained(
            t5_path, task_prefix="grammar: "
        )
        model_names["t5"] = t5_path
        logger.info("T5 model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load T5 model: {e}")

    # Load Grammarly CoEdIT-Large
    coedit_path = os.getenv("COEDIT_PATH", "grammarly/coedit-large")
    logger.info(f"Loading CoEdIT model from: {coedit_path}")
    try:
        models["coedit"] = T5GEC.from_pretrained(
            coedit_path,
            use_lora=False,
            task_prefix="Fix grammatical errors in this sentence: ",
        )
        model_names["coedit"] = coedit_path
        logger.info("CoEdIT model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load CoEdIT model: {e}")

    logger.info(f"Models loaded: {list(models.keys())}")

    # Load Llama 3.2-3B-Instruct (fine-tuned LoRA checkpoint or base model)
    llama_path = os.getenv(
        "LLAMA32_MODEL_PATH", "checkpoints/llama32_bea2019/llama_gec_lora"
    )
    logger.info(f"Loading Llama model from: {llama_path}")
    try:
        from src.models.llama_gec import LlamaGEC
        models["llama"] = LlamaGEC.from_pretrained(llama_path)
        model_names["llama"] = llama_path
        logger.info("Llama model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Llama model: {e}")

    logger.info(f"Models loaded: {list(models.keys())}")

    yield

    # Cleanup
    logger.info("Shutting down, releasing model resources")
    models.clear()


app = FastAPI(
    title="Grammar Correction API",
    description=(
        "Grammar correction system with multiple models: "
        "FLAN-T5-Large (fine-tuned with LoRA) and Grammarly CoEdIT-Large. "
        "Detects and corrects grammar, spelling, and punctuation errors."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API info."""
    return {
        "name": "Grammar Correction API",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
