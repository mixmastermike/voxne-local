"""
voxne-local — FastAPI application
==================================
Defines all HTTP routes and application lifecycle.

Endpoints
---------
GET  /health    — readiness check; poll until model_loaded is True
GET  /voices    — list voice presets (always empty — Chatterbox is clone-only)
POST /generate  — synthesise one or more audio takes

All endpoints return JSON. Errors use standard HTTP status codes with a
``{"detail": "human-readable message"}`` body so the Voxne app can show
them directly to the user without parsing.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import tts
from models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    TakeResponse,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

# Injected at startup by main.py so routes can report the server version.
_version: str = "0.0.0"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup and shutdown.

    Model loading is performed in a daemon thread so the health endpoint
    becomes available immediately. The Voxne app is expected to poll
    /health and wait for model_loaded to be True before sending requests.
    """
    logger.info("voxne-local starting up.")

    # Clean up WAV files left over from previous sessions.
    tts.cleanup_stale_temp_files(max_age_hours=24)

    # Start model loading in background — do not block startup.
    loader = threading.Thread(target=tts.load_model, name="model-loader", daemon=True)
    loader.start()

    yield  # Server is running.

    logger.info("voxne-local shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(version: str) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separated from module-level instantiation so the app can be created
    with a known version string at startup.
    """
    global _version
    _version = version

    application = FastAPI(
        title="voxne-local",
        description=(
            "Local Chatterbox TTS server for the Voxne desktop app. "
            "Listens on localhost only. Managed entirely by the Voxne app — "
            "users do not interact with this server directly."
        ),
        version=version,
        lifespan=lifespan,
        # Disable the default /docs and /redoc UIs in production builds.
        # Contributors running from source can re-enable these locally.
        docs_url="/docs",
        redoc_url=None,
    )

    # ----- Error handlers ---------------------------------------------------

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all handler — ensure no raw Python tracebacks reach the client."""
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Check server logs."},
        )

    # ----- Routes -----------------------------------------------------------

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Health and readiness check",
        tags=["meta"],
    )
    def health() -> HealthResponse:
        """
        Return the server's current readiness state.

        Poll this endpoint after starting the server and wait until
        ``model_loaded`` is ``True`` before sending generation requests.
        During model warmup (which may include a HuggingFace model download
        on first run), ``status`` will be ``"loading"``.
        """
        gpu = tts.gpu_info()
        return HealthResponse(
            status=tts.get_status(),
            version=_version,
            model_loaded=tts.is_ready(),
            gpu_detected=gpu.detected,
            vram_gb=gpu.vram_gb,
        )

    @application.get(
        "/voices",
        response_model=VoicesResponse,
        summary="List available voice presets",
        tags=["voices"],
    )
    def voices() -> VoicesResponse:
        """
        List built-in voice presets.

        Chatterbox is a voice-cloning model — there are no built-in presets.
        Voice identity is supplied per-request via ``reference_audio_path``.
        This endpoint always returns an empty list and exists to satisfy the
        shared Voxne API contract.
        """
        return VoicesResponse(voices=[])

    @application.post(
        "/generate",
        response_model=GenerateResponse,
        summary="Generate voice takes for a dialogue line",
        tags=["generation"],
    )
    def generate(req: GenerateRequest) -> GenerateResponse:
        """
        Synthesise one or more audio takes for the given text.

        Each ``variant_label`` produces a take with subtly different
        performance parameters (exaggeration and temperature are adjusted
        per-variant). Seeds are derived deterministically from the text and
        label, so the same request always produces the same result.

        Returns a list of takes, each with an absolute path to a WAV file
        written to the system temp directory. The Voxne app reads these
        files directly from disk.

        **Errors:**
        - ``400`` — empty text, reference audio file not found
        - ``503`` — model still loading; retry after polling ``/health``
        - ``500`` — generation failed (GPU OOM, model error, etc.)
        """
        if not tts.is_ready():
            raise HTTPException(
                status_code=503,
                detail=(
                    "The voice model is still loading. "
                    "Check /health and retry once model_loaded is true."
                ),
            )

        labels = req.labels_for_variants()
        takes: list[TakeResponse] = []

        for label in labels:
            try:
                take = tts.generate_take(
                    text=req.text,
                    reference_audio_path=req.reference_audio_path,
                    exaggeration=req.exaggeration,
                    cfg_scale=req.cfg_scale,
                    temperature=req.temperature,
                    variant_label=label,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except RuntimeError as exc:
                if "not ready" in str(exc).lower():
                    raise HTTPException(status_code=503, detail=str(exc))
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
            except Exception as exc:
                # Catch CUDA OOM by class name — avoids a hard torch import here.
                if "OutOfMemoryError" in type(exc).__name__:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "The GPU ran out of memory. "
                            "Try closing other GPU-intensive applications and retrying."
                        ),
                    )
                logger.exception("Unexpected error during generation for variant '%s'", label)
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

            takes.append(
                TakeResponse(
                    id=str(uuid.uuid4()),
                    wav_path=take.wav_path,
                    variant_label=label,
                    duration_seconds=take.duration_seconds,
                )
            )

        return GenerateResponse(takes=takes)

    return application
