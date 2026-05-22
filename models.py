"""
voxne-local — API models
========================
Pydantic request/response models for all HTTP endpoints.

These must match the Voxne app's expected contract exactly.
Refer to docs/api-spec.yaml in the voxne repository for the
authoritative OpenAPI specification.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """
    Request body for POST /generate.

    Voice identity is established entirely via ``reference_audio_path`` —
    Chatterbox is a voice-cloning model with no built-in voice library.
    The ``voice_profile_id`` field is passed through for the app's tracking
    purposes but is not used by this server.
    """

    text: Annotated[str, Field(min_length=1, max_length=2000)] = Field(
        description="Dialogue line text to synthesise."
    )
    voice_profile_id: str = Field(
        description="UUID of the voice profile in the Voxne database. "
                    "Informational only — not used for inference."
    )
    variants: Annotated[int, Field(ge=1, le=10)] = Field(
        default=3,
        description="Number of takes to generate.",
    )
    variant_labels: list[str] = Field(
        default=["Neutral", "Tense", "Subdued"],
        description="Label for each take. If more labels than variants are "
                    "provided, the excess are ignored. If fewer, extras are "
                    "auto-labelled as 'Take N'.",
    )
    reference_audio_path: str = Field(
        description="Absolute path to a reference WAV or MP3 file on this "
                    "machine. Used for voice cloning. 3–30 seconds of clean "
                    "speech from a single speaker is recommended."
    )
    exaggeration: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=0.5,
        description="How strongly to match the reference voice character. "
                    "0 = minimal influence, 2 = maximum exaggeration.",
    )
    cfg_scale: Annotated[float, Field(ge=0.1, le=1.0)] = Field(
        default=0.5,
        description="Classifier-free guidance weight. Higher values stay "
                    "closer to the reference but may sound less natural.",
    )
    temperature: Annotated[float, Field(ge=0.1, le=1.0)] = Field(
        default=0.8,
        description="Sampling temperature. Lower = more consistent and "
                    "deterministic. Higher = more expressive and varied.",
    )

    def labels_for_variants(self) -> list[str]:
        """
        Return a label list whose length exactly matches ``self.variants``.

        Extra labels are truncated; missing labels are filled with 'Take N'.
        """
        labels = list(self.variant_labels[: self.variants])
        while len(labels) < self.variants:
            labels.append(f"Take {len(labels) + 1}")
        return labels


class TakeResponse(BaseModel):
    """A single generated audio take."""

    id: str = Field(description="Unique identifier for this take (UUID v4).")
    wav_path: str = Field(
        description="Absolute path to the generated WAV file on this machine. "
                    "The Voxne app reads this file directly from disk."
    )
    variant_label: str = Field(
        description="Performance variant label, e.g. 'Neutral'."
    )
    duration_seconds: float = Field(
        description="Duration of the generated audio in seconds."
    )


class GenerateResponse(BaseModel):
    """Response from POST /generate."""

    takes: list[TakeResponse]


# ---------------------------------------------------------------------------
# /voices
# ---------------------------------------------------------------------------

class VoiceEntry(BaseModel):
    """A single voice preset entry."""

    id: str
    name: str
    backend: str
    preview_url: str | None = None


class VoicesResponse(BaseModel):
    """
    Response from GET /voices.

    Chatterbox is a voice-cloning model — there are no built-in voice
    presets. This endpoint always returns an empty list. Voice identity
    is supplied per-request via ``reference_audio_path``.
    """

    voices: list[VoiceEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """
    Response from GET /health.

    The Voxne app polls this endpoint after starting the server and waits
    until ``model_loaded`` is ``True`` before sending generation requests.
    During model warmup, ``status`` is ``"loading"``.
    """

    status: str = Field(
        description="'ok' when ready to generate, 'loading' during model "
                    "warmup, 'error' if startup failed."
    )
    backend: str = Field(default="chatterbox")
    version: str = Field(description="Server version string, e.g. '1.0.0'.")
    model_loaded: bool = Field(
        description="True only when the model is fully loaded and ready to "
                    "accept generation requests."
    )
    gpu_detected: bool = Field(
        description="True if a CUDA-capable GPU was found at startup."
    )
    vram_gb: float = Field(
        description="Total VRAM of the primary GPU in gigabytes. "
                    "0.0 if no GPU is present."
    )
