"""
voxne-local — TTS engine
========================
Wraps Chatterbox TTS (github.com/resemble-ai/chatterbox) behind a clean,
thread-safe interface for use by the FastAPI server.

Responsibilities:
  - Load the Chatterbox model once at startup (in a background thread)
  - Generate WAV takes for a given text + voice reference
  - Report model state and GPU availability
  - Write output WAVs to the system temp directory

API assumptions (chatterbox-tts, verified against v0.1.x):
  from chatterbox.tts import ChatterboxTTS
  model = ChatterboxTTS.from_pretrained(device="cuda" | "cpu")
  wav   = model.generate(text, audio_prompt_path=..., exaggeration=...,
                         cfg_weight=..., temperature=...)
  # wav  : torch.Tensor, shape [1, num_samples], dtype float32
  # model.sr : int — sample rate in Hz (typically 22050 or 24000)

If the chatterbox API changes in a future release, this is the only file
that needs updating.

Design note — lazy imports
--------------------------
torch and torchaudio are imported inside functions rather than at module
level. This lets the FastAPI server start and serve /health immediately
even before (or without) torch being installed, so the Voxne app can
begin polling for readiness as soon as the process starts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

class _State(Enum):
    NOT_LOADED = "not_loaded"
    LOADING    = "loading"
    READY      = "ready"
    ERROR      = "error"


# Module-level state — protected by _lock for safe concurrent reads.
_lock:  threading.Lock = threading.Lock()
_state: _State         = _State.NOT_LOADED
_model: Any            = None   # ChatterboxTTS instance once loaded
_error: str | None     = None   # human-readable error if state == ERROR


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUInfo:
    detected: bool
    vram_gb: float


def _detect_gpu() -> GPUInfo:
    """
    Return GPU availability and VRAM.

    Safe to call before torch is installed — returns no-GPU if torch
    cannot be imported.
    """
    try:
        import torch
    except ImportError:
        return GPUInfo(detected=False, vram_gb=0.0)

    if not torch.cuda.is_available():
        return GPUInfo(detected=False, vram_gb=0.0)

    try:
        props   = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1_073_741_824, 1)  # bytes → GB
        logger.info("GPU detected: %s (%.1f GB VRAM)", props.name, vram_gb)
        return GPUInfo(detected=True, vram_gb=vram_gb)
    except Exception:
        logger.warning("CUDA available but could not query device properties.")
        return GPUInfo(detected=True, vram_gb=0.0)


# Cache GPU info at import time — it doesn't change at runtime.
_gpu: GPUInfo = _detect_gpu()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the Chatterbox model.

    Intended to be called once from a daemon thread at server startup.
    Blocks until the model is fully loaded (may take 15–60 seconds on first
    run while model weights are downloaded from HuggingFace).

    Sets module-level state to READY on success or ERROR on failure.
    """
    global _state, _model, _error

    with _lock:
        if _state in (_State.LOADING, _State.READY):
            return  # already loading or loaded
        _state = _State.LOADING

    device = "cuda" if _gpu.detected else "cpu"
    logger.info("Loading Chatterbox model on %s …", device)

    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore[import]

        model = ChatterboxTTS.from_pretrained(device=device)

        with _lock:
            _model = model
            _state = _State.READY
            _error = None

        logger.info("Chatterbox model loaded. Sample rate: %d Hz.", model.sr)

    except Exception as exc:
        msg = f"Failed to load Chatterbox model: {exc}"
        logger.exception(msg)
        with _lock:
            _state = _State.ERROR
            _error = msg


# ---------------------------------------------------------------------------
# State queries (safe to call from any thread, before torch is available)
# ---------------------------------------------------------------------------

def is_ready() -> bool:
    """Return True if the model is loaded and ready to generate."""
    with _lock:
        return _state == _State.READY


def get_status() -> str:
    """Return a status string suitable for the /health response."""
    with _lock:
        match _state:
            case _State.NOT_LOADED | _State.LOADING:
                return "loading"
            case _State.READY:
                return "ok"
            case _State.ERROR:
                return "error"
    return "error"


def get_error() -> str | None:
    """Return the error message if state is ERROR, else None."""
    with _lock:
        return _error


def gpu_info() -> GPUInfo:
    """Return cached GPU information."""
    return _gpu


# ---------------------------------------------------------------------------
# Variant parameter adjustments
# ---------------------------------------------------------------------------

# Multipliers applied to base parameters for each named variant.
# "Neutral" uses base parameters unchanged.
_VARIANT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Tense":   {"exaggeration": 1.3, "temperature": 0.9},
    "Subdued": {"exaggeration": 0.7, "temperature": 1.1},
}

_EXAGGERATION_RANGE = (0.0, 2.0)
_TEMPERATURE_RANGE  = (0.1, 1.0)


def _adjust_params(
    exaggeration: float,
    cfg_weight:   float,
    temperature:  float,
    label:        str,
) -> tuple[float, float, float]:
    """
    Apply per-variant multipliers and clamp to valid ranges.

    Returns (exaggeration, cfg_weight, temperature).
    ``cfg_weight`` is not adjusted per-variant — only the sampling
    parameters are modified.
    """
    adj = _VARIANT_ADJUSTMENTS.get(label, {})

    e = exaggeration * adj.get("exaggeration", 1.0)
    t = temperature  * adj.get("temperature",  1.0)

    e = max(_EXAGGERATION_RANGE[0], min(_EXAGGERATION_RANGE[1], e))
    t = max(_TEMPERATURE_RANGE[0],  min(_TEMPERATURE_RANGE[1],  t))

    return round(e, 4), cfg_weight, round(t, 4)


def _seed_for_variant(text: str, label: str) -> int:
    """
    Derive a deterministic, unique seed from the text and variant label.

    Using a hash ensures that the same line always gets the same seed for
    a given variant, making re-generation reproducible. Different labels
    produce different seeds, ensuring varied outputs.
    """
    payload = f"{text}\x00{label}".encode("utf-8")
    digest  = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)  # 32-bit seed from first 8 hex chars


# ---------------------------------------------------------------------------
# WAV I/O helpers
# ---------------------------------------------------------------------------

def _save_wav(tensor: Any, sample_rate: int) -> str:
    """
    Save a float32 audio tensor to a uniquely-named WAV file in the system
    temp directory. Returns the absolute path to the written file.

    Accepts any tensor-like object with a ``.dim()``, ``.unsqueeze()``,
    ``.float()``, and ``.cpu()`` interface (i.e. a torch.Tensor).
    """
    import torchaudio  # lazy — only needed at generation time

    # Ensure [C, T] shape as expected by torchaudio.save.
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        # Some model versions return [batch, channels, samples] — drop batch.
        tensor = tensor.squeeze(0)

    tensor = tensor.float().cpu()

    take_id  = str(uuid.uuid4())
    wav_path = os.path.join(tempfile.gettempdir(), f"voxne_{take_id}.wav")

    torchaudio.save(wav_path, tensor, sample_rate, format="wav")
    return wav_path


def _duration_seconds(tensor: Any, sample_rate: int) -> float:
    """Return the duration of an audio tensor in seconds, rounded to 3dp."""
    num_samples = tensor.shape[-1]
    return round(num_samples / sample_rate, 3)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class Take:
    """A single generated audio take."""
    wav_path:         str
    duration_seconds: float


def generate_take(
    text:                 str,
    reference_audio_path: str,
    exaggeration:         float,
    cfg_scale:            float,
    temperature:          float,
    variant_label:        str,
) -> Take:
    """
    Generate a single audio take for ``text`` using the loaded model.

    Parameters
    ----------
    text:
        The dialogue line to synthesise.
    reference_audio_path:
        Absolute path to a reference audio file (WAV or MP3).
    exaggeration:
        Base exaggeration value (0.0–2.0).
    cfg_scale:
        Classifier-free guidance weight (0.1–1.0), passed as ``cfg_weight``
        to Chatterbox.
    temperature:
        Sampling temperature (0.1–1.0).
    variant_label:
        Named variant (e.g. 'Neutral', 'Tense', 'Subdued'). Used to adjust
        parameters and derive a deterministic seed for this variant.

    Returns
    -------
    Take
        Contains the absolute WAV path and duration in seconds.

    Raises
    ------
    RuntimeError
        If the model is not yet loaded.
    FileNotFoundError
        If ``reference_audio_path`` does not exist on disk.
    Exception
        CUDA OutOfMemoryError is re-raised as-is; the caller (app.py)
        detects it by class name and returns a user-friendly HTTP 500.
    """
    import torch  # lazy — only needed at generation time

    with _lock:
        if _state != _State.READY:
            raise RuntimeError(
                "Model is not ready. "
                f"Current state: {_state.value}. "
                f"Error: {_error or 'none'}"
            )
        model = _model

    if not os.path.isfile(reference_audio_path):
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio_path!r}"
        )

    adj_exaggeration, adj_cfg, adj_temperature = _adjust_params(
        exaggeration, cfg_scale, temperature, variant_label
    )

    seed = _seed_for_variant(text, variant_label)
    torch.manual_seed(seed)
    if _gpu.detected:
        torch.cuda.manual_seed(seed)

    logger.debug(
        "Generating '%s' variant=%s seed=%d exaggeration=%.2f cfg=%.2f temp=%.2f",
        text[:40], variant_label, seed, adj_exaggeration, adj_cfg, adj_temperature,
    )

    # cfg_scale is passed as cfg_weight to match Chatterbox's parameter name.
    wav = model.generate(
        text,
        audio_prompt_path=reference_audio_path,
        exaggeration=adj_exaggeration,
        cfg_weight=adj_cfg,
        temperature=adj_temperature,
    )

    sample_rate = model.sr
    duration    = _duration_seconds(wav, sample_rate)
    wav_path    = _save_wav(wav, sample_rate)

    logger.debug("Take written: %s (%.2fs)", wav_path, duration)
    return Take(wav_path=wav_path, duration_seconds=duration)


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------

def cleanup_stale_temp_files(max_age_hours: float = 24.0) -> int:
    """
    Delete ``voxne_*.wav`` files from the system temp directory that are
    older than ``max_age_hours``.

    Called at server startup to prevent indefinite temp-dir growth across
    multiple sessions. Returns the number of files removed.
    """
    import time

    tmp_dir = tempfile.gettempdir()
    cutoff  = time.time() - (max_age_hours * 3600)
    removed = 0

    try:
        for name in os.listdir(tmp_dir):
            if not (name.startswith("voxne_") and name.endswith(".wav")):
                continue
            path = os.path.join(tmp_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass  # File may have been removed by another process — ignore.
    except OSError as exc:
        logger.warning("Temp file cleanup failed: %s", exc)

    if removed:
        logger.info("Cleaned up %d stale temp WAV file(s).", removed)

    return removed
