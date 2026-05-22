# voxne-local

Local [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) server for the [Voxne](https://github.com/mixmastermike/voxne) desktop app.

**Most users don't need this repo.** The Voxne app downloads and manages this server automatically. This repo is for contributors who want to improve the inference server, or for developers integrating with the Voxne generation API.

---

## What it does

`voxne-local` wraps Chatterbox TTS behind a small HTTP API. The Voxne app starts it as a background process on your machine, sends generation requests to it, and reads the resulting WAV files from disk. It listens on `localhost` only and is not accessible from other machines.

---

## Requirements

- **Python 3.11** (specifically — 3.12+ is not yet supported because `spacy-pkuseg`, a Chatterbox dependency, has no pre-built wheel for newer versions and requires a C++ compiler to build from source)
- An NVIDIA GPU with CUDA 12.4+ is strongly recommended — CPU inference works but is very slow (3–8 minutes per take vs 10–30 seconds on GPU)
- ~5 GB disk space (PyTorch ~2 GB + Chatterbox model weights ~2.5 GB, downloaded automatically on first run)

---

## Development setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mixmastermike/voxne-local
cd voxne-local
```

**Windows (cmd.exe — not PowerShell):**
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

> **Windows note:** Always use `cmd.exe` to activate the venv, not PowerShell.
> PowerShell blocks `.ps1` script execution by default. If you get an execution
> policy error, switch to `cmd.exe` and use `activate.bat` instead.

### 2. Install PyTorch

PyTorch must be installed **before** `requirements.txt` because `chatterbox-tts`
requires exactly `torch==2.6.0`, which is not available on the default PyPI index.

Pick the right index for your hardware:

**NVIDIA GPU — RTX 20xx / 30xx / 40xx / 50xx (CUDA 12.4):**
```bash
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**CPU only** (slow but works, good for CI or no-GPU dev machines):
```bash
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

After installing, verify CUDA is detected before continuing:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

Expected output on a CUDA machine: `True` and `2.6.0+cu124`. If you see `False`
and have an NVIDIA GPU, see [Troubleshooting](#troubleshooting) below.

### 3. Install remaining dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the server

```bash
python main.py --port 8765 --log-level info
```

The server starts immediately and loads the Chatterbox model in a background
thread. On first run, model weights (~2.5 GB) are downloaded from HuggingFace
automatically — this can take a few minutes depending on your connection.

Poll `/health` until `model_loaded` is `true`:

```bash
curl http://localhost:8765/health
```

`status` progresses from `"loading"` → `"ok"`. Once `model_loaded` is `true`,
the server is ready to generate audio.

---

## API

### `GET /health`

Returns server readiness. Poll until `model_loaded` is `true` before generating.

```json
{
  "status": "ok",
  "backend": "chatterbox",
  "version": "0.1.0",
  "model_loaded": true,
  "gpu_detected": true,
  "vram_gb": 16.0
}
```

| Field | Description |
|---|---|
| `status` | `"loading"` during warmup, `"ok"` when ready, `"error"` if load failed |
| `model_loaded` | `true` only when fully ready to accept generation requests |
| `gpu_detected` | `true` if a CUDA-capable GPU was found at startup |
| `vram_gb` | Total VRAM of the primary GPU in GB, `0.0` if no GPU |

---

### `GET /voices`

Always returns an empty list. Chatterbox is a voice-cloning model — voice
identity is provided per-request via `reference_audio_path`.

---

### `POST /generate`

Generate one or more audio takes for a line of dialogue.

**Request:**

```json
{
  "text": "You shouldn't be here.",
  "voice_profile_id": "any-string",
  "variants": 3,
  "variant_labels": ["Neutral", "Tense", "Subdued"],
  "reference_audio_path": "/absolute/path/to/reference.wav",
  "exaggeration": 0.5,
  "cfg_scale": 0.5,
  "temperature": 0.8
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Dialogue line to synthesise (max 2000 chars) |
| `voice_profile_id` | string | required | Passed through for app tracking — not used by this server |
| `variants` | int | `3` | Number of takes to generate (1–10) |
| `variant_labels` | string[] | `["Neutral","Tense","Subdued"]` | Label per take; drives subtle parameter adjustments |
| `reference_audio_path` | string | required | Absolute path to reference WAV or MP3 on this machine |
| `exaggeration` | float | `0.5` | Voice character strength (0.0–2.0) |
| `cfg_scale` | float | `0.5` | Guidance weight — higher = closer to reference (0.1–1.0) |
| `temperature` | float | `0.8` | Sampling temperature — lower = more consistent (0.1–1.0) |

**Response:**

```json
{
  "takes": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "wav_path": "C:\\Users\\you\\AppData\\Local\\Temp\\voxne_abc123.wav",
      "variant_label": "Neutral",
      "duration_seconds": 2.4
    }
  ]
}
```

`wav_path` is an absolute path on the local machine. The Voxne app reads it
directly from disk.

**Errors:**

| Status | Meaning |
|---|---|
| `400` | Empty text, or `reference_audio_path` does not exist on disk |
| `503` | Model still loading — poll `/health` and retry when `model_loaded` is `true` |
| `500` | Generation failed (GPU out of memory, model error, etc.) |

---

## Voice reference guidelines

For best cloning quality, the reference audio should be:

- **Format:** WAV or MP3
- **Length:** 10–15 seconds (3–30s acceptable)
- **Quality:** Clean speech, minimal background noise or music
- **Content:** Natural speech from a single speaker

Shorter clips or non-speech audio (music, sound effects) will produce output
that sounds like the Chatterbox default voice rather than the reference speaker.

---

## Troubleshooting

**`gpu_detected: false` despite having an NVIDIA GPU**

PyTorch was installed without CUDA support. Reinstall with the correct index:

```bash
python -m pip uninstall torch torchaudio -y
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If it still prints `False`, check that your NVIDIA drivers
are up to date (`nvidia-smi` should show CUDA Version 12.4 or higher).

---

**`error: Microsoft Visual C++ 14.0 or greater is required` during pip install**

You're using Python 3.12 or newer. `spacy-pkuseg` (a Chatterbox dependency) has
no pre-built wheel for Python 3.12+ and attempts to compile from source.

Fix: create the venv with Python 3.11 specifically:

```bat
py -3.11 -m venv .venv
```

If `py -3.11` is not found, download Python 3.11 from
[python.org](https://www.python.org/downloads/) and install it alongside your
current version.

---

**PowerShell reports "running scripts is disabled on this system"**

PowerShell blocks `.ps1` script activation by default. Use `cmd.exe` instead:

```bat
.venv\Scripts\activate.bat
```

Or enable scripts for the current user (PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

**`pip` not found, or packages install to the wrong Python**

Always use `python -m pip` rather than `pip` directly. This ensures packages
install into the active venv and not a system Python:

```bash
python -m pip install ...
```

---

**`status: "error"` in `/health` after startup**

The model failed to load. Check the server logs for the full traceback. Common
causes: missing `chatterbox-tts` package (run the install steps again), or
insufficient VRAM (Chatterbox requires ~4 GB VRAM minimum).

---

**Dependency conflict: `chatterbox-tts requires torch==2.6.0`**

This happens if torch was installed from the default PyPI index, which provides
a different version. Always install torch first using the `--index-url` flag as
shown in the setup steps above.

---

## Building a binary

To produce a self-contained executable that doesn't require Python to be
installed on the target machine:

```bash
# Install torch first (as above), then:
python -m pip install -r requirements.txt -r requirements-build.txt
python build.py
```

Output: `dist/voxne-local` (macOS/Linux) or `dist/voxne-local.exe` (Windows).

The binary bundles Python, PyTorch, FastAPI, and Chatterbox. It does **not**
include the model weights — those are downloaded on first run.

---

## Releasing

Push a version tag to trigger the GitHub Actions release workflow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds binaries for Windows (amd64), macOS Intel (amd64), and
macOS Apple Silicon (arm64), then publishes them to a GitHub Release alongside
a `checksums.txt`.

---

## Platform notes

**Windows:** On first run, Windows Defender or antivirus software may flag the
PyInstaller binary. This is a known false positive with self-built executables.
The Voxne app warns users about this. Code signing (planned) will resolve it.

**macOS:** After downloading, macOS may quarantine the binary. The Voxne app
removes the quarantine flag automatically with `xattr -dr com.apple.quarantine`.

---

## Contributing

1. Fork the repo and create a branch
2. Make your changes — keep `tts.py` focused on inference, `app.py` focused on HTTP
3. Test with `python main.py --log-level debug`
4. Open a pull request

The server is intentionally small (~400 lines across four files). Before adding
features, check whether they belong here or in the Voxne app itself.

---

## License

MIT — see [LICENSE](LICENSE).
