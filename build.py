"""
voxne-local — binary build script
===================================
Produces a self-contained platform executable using PyInstaller.
Python, all dependencies, and the server code are bundled into a single
file — the target machine does not need Python installed.

Usage:
    pip install -r requirements.txt -r requirements-build.txt
    python build.py

Output: dist/voxne-local  (macOS/Linux)
        dist/voxne-local.exe  (Windows)

The script also writes dist/checksums.txt with the SHA-256 of the binary.
This file is published alongside the binary in GitHub Releases so the Voxne
app can verify the download.

Build notes
-----------
PyTorch is large (~1.5 GB with CUDA support). The resulting binary will be
approximately 1.5–2 GB. This is expected and acceptable — the binary is
downloaded once and cached by the Voxne app.

If the build fails with a missing module error, add the module name to the
HIDDEN_IMPORTS list below and re-run. PyInstaller cannot always detect
dynamic imports used by PyTorch and Transformers automatically.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Binary name (without extension — .exe is appended automatically on Windows).
BINARY_NAME = "voxne-local"

# PyInstaller can miss dynamic imports inside PyTorch / Transformers / uvicorn.
# Add any that cause ImportError at runtime here.
HIDDEN_IMPORTS: list[str] = [
    # uvicorn internals
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # soundfile backend
    "soundfile",
    "_soundfile_data",
]

# Packages whose data files (configs, tokenizer JSON, etc.) must be
# collected in full, not just their importable modules.
COLLECT_ALL: list[str] = [
    "chatterbox",
    "torchaudio",
    # perth (Resemble AI's audio watermarker, a chatterbox-tts dependency) ships
    # its pretrained checkpoint as package data at perth/perth_net/pretrained/
    # implicit/ (hparams.yaml, id.txt, perth_net_250000.pth.tar). PyInstaller's
    # default import-scanning only picks up Python code, not that data — without
    # --collect-all, hparams.yaml is missing at runtime and CheckpointManager.
    # __init__ hits `assert dataset_hp is not None` in its "no existing checkpoint
    # found" fallback path, since the normal inference load path never supplies
    # dataset_hp.
    "perth",
]

# Packages whose .dist-info metadata must be bundled.
# diffusers (and transformers) call importlib.metadata.version() at import
# time to version-check their dependencies. PyInstaller strips .dist-info
# directories by default, so those checks blow up with PackageNotFoundError
# before the model can load. --copy-metadata includes the metadata directory
# for each package listed here.
COPY_METADATA: list[str] = [
    "diffusers",
    "transformers",
    "requests",
    "huggingface-hub",
    "tokenizers",
    "numpy",
    "tqdm",
    "filelock",
    "packaging",
    "pyyaml",
    "regex",
    "safetensors",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binary_path() -> Path:
    """Return the expected output path of the built binary."""
    name = f"{BINARY_NAME}.exe" if platform.system() == "Windows" else BINARY_NAME
    return Path("dist") / name


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(binary: Path) -> Path:
    """Write dist/checksums.txt with the SHA-256 of the binary."""
    digest = sha256_of(binary)
    checksums_path = binary.parent / "checksums.txt"

    # Append so that if this script is run for multiple platforms in the same
    # dist/ directory (e.g. on a CI matrix that shares an artifact), all
    # checksums are collected in one file.
    with open(checksums_path, "a") as f:
        f.write(f"{digest}  {binary.name}\n")

    return checksums_path


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> None:
    print(f"Building {BINARY_NAME} for {platform.system()} / {platform.machine()} …")

    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", BINARY_NAME,
        "--noconfirm",
        "--clean",
    ]

    for module in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", module]

    for package in COLLECT_ALL:
        cmd += ["--collect-all", package]

    for package in COPY_METADATA:
        cmd += ["--copy-metadata", package]

    cmd.append("main.py")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("\nBuild failed. Check the output above for missing module errors.")
        print("If you see an ImportError for a specific module, add it to")
        print("HIDDEN_IMPORTS in build.py and re-run.")
        sys.exit(result.returncode)

    output = binary_path()
    if not output.exists():
        print(f"\nError: expected binary not found at {output}")
        sys.exit(1)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nBuild succeeded.")
    print(f"  Binary : {output.resolve()}")
    print(f"  Size   : {size_mb:.0f} MB")

    checksums_path = write_checksums(output)
    digest = sha256_of(output)
    print(f"  SHA-256: {digest}")
    print(f"  Written: {checksums_path.resolve()}")


if __name__ == "__main__":
    # Ensure we're running from the repo root.
    if not Path("main.py").exists():
        print("Error: run build.py from the repo root directory.")
        sys.exit(1)

    build()
