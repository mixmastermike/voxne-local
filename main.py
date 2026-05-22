"""
voxne-local
===========
Local Chatterbox TTS server for the Voxne desktop app.

This process is downloaded and managed automatically by the Voxne app.
Users do not run it directly. It listens on localhost only and is started
and stopped by the app as needed.

Usage (development):
    python main.py [--port PORT] [--log-level LEVEL]

The server starts immediately and begins loading the Chatterbox model in
the background. Poll GET /health until model_loaded is True before sending
generation requests.
"""

from __future__ import annotations

import argparse
import logging
import sys

# ---------------------------------------------------------------------------
# Version — keep in sync with git tags and release.yml
# ---------------------------------------------------------------------------

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    """
    Configure root logging for the process.

    In development (DEBUG / INFO), log to stderr with timestamps.
    In production (WARNING+), emit minimal output so the Voxne app's log
    capture stays readable.
    """
    level = getattr(logging, level_name.upper(), logging.WARNING)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s" if level < logging.WARNING \
          else "%(levelname)-8s  %(name)s  %(message)s"

    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,  # Override any previously set handlers.
    )

    # Quieten noisy third-party loggers unless we're in DEBUG mode.
    if level > logging.DEBUG:
        for noisy in ("uvicorn.access", "httpx", "httpcore", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voxne-local",
        description="Local Chatterbox TTS server for the Voxne desktop app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        metavar="HOST",
        help=(
            "Host address to bind to. "
            "Default is 127.0.0.1 (localhost only). "
            "Do not change this in production — the server is not "
            "designed to be exposed to a network."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["debug", "info", "warning", "error"],
        dest="log_level",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"voxne-local {__version__}",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger = logging.getLogger("voxne_local")
    logger.info("voxne-local %s", __version__)
    logger.info("Listening on %s:%d", args.host, args.port)

    # Import uvicorn and the app factory here so logging is set up first.
    import uvicorn
    from app import create_app

    application = create_app(version=__version__)

    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        # Use "warning" so uvicorn's access log doesn't spam the Voxne app's
        # log capture, but errors are still visible.
        log_level=args.log_level,
        # Single worker — the GIL means true parallelism isn't possible with
        # PyTorch anyway, and the Voxne app sends requests sequentially.
        workers=1,
    )


if __name__ == "__main__":
    main()
