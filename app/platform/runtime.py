"""
Runtime and platform helpers: Python version guidance, UTF-8 stdio, staged startup logs.
Does not raise for version warnings (only logs).
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import sys
import time
import warnings
from typing import Any, Dict, Optional

logger = logging.getLogger("mega_ai.platform")

# Official support: 3.11 and 3.12 (tested in Docker). Newer versions may work with ecosystem caveats.
MIN_PYTHON = (3, 11)
RECOMMENDED_MAX_EXCLUSIVE = (3, 13)  # warn at 3.13+


def configure_stdio_utf8() -> None:
    """Best-effort UTF-8 on stdout/stderr to avoid UnicodeEncodeError on Windows cp1252."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError, TypeError):
                pass


def configure_runtime_warnings() -> None:
    """Reduce noisy duplicate UserWarning from Groq on Python 3.13+ (Pydantic v1 compat note)."""
    if sys.version_info >= (3, 13):
        warnings.filterwarnings(
            "once",
            message=".*Pydantic V1 functionality isn't compatible with Python 3.14.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "once",
            message=".*Pydantic V1 functionality isn't compatible with Python 3.13.*",
            category=UserWarning,
        )


def warn_unsupported_python() -> None:
    """Log actionable guidance; never raises."""
    vi = sys.version_info[:2]
    if vi < MIN_PYTHON:
        logger.error(
            "python_version_unsupported",
            extra={
                "detail": f"Python {vi[0]}.{vi[1]} is below minimum {MIN_PYTHON[0]}.{MIN_PYTHON[1]}. "
                "Upgrade to Python 3.11 or 3.12.",
            },
        )
        return
    if vi >= RECOMMENDED_MAX_EXCLUSIVE:
        logger.warning(
            "python_version_not_fully_supported",
            extra={
                "detail": f"Python {vi[0]}.{vi[1]} is newer than officially tested ({MIN_PYTHON[0]}.{MIN_PYTHON[1]}–3.12). "
                "Groq and other AI packages may emit Pydantic v1 compatibility warnings or break; "
                "use Python 3.11 or 3.12 for the most stable experience.",
            },
        )


def _pkg_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def gather_runtime_diagnostics() -> Dict[str, Any]:
    """Lightweight facts for /diagnostics and startup logs (no heavy imports)."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": sys.platform,
        "groq_version": _pkg_version("groq"),
        "chromadb_version": _pkg_version("chromadb"),
        "pydantic_version": _pkg_version("pydantic"),
        "httpx_version": _pkg_version("httpx"),
        "sentence_transformers_version": _pkg_version("sentence-transformers"),
    }


def log_startup_stage(
    stage: str,
    *,
    ok: bool = True,
    latency_ms: Optional[float] = None,
    detail: Optional[str] = None,
    remediation: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"stage": stage, "ok": ok}
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    if detail:
        payload["detail"] = detail
    if remediation:
        payload["remediation"] = remediation
    if ok:
        logger.info("startup_stage", extra=payload)
    else:
        logger.error("startup_stage_failed", extra=payload)


def stage_timer() -> float:
    return time.perf_counter()
