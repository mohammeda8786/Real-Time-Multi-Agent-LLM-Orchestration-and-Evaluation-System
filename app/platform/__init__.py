"""Platform portability: Python version checks, console encoding, startup diagnostics."""

from app.platform.runtime import (
    configure_runtime_warnings,
    configure_stdio_utf8,
    gather_runtime_diagnostics,
    log_startup_stage,
    warn_unsupported_python,
)

__all__ = [
    "configure_runtime_warnings",
    "configure_stdio_utf8",
    "gather_runtime_diagnostics",
    "log_startup_stage",
    "warn_unsupported_python",
]
