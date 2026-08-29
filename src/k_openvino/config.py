"""
config.py — Canonical KpihX configuration for openvino
=============================================================
Minimal, dependency-light config. No YAML loader, no ConfigManager
singleton — runtime values are plain constants or a frozen dataclass
declared directly here.

Secrets in .env (none required for local openvino), metadata in pyproject.toml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv  # python-dotenv

# ---------------------------------------------------------------------------
# Secrets — .env for local dev, process env for prod (none required here)
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).parent
_DOT_ENV = _PKG_DIR / ".env"
load_dotenv(_DOT_ENV, override=False)

REQUIRED_SECRETS: list[str] = []


class SecretsUnavailableError(RuntimeError):
    """Raised when a required secret cannot be resolved."""


def get_secret(name: str, *, required: bool = True) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if required:
        raise SecretsUnavailableError(
            f"Secret '{name}' not available. Set it in {_DOT_ENV}."
        )
    return ""


# ---------------------------------------------------------------------------
# Runtime config — plain frozen dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 11437
    log_level: str = "INFO"
    debug: bool = False
    max_context: int = 32768  # Central cap — IR on Intel Arc GPU unreliable above this
    # Absolute default (industry range 4K-16K) — never a ratio of max_context: real
    # providers (GPT-4o, Gemini, DeepSeek) use a flat cap independent of context size.
    default_output_tokens: int = 8192
    # Qwen3's chat template defaults to a verbose <think> chain-of-thought that
    # can burn hundreds of tokens before ever emitting a tool call — great for
    # hard reasoning, bad for a fast agent (Lite). Per-request `enable_thinking`
    # in the request body overrides this; this is only the server default.
    default_enable_thinking: bool = True
    # Streaming requests run in a dedicated subprocess so a client disconnect can
    # kill a still-prefilling generation that Python's high-level
    # LLMPipeline.generate() API cannot otherwise interrupt.
    stream_worker_spawn_method: Literal["spawn", "fork", "forkserver"] = "spawn"
    # After a client disconnect, first ask the worker to cancel gracefully via
    # StreamingStatus.CANCEL; if it still hasn't stopped after this grace period,
    # kill the worker subprocess (covers the prefill-before-first-token case).
    stream_cancel_grace_seconds: float = 0.75
    stream_worker_shutdown_timeout_seconds: float = 2.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def home(self) -> Path:
        # Respects OPENVINO_HOME like the server, fallback to default
        return Path(
            os.environ.get("OPENVINO_HOME", str(Path.home() / ".local/share/openvino"))
        )

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


CONFIG = Config()
