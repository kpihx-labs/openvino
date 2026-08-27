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
    port: int = 11436
    log_level: str = "INFO"
    debug: bool = False

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
