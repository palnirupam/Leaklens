"""Application configuration loaded from backend/.env (or defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory = config.py -> core -> app -> backend (3 parents)
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings.

    Values are read from environment variables or from ``backend/.env``
    (see ``SettingsConfigDict`` below). No secrets are hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    allow_data_to_gemini: bool = False

    # --- API security ---
    # Shared token required on all /api routes (except /api/health). If empty,
    # auth is disabled — but then the server refuses to bind to a non-loopback
    # interface (see app.core.security.assert_safe_binding).
    api_key: str = ""

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Storage ---
    storage_dir: str = str(PROJECT_ROOT / "data" / "leaklens")

    # --- Limits / safety ---
    max_upload_mb: int = 25
    max_csv_rows: int = 500_000
    experiment_timeout_sec: int = 120
    gemini_max_retries: int = 3
    gemini_cache: bool = True
    random_seed: int = 42

    # --- Derived paths ---
    @property
    def data_dir(self) -> Path:
        return Path(self.storage_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "leaklens.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.reports_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class RuntimeContext:
    """Bundle of shared services passed through the app."""

    settings: Settings = field(default_factory=Settings)
    # pluggable store populated by the app factory
    store: object = None  # type: ignore[assignment]


# Module-level singleton settings (standard FastAPI pattern).
settings = Settings()
settings.ensure_dirs()
