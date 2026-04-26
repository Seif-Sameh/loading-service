"""Runtime settings loaded from environment variables (.env)."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8009

    rl_weights_path: Path = REPO_ROOT / "models" / "gopt_v1.pt"
    heightmap_resolution_mm: int = 10
    max_ems_per_step: int = 80


settings = Settings()
