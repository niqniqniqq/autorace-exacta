from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://autorace:autorace@db:5432/autorace"

    user_agent: str = "autorace-exacta-mvp/0.1 (+local-research)"
    request_delay_min: float = 1.0
    request_delay_max: float = 3.0

    data_dir: Path = Path("/app/data")
    models_dir: Path = Path("/app/models")

    model_config = {"env_file": ".env", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
