"""Snapshot storage — save raw API responses to gzipped files."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def _snapshot_dir(source: str, track_code: str, date_str: str) -> Path:
    cfg = get_settings()
    return cfg.data_dir / "snapshots" / source / track_code / date_str


def save_snapshot(
    source: str,
    track_code: str,
    date_str: str,
    content: bytes,
    suffix: str = ".json.gz",
) -> tuple[str, str]:
    """Save content to gzipped file. Returns (storage_uri, content_hash)."""
    sha = hashlib.sha256(content).hexdigest()
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    directory = _snapshot_dir(source, track_code, date_str)
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{ts}_{sha[:16]}{suffix}"
    filepath = directory / filename

    if filepath.exists():
        logger.debug("Snapshot already exists: %s", filepath)
        return f"file://{filepath}", sha

    with gzip.open(filepath, "wb") as f:
        f.write(content)

    logger.info("Saved snapshot: %s (%d bytes)", filepath, len(content))
    return f"file://{filepath}", sha


def save_json_snapshot(
    source: str, track_code: str, date_str: str, data: dict
) -> tuple[str, str]:
    """Serialize dict to JSON bytes and save."""
    content = json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")
    return save_snapshot(source, track_code, date_str, content)


def content_hash_exists(sha: str, source: str, track_code: str, date_str: str) -> bool:
    """Check if a snapshot with this hash already exists in the directory."""
    directory = _snapshot_dir(source, track_code, date_str)
    if not directory.exists():
        return False
    for f in directory.iterdir():
        if sha[:16] in f.name:
            return True
    return False
