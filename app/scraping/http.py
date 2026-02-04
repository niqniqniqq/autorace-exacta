"""HTTP client with rate limiting, CSRF handling, and retry logic for autorace.jp."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from typing import Any

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://autorace.jp"
DEFAULT_INIT_TRACK = "kawaguchi"
INIT_PATH_TEMPLATE = "/race_info/Live/{track_code}"

PLACE_CODES: dict[str, int] = {
    "kawaguchi": 2,
    "isesaki": 3,
    "hamamatsu": 4,
    "iizuka": 5,
    "sanyou": 6,
}
NIGHT_SUFFIX = "2"


def resolve_place_code(track_code: str) -> int:
    if track_code in PLACE_CODES:
        return PLACE_CODES[track_code]
    if track_code.endswith(NIGHT_SUFFIX):
        base = track_code[: -len(NIGHT_SUFFIX)]
        if base in PLACE_CODES:
            return PLACE_CODES[base]
    raise KeyError(f"Unsupported track_code: {track_code}")


class AutoraceClient:
    """Session-based client with CSRF token management and polite rate limiting."""

    def __init__(self, *, init_track_code: str | None = None) -> None:
        cfg = get_settings()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": cfg.user_agent,
                "Accept": "application/json, text/html, */*",
            }
        )
        self._init_track_code = init_track_code or DEFAULT_INIT_TRACK
        self._delay_min = cfg.request_delay_min
        self._delay_max = cfg.request_delay_max
        self._csrf_token: str | None = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # CSRF
    # ------------------------------------------------------------------
    def _ensure_csrf(self) -> str:
        if self._csrf_token:
            return self._csrf_token
        self._wait()
        init_path = INIT_PATH_TEMPLATE.format(track_code=self._init_track_code)
        resp = self._session.get(f"{BASE_URL}{init_path}", timeout=30)
        resp.raise_for_status()
        match = re.search(r'<meta name="csrf-token" content="([^"]+)"', resp.text)
        if not match:
            raise RuntimeError("Failed to extract CSRF token from autorace.jp")
        self._csrf_token = match.group(1)
        logger.info("CSRF token acquired")
        return self._csrf_token

    def _refresh_csrf(self) -> str:
        self._csrf_token = None
        return self._ensure_csrf()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        delay = random.uniform(self._delay_min, self._delay_max)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------
    def get(self, path: str, **kwargs: Any) -> requests.Response:
        self._wait()
        url = f"{BASE_URL}{path}"
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    def post_api(
        self, path: str, payload: dict[str, Any], *, max_retries: int = 3
    ) -> dict[str, Any]:
        """POST to a JSON API endpoint with CSRF and exponential backoff."""
        csrf = self._ensure_csrf()
        headers = {
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-TOKEN": csrf,
        }
        url = f"{BASE_URL}{path}"
        backoff = 1.0

        for attempt in range(max_retries):
            self._wait()
            logger.debug("POST %s (attempt %d)", url, attempt + 1)
            resp = self._session.post(url, json=payload, headers=headers, timeout=30)

            if resp.status_code == 419:
                logger.warning("CSRF mismatch (419), refreshing token")
                csrf = self._refresh_csrf()
                headers["X-CSRF-TOKEN"] = csrf
                continue

            if resp.status_code >= 500:
                logger.warning("Server error %d, backing off %.1fs", resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue

            resp.raise_for_status()
            data = resp.json()
            return data

        raise RuntimeError(f"Failed after {max_retries} attempts: POST {url}")

    @property
    def raw_session(self) -> requests.Session:
        return self._session


def content_hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()
