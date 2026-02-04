from __future__ import annotations

import pytest

from app.scraping.http import resolve_place_code


@pytest.mark.parametrize(
    ("track_code", "expected"),
    [
        ("kawaguchi", 2),
        ("kawaguchi2", 2),
        ("isesaki", 3),
        ("isesaki2", 3),
    ],
)
def test_resolve_place_code(track_code, expected):
    assert resolve_place_code(track_code) == expected
