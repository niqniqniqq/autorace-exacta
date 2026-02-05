from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services import meet_resolver


def _make_program_response(has_players: bool) -> dict[str, Any]:
    """Build a fake Program API response."""
    players = [{"playerName": "Test"}] if has_players else []
    return {"result": "Success", "body": {"playerList": players}}


def _stub_fetch(active_set: set[int]):
    """Return a fake fetch_program that responds based on *active_set*."""

    def fake_fetch(client, track_code, race_date, race_no):
        return _make_program_response(race_no in active_set)

    return fake_fetch


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_all_12_active(monkeypatch):
    """Races 1-12 have entries, 13-14 empty → returns [1..12]."""
    monkeypatch.setattr(
        meet_resolver, "fetch_program", _stub_fetch(set(range(1, 13)))
    )
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi", date(2026, 2, 4)
    )
    assert result == list(range(1, 13))


def test_8_races_night(monkeypatch):
    """Night meet with 8 races → returns [1..8]."""
    monkeypatch.setattr(
        meet_resolver, "fetch_program", _stub_fetch(set(range(1, 9)))
    )
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi2", date(2026, 2, 4)
    )
    assert result == list(range(1, 9))


def test_stops_after_3_consecutive_empty(monkeypatch):
    """Races 1-8 active, 9-11 empty → stops probing at 11, returns [1..8]."""
    monkeypatch.setattr(
        meet_resolver, "fetch_program", _stub_fetch(set(range(1, 9)))
    )
    call_count = {"n": 0}
    original = meet_resolver.fetch_program

    def counting_fetch(client, track_code, race_date, race_no):
        call_count["n"] += 1
        return _stub_fetch(set(range(1, 9)))(client, track_code, race_date, race_no)

    monkeypatch.setattr(meet_resolver, "fetch_program", counting_fetch)
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi2", date(2026, 2, 4)
    )
    assert result == list(range(1, 9))
    # Should have probed 1..11 (8 active + 3 empty) = 11 calls
    assert call_count["n"] == 11


def test_no_active_races(monkeypatch):
    """All races empty → returns []."""
    monkeypatch.setattr(meet_resolver, "fetch_program", _stub_fetch(set()))
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi", date(2026, 2, 4)
    )
    assert result == []


def test_gap_in_middle(monkeypatch):
    """Races 1-5 active, 6 empty, 7-8 active, 9-11 empty → [1..5, 7, 8]."""
    active = {1, 2, 3, 4, 5, 7, 8}
    monkeypatch.setattr(meet_resolver, "fetch_program", _stub_fetch(active))
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi", date(2026, 2, 4)
    )
    assert result == [1, 2, 3, 4, 5, 7, 8]


def test_cache_populated(monkeypatch):
    """When program_cache is passed, active entries are stored there."""
    active = {1, 2, 3}
    monkeypatch.setattr(meet_resolver, "fetch_program", _stub_fetch(active))
    cache: dict[int, dict] = {}
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi", date(2026, 2, 4), program_cache=cache
    )
    assert result == [1, 2, 3]
    assert set(cache.keys()) == {1, 2, 3}
    for v in cache.values():
        assert v["body"]["playerList"]


def test_api_error_treated_as_empty(monkeypatch):
    """If a race raises an exception, it's treated as empty and probing continues."""
    active = {1, 2, 4, 5}

    def flaky_fetch(client, track_code, race_date, race_no):
        if race_no == 3:
            raise RuntimeError("network error")
        return _make_program_response(race_no in active)

    monkeypatch.setattr(meet_resolver, "fetch_program", flaky_fetch)
    result = meet_resolver.resolve_active_race_nos(
        MagicMock(), "kawaguchi", date(2026, 2, 4)
    )
    assert 3 not in result
    assert 1 in result
    assert 2 in result
    assert 4 in result
    assert 5 in result
