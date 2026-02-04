from __future__ import annotations

from datetime import date, timedelta

from app.services import date_resolver


def test_resolve_date_today_no_meet(monkeypatch):
    monkeypatch.setattr(date_resolver, "_today", lambda: date(2026, 2, 1))

    def fake_probe(client, track, race_date):
        return False

    monkeypatch.setattr(date_resolver, "_probe_meet_day", fake_probe)

    resolved = date_resolver.resolve_date(
        "kawaguchi", "today", mode="fetch:odds", lookback_days=14
    )
    assert resolved is None


def test_resolve_date_auto_lookback(monkeypatch):
    today = date(2026, 2, 1)
    monkeypatch.setattr(date_resolver, "_today", lambda: today)

    def fake_probe(client, track, race_date):
        return race_date == today - timedelta(days=2)

    monkeypatch.setattr(date_resolver, "_probe_meet_day", fake_probe)

    resolved = date_resolver.resolve_date(
        "kawaguchi", "auto", mode="fetch:odds", lookback_days=5
    )
    assert resolved == today - timedelta(days=2)
