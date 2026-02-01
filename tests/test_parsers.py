"""Snapshot tests for parsers using fixture data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class TestProgramParser:
    def test_parse_program_entries(self):
        from app.scraping.parsers.program_parser import parse_program

        with open(FIXTURES / "program_sample.json") as f:
            data = json.load(f)

        entries = parse_program(data, race_no=1)

        assert len(entries) == 8

        e1 = entries[0]
        assert e1.car_no == 1
        assert e1.racer_name == "鈴木圭一郎"
        assert e1.racer_code == "3901"
        assert e1.handicap_m == 10
        assert e1.trial_time == 3.37
        assert e1.deviation == 56.2
        assert e1.quinella_rate == 45.3
        assert e1.trio_rate == 62.1
        assert e1.machine_name == "ダイナ100"

        e8 = entries[7]
        assert e8.car_no == 8
        assert e8.racer_name == "中村三郎"
        assert e8.handicap_m == 70

    def test_parse_empty_program(self):
        from app.scraping.parsers.program_parser import parse_program

        entries = parse_program({"result": "Success", "body": {}}, race_no=1)
        assert entries == []


class TestOddsParser:
    def test_parse_exacta_odds(self):
        from app.scraping.parsers.odds_parser import parse_exacta_odds

        with open(FIXTURES / "odds_sample.json") as f:
            data = json.load(f)

        odds = parse_exacta_odds(data)

        # 8 runners * 7 opponents = 56 combinations
        assert len(odds) == 56

        # Check specific odds
        o12 = next(o for o in odds if o.first_car_no == 1 and o.second_car_no == 2)
        assert o12.odds == 3.5

        o21 = next(o for o in odds if o.first_car_no == 2 and o.second_car_no == 1)
        assert o21.odds == 4.2

        # All odds positive
        for o in odds:
            assert o.odds > 0
            assert o.first_car_no != o.second_car_no

    def test_parse_empty_odds(self):
        from app.scraping.parsers.odds_parser import parse_exacta_odds

        odds = parse_exacta_odds({"result": "Success", "body": {}})
        assert odds == []


class TestResultsParser:
    def test_parse_result(self):
        from app.scraping.parsers.results_parser import parse_result

        with open(FIXTURES / "results_sample.json") as f:
            data = json.load(f)

        result = parse_result(data)
        assert result is not None
        assert result.is_valid is True
        assert result.order[0] == 1  # winner car_no
        assert result.order[1] == 3  # second
        assert result.order[2] == 2  # third
        assert len(result.order) == 8

    def test_parse_exacta_payout(self):
        from app.scraping.parsers.results_parser import parse_exacta_payout

        with open(FIXTURES / "results_sample.json") as f:
            data = json.load(f)

        payout = parse_exacta_payout(data)
        assert payout is not None
        assert payout.first_car_no == 1
        assert payout.second_car_no == 3
        assert payout.payout_yen == 820
        assert payout.popularity_rank == 2
