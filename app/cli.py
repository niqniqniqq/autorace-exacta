"""CLI entrypoint — Typer commands for data collection, training, prediction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    stream=sys.stderr,
)

app = typer.Typer(help="autorace-exacta CLI")
logger = logging.getLogger(__name__)

TRACK_NAMES = {
    "kawaguchi": "川口",
    "kawaguchi2": "川口ナイト",
    "isesaki2": "伊勢崎ナイト",
}


# ---------------------------------------------------------------
# sync:race-days
# ---------------------------------------------------------------
@app.command("sync:race-days")
def sync_race_days(
    track: str = typer.Option(..., help="Track code, e.g. kawaguchi"),
    from_date: str = typer.Option(..., "--from", help="Start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="End date YYYY-MM-DD"),
) -> None:
    """Create race_day rows for the given date range."""
    from app.db.session import get_db
    from app.services.upsert import upsert_race_day, upsert_track

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        current = d_from
        count = 0
        while current <= d_to:
            upsert_race_day(db, t.track_id, current)
            count += 1
            current = date.fromordinal(current.toordinal() + 1)
        typer.echo(f"Synced {count} race days for {track} ({from_date} to {to_date})")


# ---------------------------------------------------------------
# fetch:program
# ---------------------------------------------------------------
@app.command("fetch:program")
def fetch_program(
    track: str = typer.Option(..., help="Track code"),
    dt: str = typer.Option(..., "--date", help="Date YYYY-MM-DD | auto | latest | today"),
    lookback_days: int = typer.Option(14, help="Lookback days for auto/latest"),
    skip_if_no_meet: bool = typer.Option(
        True, "--skip-if-no-meet/--no-skip-if-no-meet", help="Skip if no meet"
    ),
) -> None:
    """Fetch and store program (出走表) for all races of the day."""
    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.scraping.parsers.program_parser import parse_program
    from app.scraping.sources.autorace_program import fetch_all_programs
    from app.services.storage import save_json_snapshot
    from app.services.upsert import (
        upsert_entry,
        upsert_race,
        upsert_race_day,
        upsert_snapshot,
        upsert_track,
    )

    from app.services.date_resolver import is_date_keyword, resolve_date_with_reason

    client = AutoraceClient(init_track_code=track) if is_date_keyword(dt) else None
    race_date, reason = resolve_date_with_reason(
        track, dt, mode="fetch:program", lookback_days=lookback_days, client=client
    )
    if race_date is None:
        if skip_if_no_meet:
            reason_text = "開催なし" if reason == "no_meet" else "解決失敗"
            logger.info("Skip fetch:program (%s) track=%s date=%s", reason_text, track, dt)
            return
        raise typer.Exit(code=1)

    dt = race_date.isoformat()
    if client is None:
        client = AutoraceClient(init_track_code=track)

    from app.services.meet_resolver import resolve_active_race_nos

    cache: dict[int, dict] = {}
    active_nos = resolve_active_race_nos(client, track, race_date, program_cache=cache)
    if not active_nos:
        logger.info("No active races for %s %s — nothing to fetch.", track, dt)
        return

    programs = fetch_all_programs(
        client, track, race_date, race_nos=active_nos, preloaded=cache
    )

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        total_entries = 0
        for race_no, prog_data in programs:
            storage_uri, chash = save_json_snapshot("program", track, dt, prog_data)
            snap = upsert_snapshot(
                db,
                source="program",
                url=f"autorace.jp/program/{track}/{dt}/R{race_no}",
                fetched_at=datetime.now(timezone.utc),
                http_status=200,
                content_hash=chash,
                content_type="application/json",
                storage_uri=storage_uri,
            )
            race = upsert_race(db, rd.race_day_id, race_no)
            entries = parse_program(prog_data, race_no)
            for entry in entries:
                upsert_entry(db, race.race_id, entry, snap.snapshot_id)
                total_entries += 1

        typer.echo(
            f"Fetched {len(programs)} races, {total_entries} entries for {track} {dt}"
        )


# ---------------------------------------------------------------
# fetch:odds
# ---------------------------------------------------------------
@app.command("fetch:odds")
def fetch_odds_cmd(
    track: str = typer.Option(..., help="Track code"),
    dt: str = typer.Option(..., "--date", help="Date YYYY-MM-DD | auto | latest | today"),
    race_no: int | None = typer.Option(None, "--race-no", help="Single race number"),
    lookback_days: int = typer.Option(14, help="Lookback days for auto/latest"),
    skip_if_no_meet: bool = typer.Option(
        True, "--skip-if-no-meet/--no-skip-if-no-meet", help="Skip if no meet"
    ),
) -> None:
    """Fetch and store exacta odds."""
    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.scraping.parsers.odds_parser import parse_exacta_odds
    from app.scraping.sources.autorace_odds import fetch_all_odds
    from app.services.guards import guard_odds_race
    from app.services.storage import save_json_snapshot
    from app.services.upsert import (
        upsert_odds_exacta,
        upsert_race_day,
        upsert_snapshot,
        upsert_track,
    )

    from app.services.date_resolver import is_date_keyword, resolve_date_with_reason
    from app.services.odds_freshness import has_fresh_odds

    client = AutoraceClient(init_track_code=track) if is_date_keyword(dt) else None
    race_date, reason = resolve_date_with_reason(
        track, dt, mode="fetch:odds", lookback_days=lookback_days, client=client
    )
    if race_date is None:
        if skip_if_no_meet:
            reason_text = "開催なし" if reason == "no_meet" else "解決失敗"
            logger.info("Skip fetch:odds (%s) track=%s date=%s", reason_text, track, dt)
            return
        raise typer.Exit(code=1)

    dt = race_date.isoformat()
    with get_db() as db:
        if has_fresh_odds(db, track, race_date):
            logger.info(
                "Skip fetch:odds (already fresh within 3 minutes) track=%s date=%s",
                track,
                dt,
            )
            return

    if client is None:
        client = AutoraceClient(init_track_code=track)

    if race_no:
        race_nos: list[int] | None = [race_no]
    else:
        from app.services.meet_resolver import resolve_active_race_nos

        race_nos = resolve_active_race_nos(client, track, race_date) or None
        if race_nos is None:
            logger.info("No active races for %s %s — nothing to fetch.", track, dt)
            return

    odds_data = fetch_all_odds(client, track, race_date, race_nos=race_nos)

    captured_at = datetime.now(timezone.utc)

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        total_odds = 0
        for rno, data in odds_data:
            guard = guard_odds_race(db, race_day_id=rd.race_day_id, race_no=rno)
            if guard.race is None:
                logger.warning(
                    "Skip odds save (guard:%s entries=%d) track=%s date=%s R%d",
                    guard.reason,
                    guard.entries_count,
                    track,
                    dt,
                    rno,
                )
                continue
            storage_uri, chash = save_json_snapshot("odds", track, dt, data)
            upsert_snapshot(
                db,
                source="odds",
                url=f"autorace.jp/odds/{track}/{dt}/R{rno}",
                fetched_at=captured_at,
                http_status=200,
                content_hash=chash,
                content_type="application/json",
                storage_uri=storage_uri,
            )
            odds_list = parse_exacta_odds(data)
            cnt = upsert_odds_exacta(
                db, guard.race.race_id, odds_list, captured_at
            )
            total_odds += cnt

        typer.echo(f"Fetched odds for {len(odds_data)} races, {total_odds} new entries")


# ---------------------------------------------------------------
# fetch:results
# ---------------------------------------------------------------
@app.command("fetch:results")
def fetch_results_cmd(
    track: str = typer.Option(..., help="Track code"),
    dt: str = typer.Option(..., "--date", help="Date YYYY-MM-DD | auto | latest | today"),
    race_no: int | None = typer.Option(None, "--race-no", help="Single race number"),
    lookback_days: int = typer.Option(14, help="Lookback days for auto/latest"),
    skip_if_no_meet: bool = typer.Option(
        True, "--skip-if-no-meet/--no-skip-if-no-meet", help="Skip if no meet"
    ),
) -> None:
    """Fetch and store race results and exacta payouts."""
    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.scraping.parsers.results_parser import parse_exacta_payout, parse_result
    from app.scraping.sources.autorace_results import fetch_all_results
    from app.services.storage import save_json_snapshot
    from app.services.upsert import (
        upsert_payout_exacta,
        upsert_race,
        upsert_race_day,
        upsert_result,
        upsert_snapshot,
        upsert_track,
    )

    from app.services.date_resolver import is_date_keyword, resolve_date_with_reason

    client = AutoraceClient(init_track_code=track) if is_date_keyword(dt) else None
    race_date, reason = resolve_date_with_reason(
        track, dt, mode="fetch:results", lookback_days=lookback_days, client=client
    )
    if race_date is None:
        if skip_if_no_meet:
            reason_text = "開催なし" if reason == "no_meet" else "解決失敗"
            logger.info("Skip fetch:results (%s) track=%s date=%s", reason_text, track, dt)
            return
        raise typer.Exit(code=1)

    dt = race_date.isoformat()
    if client is None:
        client = AutoraceClient(init_track_code=track)

    if race_no:
        race_nos: list[int] | None = [race_no]
    else:
        from app.services.meet_resolver import resolve_active_race_nos

        race_nos = resolve_active_race_nos(client, track, race_date) or None
        if race_nos is None:
            logger.info("No active races for %s %s — nothing to fetch.", track, dt)
            return

    results_data = fetch_all_results(client, track, race_date, race_nos=race_nos)

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        for rno, data in results_data:
            storage_uri, chash = save_json_snapshot("results", track, dt, data)
            upsert_snapshot(
                db,
                source="results",
                url=f"autorace.jp/results/{track}/{dt}/R{rno}",
                fetched_at=datetime.now(timezone.utc),
                http_status=200,
                content_hash=chash,
                content_type="application/json",
                storage_uri=storage_uri,
            )
            race = upsert_race(db, rd.race_day_id, rno)

            result = parse_result(data)
            if result:
                upsert_result(db, race.race_id, result)
                typer.echo(f"  R{rno}: {result.order[:3]}")

            payout = parse_exacta_payout(data)
            if payout:
                upsert_payout_exacta(db, race.race_id, payout)
                typer.echo(
                    f"  R{rno} payout: {payout.first_car_no}-{payout.second_car_no} "
                    f"¥{payout.payout_yen}"
                )

        typer.echo(f"Fetched results for {len(results_data)} races")


# ---------------------------------------------------------------
# train:model
# ---------------------------------------------------------------
@app.command("train:model")
def train_model(
    from_date: str = typer.Option(..., "--from", help="Train start date"),
    to_date: str = typer.Option(..., "--to", help="Train end date"),
    out: str = typer.Option("models/model.pkl", help="Output model path"),
) -> None:
    """Train an exacta prediction model from historical data."""
    from app.db.session import get_db
    from app.services.evaluation import compute_brier, compute_logloss
    from app.services.features import build_training_data, get_race_features
    from app.services.modeling import ExactaModel
    from app.services.upsert import upsert_model_run

    with get_db() as db:
        X, y, meta = build_training_data(db, from_date, to_date)

        if len(X) == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model = ExactaModel()
            model.save(out)
            return

        model = ExactaModel()
        model.fit(X, y, meta)

        predictions = []
        actuals = []
        for m in meta:
            car_nos, feats = get_race_features(db, m["race_id"])
            preds = model.predict_exacta(feats, car_nos)
            predictions.append(preds)
            actuals.append((m["winner"], m["second"]))

        logloss = compute_logloss(predictions, actuals)
        brier = compute_brier(predictions, actuals)

        version = Path(out).stem
        upsert_model_run(
            db,
            model_version=version,
            train_from=date.fromisoformat(from_date),
            train_to=date.fromisoformat(to_date),
            created_at=datetime.now(timezone.utc),
            logloss=logloss,
            brier=brier,
            n_races=len(meta),
            n_samples=len(y),
        )

        model.save(out)
        typer.echo(
            f"Trained on {len(meta)} races. LogLoss={logloss:.4f} Brier={brier:.6f}"
        )


# ---------------------------------------------------------------
# predict:exacta
# ---------------------------------------------------------------
@app.command("predict:exacta")
def predict_exacta(
    track: str = typer.Option(..., help="Track code"),
    dt: str = typer.Option(..., "--date", help="Date YYYY-MM-DD | auto | latest | today"),
    model_path: str = typer.Option(..., "--model", help="Path to model .pkl"),
    model_version: str = typer.Option("v0", "--model-version"),
    top: int = typer.Option(10, help="Top N predictions to show"),
    lookback_days: int = typer.Option(14, help="Lookback days for auto/latest"),
    skip_if_no_meet: bool = typer.Option(
        True, "--skip-if-no-meet/--no-skip-if-no-meet", help="Skip if no meet"
    ),
) -> None:
    """Generate and store exacta predictions for all races of a day."""
    from sqlalchemy import func, select

    from app.db.models import OddsExacta, Race, RaceDay, RaceEntry
    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.services.guards import guard_predict_race
    from app.services.features import get_race_features
    from app.services.modeling import ExactaModel
    from app.services.modeling_v11 import ExactaModelV11, extract_v11_features
    from app.services.upsert import upsert_prediction_exacta, upsert_race_day, upsert_track

    from app.services.date_resolver import is_date_keyword, resolve_date_with_reason

    client = AutoraceClient(init_track_code=track) if is_date_keyword(dt) else None
    race_date, reason = resolve_date_with_reason(
        track, dt, mode="predict:exacta", lookback_days=lookback_days, client=client
    )
    if race_date is None:
        if skip_if_no_meet:
            reason_text = "開催なし" if reason == "no_meet" else "解決失敗"
            logger.info("Skip predict:exacta (%s) track=%s date=%s", reason_text, track, dt)
            return
        raise typer.Exit(code=1)

    dt = race_date.isoformat()
    if client is None:
        client = AutoraceClient(init_track_code=track)

    from app.services.meet_resolver import resolve_active_race_nos

    active_nos = resolve_active_race_nos(client, track, race_date)
    if not active_nos:
        logger.info("No active races for %s %s — nothing to predict.", track, dt)
        return

    # Detect model type and load accordingly
    is_v11 = ExactaModelV11.is_v11_model(model_path)
    if is_v11:
        model = ExactaModelV11.load(model_path)
        logger.info("Using v11 LightGBM model with 8 raw features")
    else:
        model = ExactaModel.load(model_path)

    predicted_at = datetime.now(timezone.utc)

    with get_db() as db:
        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = upsert_race_day(db, t.track_id, race_date)

        races = (
            db.execute(
                select(Race)
                .where(Race.race_day_id == rd.race_day_id)
                .where(Race.race_no.in_(active_nos))
                .order_by(Race.race_no)
            )
            .scalars()
            .all()
        )

        for race in races:
            guard = guard_predict_race(db, race_id=race.race_id)
            if not guard.ok:
                logger.warning(
                    "Skip predict (guard:%s entries=%d odds=%d/%d) track=%s date=%s R%d",
                    guard.reason,
                    guard.entries_count,
                    guard.odds_count,
                    guard.required_odds,
                    track,
                    dt,
                    race.race_no,
                )
                continue

            # Get active car_nos from odds (handles absent players)
            latest_captured = db.scalar(
                select(func.max(OddsExacta.captured_at)).where(OddsExacta.race_id == race.race_id)
            )
            active_car_nos = list(
                db.scalars(
                    select(OddsExacta.first_car_no.distinct()).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
            )

            if is_v11:
                # v11 uses raw features extracted directly from entries
                entries = sorted(
                    [e for e in race.entries if e.car_no in active_car_nos],
                    key=lambda e: e.car_no
                )
                car_nos = [e.car_no for e in entries]
                if len(car_nos) < 2:
                    continue
                import numpy as np
                feats = np.array([extract_v11_features(e) for e in entries])
            else:
                car_nos, feats = get_race_features(db, race.race_id, active_car_nos=active_car_nos)
                if len(car_nos) < 2:
                    continue

            preds = model.predict_exacta(feats, car_nos)

            latest_odds: dict[tuple[int, int], float] = {}
            odds_rows = (
                db.execute(
                    select(OddsExacta)
                    .where(OddsExacta.race_id == race.race_id)
                    .order_by(OddsExacta.captured_at.desc())
                )
                .scalars()
                .all()
            )
            for o in odds_rows:
                key = (o.first_car_no, o.second_car_no)
                if key not in latest_odds:
                    latest_odds[key] = o.odds

            typer.echo(f"\n=== R{race.race_no} ===")
            for first, second, prob in preds[:top]:
                fair = 1.0 / prob if prob > 0 else float("inf")
                mkt = latest_odds.get((first, second))
                ev = (prob * mkt - 1) if mkt else None

                upsert_prediction_exacta(
                    db,
                    race_id=race.race_id,
                    first_car_no=first,
                    second_car_no=second,
                    prob=prob,
                    fair_odds=fair,
                    model_version=model_version,
                    predicted_at=predicted_at,
                    market_odds=mkt,
                    ev=ev,
                )

                ev_str = f"  EV={ev:+.3f}" if ev is not None else ""
                mkt_str = f"  mkt={mkt:.1f}" if mkt is not None else ""
                typer.echo(
                    f"  {first}-{second}  prob={prob:.4f}  fair={fair:.1f}{mkt_str}{ev_str}"
                )

        typer.echo(f"\nPredictions saved ({model_version})")


# ---------------------------------------------------------------
# backtest:exacta
# ---------------------------------------------------------------
@app.command("backtest:exacta")
def backtest_exacta(
    model_path: str = typer.Option(..., "--model", help="Path to model .pkl"),
    bet_amount: float = typer.Option(100.0, "--bet", help="Bet amount per combination"),
    min_odds: float = typer.Option(3.0, "--min-odds", help="Minimum odds for top prediction"),
) -> None:
    """Run backtest on races with both odds and results."""
    import numpy as np

    from app.db.session import get_db
    from app.services.backtest import get_backtest_races, run_backtest
    from app.services.features import extract_features
    from app.services.modeling import ExactaModel
    from app.services.modeling_v11 import ExactaModelV11, extract_v11_features

    # Detect model type and load accordingly
    is_v11 = ExactaModelV11.is_v11_model(model_path)
    if is_v11:
        model = ExactaModelV11.load(model_path)
        feature_extractor = extract_v11_features
        logger.info("Using v11 LightGBM model with 8 raw features")
    else:
        model = ExactaModel.load(model_path)
        feature_extractor = extract_features
        logger.info("Using v2 LogisticRegression model")

    with get_db() as db:
        races = get_backtest_races(db)
        if not races:
            typer.echo("No races found with both odds and results.")
            return

        typer.echo(f"Found {len(races)} races for backtest")

        summary = run_backtest(
            db, model, feature_extractor, bet_amount=bet_amount, min_odds=min_odds
        )

        # Display summary
        typer.echo("\n=== Backtest Summary ===")
        typer.echo(f"Total races: {summary.total_races}")
        typer.echo(f"Top-1 hits: {summary.top1_hits}/{summary.total_races} ({summary.top1_accuracy:.1%})")
        typer.echo(f"EV+ races: {summary.ev_plus_races}")
        typer.echo(f"EV+ hits: {summary.ev_plus_hits}/{summary.ev_plus_races} ({summary.ev_plus_accuracy:.1%})" if summary.ev_plus_races > 0 else "EV+ hits: 0/0")
        typer.echo(f"Total bets: {summary.total_bets}")
        typer.echo(f"Total invested: ¥{summary.total_invested:,.0f}")
        typer.echo(f"Total return: ¥{summary.total_return:,.0f}")
        typer.echo(f"Profit: ¥{summary.profit:+,.0f}")
        typer.echo(f"ROI: {summary.roi:.1%}")

        # Show individual race results
        typer.echo("\n=== Race Details ===")
        for r in summary.results:
            hit_mark = "✓" if r.hit else " "
            ev_mark = "EV✓" if r.ev_plus_hit else "EV " if r.ev_plus_bets else "   "
            typer.echo(
                f"{r.race_date} {r.track_code} R{r.race_no:2d}: "
                f"pred={r.pred_1st}-{r.pred_2nd} actual={r.actual_1st}-{r.actual_2nd} "
                f"[{hit_mark}] [{ev_mark}] profit=¥{r.profit:+,.0f}"
            )


# ---------------------------------------------------------------
# recommend:purchase
# ---------------------------------------------------------------
@app.command("recommend:purchase")
def recommend_purchase(
    track: str = typer.Option(..., help="Track code"),
    dt: str = typer.Option(..., "--date", help="Date YYYY-MM-DD | auto | latest | today"),
    bankroll: int = typer.Option(10000, help="Available bankroll in Yen"),
    kelly: float = typer.Option(0.25, help="Kelly fraction (0.25 = quarter Kelly)"),
    min_ev: float = typer.Option(0.0, "--min-ev", help="Minimum EV threshold"),
    model_version: str = typer.Option("v0", "--model-version", help="Filter by model version"),
    lookback_days: int = typer.Option(14, help="Lookback days for auto/latest"),
    skip_if_no_meet: bool = typer.Option(
        True, "--skip-if-no-meet/--no-skip-if-no-meet", help="Skip if no meet"
    ),
) -> None:
    """Generate purchase recommendations using Kelly Criterion."""
    from sqlalchemy import select

    from app.db.models import PredictionExacta, Race, RaceDay
    from app.db.session import get_db
    from app.scraping.http import AutoraceClient
    from app.services.betting import format_purchase_plan, generate_purchase_plan
    from app.services.date_resolver import is_date_keyword, resolve_date_with_reason

    client = AutoraceClient(init_track_code=track) if is_date_keyword(dt) else None
    race_date, reason = resolve_date_with_reason(
        track, dt, mode="recommend:purchase", lookback_days=lookback_days, client=client
    )
    if race_date is None:
        if skip_if_no_meet:
            reason_text = "開催なし" if reason == "no_meet" else "解決失敗"
            logger.info("Skip recommend:purchase (%s) track=%s date=%s", reason_text, track, dt)
            return
        raise typer.Exit(code=1)

    with get_db() as db:
        # Get RaceDay
        from app.services.upsert import upsert_track

        t = upsert_track(db, track, TRACK_NAMES.get(track, track))
        rd = db.execute(
            select(RaceDay).where(
                RaceDay.track_id == t.track_id, RaceDay.race_date == race_date
            )
        ).scalar_one_or_none()

        if rd is None:
            typer.echo(f"No race day found for {track} {race_date}")
            raise typer.Exit(code=1)

        # Get latest predictions for each race
        races = (
            db.execute(
                select(Race)
                .where(Race.race_day_id == rd.race_day_id)
                .order_by(Race.race_no)
            )
            .scalars()
            .all()
        )

        predictions: list[dict] = []
        for race in races:
            # Get latest prediction timestamp for this race and model version
            latest_pred = db.execute(
                select(PredictionExacta)
                .where(
                    PredictionExacta.race_id == race.race_id,
                    PredictionExacta.model_version == model_version,
                )
                .order_by(PredictionExacta.predicted_at.desc())
                .limit(1)
            ).scalar_one_or_none()

            if latest_pred is None:
                continue

            # Get all predictions at that timestamp
            preds = (
                db.execute(
                    select(PredictionExacta).where(
                        PredictionExacta.race_id == race.race_id,
                        PredictionExacta.model_version == model_version,
                        PredictionExacta.predicted_at == latest_pred.predicted_at,
                    )
                )
                .scalars()
                .all()
            )

            for p in preds:
                if p.market_odds is not None and p.ev is not None:
                    predictions.append(
                        {
                            "race_no": race.race_no,
                            "first_car_no": p.first_car_no,
                            "second_car_no": p.second_car_no,
                            "prob": p.prob,
                            "market_odds": p.market_odds,
                            "ev": p.ev,
                        }
                    )

        # Generate purchase plan
        plan = generate_purchase_plan(
            predictions=predictions,
            bankroll=bankroll,
            kelly_multiplier=kelly,
            min_ev=min_ev,
        )

        typer.echo(f"\n{track.upper()} {race_date} (model: {model_version})")
        typer.echo(format_purchase_plan(plan))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
