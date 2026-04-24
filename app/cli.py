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
    force: bool = typer.Option(False, "--force", help="Skip freshness check"),
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
        if not force and has_fresh_odds(db, track, race_date):
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
# train:model-v12
# ---------------------------------------------------------------
@app.command("train:model-v12")
def train_model_v12(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v12_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
) -> None:
    """Train v12 LightGBM model with time-series CV (14 features)."""
    from app.db.session import get_db
    from app.services.training import train_v12_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v12_model(db, d_from, d_to, n_folds=n_folds)

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v12 Training Report ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>25s}: {imp:4d} {bar}")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v12",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


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
    from app.services.modeling_v12 import ExactaModelV12, extract_v12_features
    from app.services.modeling_v13 import ExactaModelV13, extract_v13_features, compute_runner_odds_stats
    from app.services.modeling_v14 import ExactaModelV14, extract_v14_features
    from app.services.modeling_v15 import ExactaModelV15
    from app.services.modeling_v16 import ExactaModelV16, extract_v16_features
    from app.services.modeling_v17 import ExactaModelV17, extract_v17_features
    from app.services.modeling_v18 import ExactaModelV18, extract_v18_features, compute_race_context
    from app.services.modeling_v19 import ExactaModelV19
    from app.services.modeling_v20 import ExactaModelV20
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

    # Detect model type (v20 → v19 → v18 → v17 → v16 → v15 → v14 → v13 → v12 → v11 → legacy)
    is_v20 = ExactaModelV20.is_v20_model(model_path)
    is_v19 = not is_v20 and ExactaModelV19.is_v19_model(model_path)
    is_v18 = not is_v20 and not is_v19 and ExactaModelV18.is_v18_model(model_path)
    is_v17 = not is_v20 and not is_v19 and not is_v18 and ExactaModelV17.is_v17_model(model_path)
    is_v16 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and ExactaModelV16.is_v16_model(model_path)
    is_v15 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and ExactaModelV15.is_v15_model(model_path)
    is_v14 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and ExactaModelV14.is_v14_model(model_path)
    is_v13 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and ExactaModelV13.is_v13_model(model_path)
    is_v12 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and not is_v13 and ExactaModelV12.is_v12_model(model_path)
    is_v11 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and not is_v13 and not is_v12 and ExactaModelV11.is_v11_model(model_path)
    if is_v20:
        model = ExactaModelV20.load(model_path)
        logger.info("Using v20 multi-track LightGBM model (track=%s)", track)
    elif is_v19:
        model = ExactaModelV19.load(model_path)
        logger.info("Using v19 LightGBM model with 22 features (isotonic + conditional alpha)")
    elif is_v18:
        model = ExactaModelV18.load(model_path)
        logger.info("Using v18 LightGBM model with 22 features (race-relative + interactions)")
    elif is_v17:
        model = ExactaModelV17.load(model_path)
        logger.info("Using v17 LightGBM model with 16 features (odds-free)")
    elif is_v16:
        model = ExactaModelV16.load(model_path)
        logger.info("Using v16 LightGBM model with 19 features (API stats)")
    elif is_v15:
        model = ExactaModelV15.load(model_path)
        logger.info("Using v15 pairwise model with 37 pair features")
    elif is_v14:
        model = ExactaModelV14.load(model_path)
        logger.info("Using v14 LightGBM model with 15 features (racer history)")
    elif is_v13:
        model = ExactaModelV13.load(model_path)
        logger.info("Using v13 LightGBM model with 12 features (odds + calibration)")
    elif is_v12:
        model = ExactaModelV12.load(model_path)
        logger.info("Using v12 LightGBM model with 9 features")
    elif is_v11:
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

            if is_v20 or is_v19 or is_v18:
                import numpy as np
                from app.services.racer_stats import get_racer_stats
                entries = sorted(
                    [e for e in race.entries if e.car_no in active_car_nos],
                    key=lambda e: e.car_no,
                )
                car_nos = [e.car_no for e in entries]
                if len(car_nos) < 2:
                    continue
                odds_rows = db.scalars(
                    select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
                race_ctx = compute_race_context(entries)
                feats = np.array([
                    extract_v18_features(
                        e,
                        racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                        race_context=race_ctx,
                    )
                    for e in entries
                ])
            elif is_v17:
                import numpy as np
                from app.services.racer_stats import get_racer_stats
                entries = sorted(
                    [e for e in race.entries if e.car_no in active_car_nos],
                    key=lambda e: e.car_no,
                )
                car_nos = [e.car_no for e in entries]
                if len(car_nos) < 2:
                    continue
                # v17: no odds features needed, but we need odds_dict for market blend
                odds_rows = db.scalars(
                    select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
                feats = np.array([
                    extract_v17_features(
                        e,
                        racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    )
                    for e in entries
                ])
            elif is_v16 or is_v15 or is_v14 or is_v13:
                import numpy as np
                entries = sorted(
                    [e for e in race.entries if e.car_no in active_car_nos],
                    key=lambda e: e.car_no,
                )
                car_nos = [e.car_no for e in entries]
                if len(car_nos) < 2:
                    continue
                # Get odds for v13+ features
                odds_rows = db.scalars(
                    select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
                odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
                if is_v16:
                    from app.services.racer_stats import get_racer_stats
                    feats = np.array([
                        extract_v16_features(
                            e, odds_stats,
                            racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                        )
                        for e in entries
                    ])
                elif is_v15 or is_v14:
                    from app.services.racer_stats import get_racer_stats
                    feats = np.array([
                        extract_v14_features(
                            e, odds_stats,
                            racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                        )
                        for e in entries
                    ])
                else:
                    feats = np.array([extract_v13_features(e, odds_stats) for e in entries])
            elif is_v12 or is_v11:
                import numpy as np
                entries = sorted(
                    [e for e in race.entries if e.car_no in active_car_nos],
                    key=lambda e: e.car_no
                )
                car_nos = [e.car_no for e in entries]
                if len(car_nos) < 2:
                    continue
                if is_v12:
                    feats = np.array([extract_v12_features(e) for e in entries])
                else:
                    feats = np.array([extract_v11_features(e) for e in entries])
            else:
                car_nos, feats = get_race_features(db, race.race_id, active_car_nos=active_car_nos)
                if len(car_nos) < 2:
                    continue

            # Pass market pair probs for v13+ blend
            market_pair_probs = None
            if (is_v20 or is_v19 or is_v18 or is_v17 or is_v16 or is_v15 or is_v14 or is_v13) and (is_v20 or hasattr(model, "market_alpha")):
                total_inv = sum(1.0 / o for o in odds_dict.values() if o > 0)
                if total_inv > 0:
                    market_pair_probs = {
                        pair: (1.0 / o) / total_inv
                        for pair, o in odds_dict.items() if o > 0
                    }

            if market_pair_probs is not None:
                if is_v20:
                    preds = model.predict_exacta(feats, car_nos, track_code=track, market_pair_probs=market_pair_probs, odds_dict=odds_dict)
                elif is_v19:
                    preds = model.predict_exacta(feats, car_nos, market_pair_probs=market_pair_probs, odds_dict=odds_dict)
                else:
                    preds = model.predict_exacta(feats, car_nos, market_pair_probs=market_pair_probs)
            else:
                if is_v20:
                    preds = model.predict_exacta(feats, car_nos, track_code=track)
                else:
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

            # Upsert all predictions and collect EV data
            all_ev: list[tuple[int, int, float, float, float]] = []  # (1st,2nd,prob,mkt,ev)
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

                if ev is not None and mkt is not None:
                    all_ev.append((first, second, prob, mkt, ev))

            # --- EV+ display: split by odds bracket ---
            min_ev_display = 0.15
            ev_low  = [(f,s,p,m,e) for f,s,p,m,e in all_ev if m <  20 and e > min_ev_display]
            ev_high = [(f,s,p,m,e) for f,s,p,m,e in all_ev if m >= 20 and e > min_ev_display]
            ev_low.sort(key=lambda x: -x[4])   # sort by EV desc
            ev_high.sort(key=lambda x: -x[4])

            has_ev = ev_low or ev_high
            if has_ev:
                if ev_low:
                    typer.echo("  【本命帯EV+】")
                    for f,s,p,m,e in ev_low[:3]:
                        typer.echo(f"    {f}-{s}  prob={p:.3f}  mkt={m:.1f}x  EV={e:+.2f}")
                if ev_high:
                    typer.echo("  【穴帯EV+】")
                    for f,s,p,m,e in ev_high[:3]:
                        typer.echo(f"    {f}-{s}  prob={p:.3f}  mkt={m:.1f}x  EV={e:+.2f}")
            else:
                # No EV+ bets: show top 3 by probability
                typer.echo("  【EV+なし / トップ3】")
                for first, second, prob in preds[:3]:
                    mkt = latest_odds.get((first, second))
                    fair = 1.0 / prob if prob > 0 else float("inf")
                    ev = (prob * mkt - 1) if mkt else None
                    ev_str = f"  EV={ev:+.2f}" if ev is not None else ""
                    mkt_str = f"  mkt={mkt:.1f}x" if mkt is not None else ""
                    typer.echo(f"    {first}-{second}  prob={prob:.3f}  fair={fair:.1f}{mkt_str}{ev_str}")

        typer.echo(f"\nPredictions saved ({model_version})")


# ---------------------------------------------------------------
# backtest:exacta
# ---------------------------------------------------------------
@app.command("backtest:exacta")
def backtest_exacta(
    model_path: str = typer.Option(..., "--model", help="Path to model .pkl"),
    bet_amount: float = typer.Option(100.0, "--bet", help="Bet amount per combination"),
    min_odds: float = typer.Option(3.0, "--min-odds", help="Minimum odds for top prediction"),
    min_ev: float = typer.Option(0.0, "--min-ev", help="Minimum EV threshold (e.g. 0.10)"),
    kelly: float = typer.Option(0.0, "--kelly", help="Kelly fraction (0=flat, 0.25=quarter-Kelly)"),
) -> None:
    """Run backtest on races with both odds and results."""
    import numpy as np
    from sqlalchemy import func, select as sa_select

    from app.db.session import get_db
    from app.db.models import OddsExacta, Race, RaceEntry
    from app.services.backtest import get_backtest_races, run_backtest
    from app.services.features import entries_to_features
    from app.services.modeling_v2 import ExactaModelV2
    from app.services.modeling_v11 import ExactaModelV11, extract_v11_features
    from app.services.modeling_v12 import ExactaModelV12, extract_v12_features
    from app.services.modeling_v13 import ExactaModelV13, extract_v13_features, compute_runner_odds_stats
    from app.services.modeling_v14 import ExactaModelV14, extract_v14_features
    from app.services.modeling_v15 import ExactaModelV15
    from app.services.modeling_v16 import ExactaModelV16, extract_v16_features
    from app.services.modeling_v17 import ExactaModelV17, extract_v17_features
    from app.services.modeling_v18 import ExactaModelV18, extract_v18_features, compute_race_context
    from app.services.modeling_v19 import ExactaModelV19
    from app.services.modeling_v20 import ExactaModelV20

    # Detect model type (v20 → v19 → v18 → v17 → v16 → v15 → v14 → v13 → v12 → v11 → legacy)
    is_v20 = ExactaModelV20.is_v20_model(model_path)
    is_v19 = not is_v20 and ExactaModelV19.is_v19_model(model_path)
    is_v18 = not is_v20 and not is_v19 and ExactaModelV18.is_v18_model(model_path)
    is_v17 = not is_v20 and not is_v19 and not is_v18 and ExactaModelV17.is_v17_model(model_path)
    is_v16 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and ExactaModelV16.is_v16_model(model_path)
    is_v15 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and ExactaModelV15.is_v15_model(model_path)
    is_v14 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and ExactaModelV14.is_v14_model(model_path)
    is_v13 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and ExactaModelV13.is_v13_model(model_path)
    is_v12 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and not is_v13 and ExactaModelV12.is_v12_model(model_path)
    is_v11 = not is_v20 and not is_v19 and not is_v18 and not is_v17 and not is_v16 and not is_v15 and not is_v14 and not is_v13 and not is_v12 and ExactaModelV11.is_v11_model(model_path)

    if is_v20 or is_v19 or is_v18:
        if is_v20:
            model = ExactaModelV20.load(model_path)
            logger.info("Using v20 multi-track LightGBM model")
        elif is_v19:
            model = ExactaModelV19.load(model_path)
            logger.info("Using v19 LightGBM model with 22 features (isotonic + conditional alpha)")
        else:
            model = ExactaModelV18.load(model_path)
            logger.info("Using v18 LightGBM model with 22 features (race-relative + interactions)")

        def feature_extractor_v18(db, race: Race, entries: list[RaceEntry]):
            from app.services.racer_stats import get_racer_stats
            car_nos = [e.car_no for e in entries]
            race_date = race.race_day.race_date
            race_ctx = compute_race_context(entries)
            feats = np.array([
                extract_v18_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    race_context=race_ctx,
                )
                for e in entries
            ])
            return car_nos, feats

        feature_extractor = feature_extractor_v18
    elif is_v17:
        model = ExactaModelV17.load(model_path)
        logger.info("Using v17 LightGBM model with 16 features (odds-free)")

        def feature_extractor_v17(db, race: Race, entries: list[RaceEntry]):
            from app.services.racer_stats import get_racer_stats
            car_nos = [e.car_no for e in entries]
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v17_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        feature_extractor = feature_extractor_v17
    elif is_v16:
        model = ExactaModelV16.load(model_path)
        logger.info("Using v16 LightGBM model with 19 features (API stats)")

        def feature_extractor_v16(db, race: Race, entries: list[RaceEntry]):
            from app.services.racer_stats import get_racer_stats
            car_nos = [e.car_no for e in entries]
            latest_captured = db.scalar(
                sa_select(func.max(OddsExacta.captured_at)).where(
                    OddsExacta.race_id == race.race_id
                )
            )
            if latest_captured:
                odds_rows = db.scalars(
                    sa_select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
            else:
                odds_dict = {}
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v16_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        feature_extractor = feature_extractor_v16
    elif is_v15:
        model = ExactaModelV15.load(model_path)
        logger.info("Using v15 pairwise model with 37 pair features")

        def feature_extractor_v15(db, race: Race, entries: list[RaceEntry]):
            from app.services.racer_stats import get_racer_stats
            car_nos = [e.car_no for e in entries]
            latest_captured = db.scalar(
                sa_select(func.max(OddsExacta.captured_at)).where(
                    OddsExacta.race_id == race.race_id
                )
            )
            if latest_captured:
                odds_rows = db.scalars(
                    sa_select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
            else:
                odds_dict = {}
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v14_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        feature_extractor = feature_extractor_v15
    elif is_v14:
        model = ExactaModelV14.load(model_path)
        logger.info("Using v14 LightGBM model with 15 features (racer history)")

        def feature_extractor_v14(db, race: Race, entries: list[RaceEntry]):
            from app.services.racer_stats import get_racer_stats
            car_nos = [e.car_no for e in entries]
            latest_captured = db.scalar(
                sa_select(func.max(OddsExacta.captured_at)).where(
                    OddsExacta.race_id == race.race_id
                )
            )
            if latest_captured:
                odds_rows = db.scalars(
                    sa_select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
            else:
                odds_dict = {}
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v14_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        feature_extractor = feature_extractor_v14
    elif is_v13:
        model = ExactaModelV13.load(model_path)
        logger.info("Using v13 LightGBM model with 12 features (odds + calibration)")

        def feature_extractor_v13(db, race: Race, entries: list[RaceEntry]):
            car_nos = [e.car_no for e in entries]
            latest_captured = db.scalar(
                sa_select(func.max(OddsExacta.captured_at)).where(
                    OddsExacta.race_id == race.race_id
                )
            )
            if latest_captured:
                odds_rows = db.scalars(
                    sa_select(OddsExacta).where(
                        OddsExacta.race_id == race.race_id,
                        OddsExacta.captured_at == latest_captured,
                    )
                ).all()
                odds_dict = {(o.first_car_no, o.second_car_no): o.odds for o in odds_rows}
            else:
                odds_dict = {}
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            feats = np.array([extract_v13_features(e, odds_stats) for e in entries])
            return car_nos, feats

        feature_extractor = feature_extractor_v13
    elif is_v12:
        model = ExactaModelV12.load(model_path)
        logger.info("Using v12 LightGBM model with 9 features")

        def feature_extractor_v12(db, race: Race, entries: list[RaceEntry]):
            car_nos = [e.car_no for e in entries]
            feats = [extract_v12_features(entry) for entry in entries]
            return car_nos, np.array(feats)

        feature_extractor = feature_extractor_v12
    elif is_v11:
        model = ExactaModelV11.load(model_path)
        logger.info("Using v11 LightGBM model with 8 raw features")

        def feature_extractor_v11(db, race: Race, entries: list[RaceEntry]):
            car_nos = [e.car_no for e in entries]
            features = np.array([extract_v11_features(e) for e in entries])
            return car_nos, features

        feature_extractor = feature_extractor_v11
    else:
        model = ExactaModelV2.load(model_path)
        logger.info("Using v2 LogisticRegression model with 23 features")

        def feature_extractor_v2(db, race: Race, entries: list[RaceEntry]):
            race_date = race.race_day.race_date
            track_code = race.race_day.track.track_code
            car_nos = [e.car_no for e in entries]
            features = entries_to_features(entries, db=db, race_date=race_date, track_code=track_code)
            return car_nos, features

        feature_extractor = feature_extractor_v2

    with get_db() as db:
        races = get_backtest_races(db)
        if not races:
            typer.echo("No races found with both odds and results.")
            return

        typer.echo(f"Found {len(races)} races for backtest")

        summary = run_backtest(
            db, model, feature_extractor,
            bet_amount=bet_amount, min_odds=min_odds,
            min_ev=min_ev, kelly_fraction=kelly,
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
# train:model-v13
# ---------------------------------------------------------------
@app.command("train:model-v13")
def train_model_v13(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v13_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit Platt calibration"),
) -> None:
    """Train v13 LightGBM model with odds features + Platt calibration."""
    from app.db.session import get_db
    from app.services.training import train_v13_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v13_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v13 Training Report ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>25s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "calibrator") and model.calibrator:
            a, b = model.calibrator
            typer.echo(f"\n--- Platt Calibration ---")
            typer.echo(f"  a={a:.4f}, b={b:.4f}")

        # Market blend info
        if hasattr(model, "market_alpha") and model.market_alpha > 0:
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v13",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v14
# ---------------------------------------------------------------
@app.command("train:model-v14")
def train_model_v14(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v14_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit Platt calibration"),
) -> None:
    """Train v14 LightGBM model with racer history + home track + Platt calibration."""
    from app.db.session import get_db
    from app.services.training import train_v14_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v14_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v14 Training Report ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>25s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "calibrator") and model.calibrator:
            a, b = model.calibrator
            typer.echo(f"\n--- Platt Calibration ---")
            typer.echo(f"  a={a:.4f}, b={b:.4f}")

        # Market blend info
        if hasattr(model, "market_alpha") and model.market_alpha > 0:
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v14",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v15
# ---------------------------------------------------------------
@app.command("train:model-v15")
def train_model_v15(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v15_pair.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
) -> None:
    """Train v15 pairwise model (direct pair scoring, no Plackett-Luce)."""
    from app.db.session import get_db
    from app.services.training import train_v15_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v15_model(db, d_from, d_to, n_folds=n_folds)

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v15 Training Report (Pairwise) ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} pair samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances (top 20)
        typer.echo(f"\n--- Feature Importances (top 20) ---")
        for name, imp in report.feature_importances[:20]:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>30s}: {imp:4d} {bar}")

        # Market blend info
        if hasattr(model, "market_alpha") and model.market_alpha > 0:
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v15",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v16
# ---------------------------------------------------------------
@app.command("train:model-v16")
def train_model_v16(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v16_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit Platt calibration"),
) -> None:
    """Train v16 LightGBM model with API stats + race context + Platt calibration."""
    from app.db.session import get_db
    from app.services.training import train_v16_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v16_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v16 Training Report ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>25s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "calibrator") and model.calibrator:
            a, b = model.calibrator
            typer.echo(f"\n--- Platt Calibration ---")
            typer.echo(f"  a={a:.4f}, b={b:.4f}")

        # Market blend info
        if hasattr(model, "market_alpha") and model.market_alpha > 0:
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v16",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v17
# ---------------------------------------------------------------
@app.command("train:model-v17")
def train_model_v17(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v17_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit Platt calibration"),
) -> None:
    """Train v17 odds-free LightGBM model + Platt calibration + market blend."""
    from app.db.session import get_db
    from app.services.training import train_v17_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v17_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v17 Training Report (Odds-Free) ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>25s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "calibrator") and model.calibrator:
            a, b = model.calibrator
            typer.echo(f"\n--- Platt Calibration ---")
            typer.echo(f"  a={a:.4f}, b={b:.4f}")

        # Market blend info
        if hasattr(model, "market_alpha"):
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v17",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v18
# ---------------------------------------------------------------
@app.command("train:model-v18")
def train_model_v18(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v18_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit Platt calibration"),
) -> None:
    """Train v18 LightGBM model with race-relative features + interactions + Platt calibration."""
    from app.db.session import get_db
    from app.services.training import train_v18_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v18_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v18 Training Report (Race-Relative + Interactions) ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>30s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "calibrator") and model.calibrator:
            a, b = model.calibrator
            typer.echo(f"\n--- Platt Calibration ---")
            typer.echo(f"  a={a:.4f}, b={b:.4f}")

        # Market blend info
        if hasattr(model, "market_alpha"):
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  alpha={model.market_alpha:.2f} ({model.market_alpha:.0%} model + {1-model.market_alpha:.0%} market)")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v18",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v19
# ---------------------------------------------------------------
@app.command("train:model-v19")
def train_model_v19(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v19_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit isotonic calibration"),
) -> None:
    """Train v19 LightGBM model with isotonic calibration + conditional alpha."""
    from app.db.session import get_db
    from app.services.training import train_v19_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v19_model(
            db, d_from, d_to, n_folds=n_folds, calibrate=calibrate
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        # Display CV results
        typer.echo(f"\n=== v19 Training Report (Isotonic Cal + Conditional Alpha) ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"\n--- Cross-Validation ({len(report.fold_results)} folds) ---")
        for fr in report.fold_results:
            typer.echo(
                f"  Fold {fr.fold}: train={fr.train_size} val={fr.val_size}"
                f" ({fr.val_races} races) logloss={fr.logloss:.4f}"
                f" top1={fr.top1_accuracy:.1%}"
            )
        typer.echo(f"\n  CV Mean LogLoss: {report.cv_mean_logloss:.4f}")
        typer.echo(f"  CV Mean Top-1:   {report.cv_mean_top1:.1%}")

        # Feature importances
        typer.echo(f"\n--- Feature Importances ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>30s}: {imp:4d} {bar}")

        # Calibration info
        if hasattr(model, "_isotonic") and model._isotonic is not None:
            typer.echo(f"\n--- Isotonic Calibration ---")
            typer.echo(f"  Fitted (nonparametric)")

        # Market blend info
        if hasattr(model, "market_alpha"):
            typer.echo(f"\n--- Market Blend ---")
            typer.echo(f"  Global alpha={model.market_alpha:.2f}")
        if hasattr(model, "alpha_map") and model.alpha_map:
            typer.echo(f"\n--- Conditional Alpha (by odds bucket) ---")
            for bucket, alpha in model.alpha_map.items():
                typer.echo(f"  {bucket:>8s}: alpha={alpha:.2f}")

        # Save model
        model.save(out)

        # Record in model_runs
        upsert_model_run(
            db,
            model_version="v19",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# train:model-v20
# ---------------------------------------------------------------
@app.command("train:model-v20")
def train_model_v20(
    from_date: str = typer.Option(..., "--from", help="Train start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Train end date YYYY-MM-DD"),
    out: str = typer.Option("models/model_v20_lgb.pkl", "--out", help="Output model path"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
    min_track_races: int = typer.Option(100, "--min-track-races", help="Min races to train per-track model"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate", help="Fit isotonic calibration"),
) -> None:
    """Train v20 multi-track LightGBM model (one model per track_code)."""
    from app.db.session import get_db
    from app.services.training import train_v20_model
    from app.services.upsert import upsert_model_run

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    with get_db() as db:
        model, report = train_v20_model(
            db, d_from, d_to, n_folds=n_folds,
            calibrate=calibrate, min_track_races=min_track_races,
        )

        if report.total_samples == 0:
            typer.echo("No training data found. Saving unfitted model.")
            model.save(out)
            return

        typer.echo(f"\n=== v20 Training Report (Multi-Track) ===")
        typer.echo(f"Total: {report.total_races} races, {report.total_samples} samples")
        typer.echo(f"Track models trained: {model.track_codes()}")

        typer.echo(f"\n--- Feature Importances (fallback model) ---")
        for name, imp in report.feature_importances:
            bar = "#" * min(imp, 50)
            typer.echo(f"  {name:>30s}: {imp:4d} {bar}")

        from app.services.modeling_v20 import _FALLBACK_KEY
        for tc, sub in model.track_models.items():
            label = tc if tc != _FALLBACK_KEY else "fallback"
            typer.echo(f"\n  [{label}] alpha={sub.market_alpha:.2f}  alpha_map={sub.alpha_map}")

        model.save(out)

        upsert_model_run(
            db,
            model_version="v20",
            train_from=d_from,
            train_to=d_to,
            created_at=datetime.now(timezone.utc),
            logloss=report.cv_mean_logloss,
            n_races=report.total_races,
            n_samples=report.total_samples,
        )

        typer.echo(f"\nModel saved to {out}")


# ---------------------------------------------------------------
# evaluate:exacta
# ---------------------------------------------------------------
@app.command("evaluate:exacta")
def evaluate_exacta(
    from_date: str = typer.Option(..., "--from", help="Evaluation start date YYYY-MM-DD"),
    to_date: str = typer.Option(..., "--to", help="Evaluation end date YYYY-MM-DD"),
    train_days: int = typer.Option(60, "--train-days", help="Training window in days"),
    val_days: int = typer.Option(7, "--val-days", help="Validation window in days"),
    test_days: int = typer.Option(7, "--test-days", help="Test window in days"),
    step_days: int = typer.Option(7, "--step-days", help="Step size in days"),
    version: str = typer.Option("v18", "--version", help="Model version (v16, v17, or v18)"),
) -> None:
    """Walk-forward evaluation with market baseline comparison."""
    import numpy as np

    from app.db.session import get_db
    from app.services.racer_stats import get_racer_stats
    from app.services.walkforward import generate_splits, run_walkforward

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)

    splits = generate_splits(
        d_from, d_to,
        train_days=train_days, val_days=val_days,
        test_days=test_days, step_days=step_days,
    )

    if not splits:
        typer.echo("No valid splits generated. Check date range and window sizes.")
        raise typer.Exit(code=1)

    typer.echo(f"\n=== Walk-Forward Evaluation ({version}) ===")
    typer.echo(f"Period: {from_date} ~ {to_date}")
    typer.echo(
        f"Splits: {len(splits)}, Train={train_days}d, Val={val_days}d, "
        f"Test={test_days}d, Step={step_days}d"
    )

    if version == "v18":
        from app.services.modeling_v18 import ExactaModelV18, extract_v18_features, compute_race_context
        from app.services.training import build_v18_training_data

        def train_model_fn(rows):
            from lightgbm import LGBMClassifier

            X = np.array([r.features for r in rows])
            y = np.array([r.label for r in rows])
            clf = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            clf.fit(X, y)
            return ExactaModelV18(model=clf)

        def extract_features_fn(db, race, entries, odds_dict):
            car_nos = [e.car_no for e in entries]
            race_date = race.race_day.race_date
            race_ctx = compute_race_context(entries)
            feats = np.array([
                extract_v18_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                    race_context=race_ctx,
                )
                for e in entries
            ])
            return car_nos, feats

        build_training_data_fn = build_v18_training_data
    elif version == "v17":
        from app.services.modeling_v17 import ExactaModelV17, extract_v17_features
        from app.services.training import build_v17_training_data

        def train_model_fn(rows):
            from lightgbm import LGBMClassifier

            X = np.array([r.features for r in rows])
            y = np.array([r.label for r in rows])
            clf = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            clf.fit(X, y)
            return ExactaModelV17(model=clf)

        def extract_features_fn(db, race, entries, odds_dict):
            car_nos = [e.car_no for e in entries]
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v17_features(
                    e,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        build_training_data_fn = build_v17_training_data
    else:
        from app.services.modeling_v13 import compute_runner_odds_stats
        from app.services.modeling_v16 import ExactaModelV16, extract_v16_features
        from app.services.training import build_v16_training_data

        def train_model_fn(rows):
            from lightgbm import LGBMClassifier

            X = np.array([r.features for r in rows])
            y = np.array([r.label for r in rows])
            clf = LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15,
                min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1,
            )
            clf.fit(X, y)
            return ExactaModelV16(model=clf)

        def extract_features_fn(db, race, entries, odds_dict):
            car_nos = [e.car_no for e in entries]
            odds_stats = compute_runner_odds_stats(odds_dict, car_nos)
            race_date = race.race_day.race_date
            feats = np.array([
                extract_v16_features(
                    e, odds_stats,
                    racer_stats=get_racer_stats(db, e.racer_id, before_date=race_date),
                )
                for e in entries
            ])
            return car_nos, feats

        build_training_data_fn = build_v16_training_data

    with get_db() as db:
        report = run_walkforward(
            db, splits,
            build_training_data_fn=build_training_data_fn,
            train_model_fn=train_model_fn,
            extract_features_fn=extract_features_fn,
        )

    if not report.splits:
        typer.echo("\nNo evaluable splits. Insufficient data.")
        return

    # Display per-split results
    for sr in report.splits:
        s = sr.split
        typer.echo(
            f"\n--- Split {sr.split_idx}: "
            f"[train {s.train_from}~{s.train_to}] "
            f"[val {s.val_from}~{s.val_to}] "
            f"[test {s.test_from}~{s.test_to}] ---"
        )
        typer.echo(
            f"  Model:    LogLoss={sr.model_logloss:.3f}  "
            f"Brier={sr.model_brier:.4f}  Top1={sr.model_top1:.1%}  "
            f"({sr.n_races} races)"
        )
        typer.echo(
            f"  Baseline: LogLoss={sr.baseline_logloss:.3f}  "
            f"Brier={sr.baseline_brier:.4f}  Top1={sr.baseline_top1:.1%}"
        )
        delta_ll = sr.model_logloss - sr.baseline_logloss
        mark = "+" if delta_ll >= 0 else "-"
        typer.echo(f"  Delta LogLoss: {delta_ll:+.3f} {'x' if delta_ll >= 0 else 'v'}")

    # Aggregated summary
    m_ll, m_ll_std = report.model_logloss
    b_ll, b_ll_std = report.baseline_logloss
    m_br, m_br_std = report.model_brier
    b_br, b_br_std = report.baseline_brier
    m_t1, m_t1_std = report.model_top1
    b_t1, b_t1_std = report.baseline_top1

    typer.echo(f"\n--- Summary ({len(report.splits)} splits) ---")
    typer.echo(f"{'':>14s}  {'Model':>16s}  {'Baseline':>16s}  {'Delta':>10s}")
    typer.echo(
        f"{'LogLoss':>14s}  {m_ll:.3f}+/-{m_ll_std:.3f}  "
        f"{b_ll:.3f}+/-{b_ll_std:.3f}  {m_ll - b_ll:+.3f} "
        f"{'v' if m_ll < b_ll else 'x'}"
    )
    typer.echo(
        f"{'Brier':>14s}  {m_br:.4f}+/-{m_br_std:.4f}  "
        f"{b_br:.4f}+/-{b_br_std:.4f}  {m_br - b_br:+.4f} "
        f"{'v' if m_br < b_br else 'x'}"
    )
    typer.echo(
        f"{'Top-1':>14s}  {m_t1:.1%}+/-{m_t1_std:.1%}  "
        f"{b_t1:.1%}+/-{b_t1_std:.1%}  {m_t1 - b_t1:+.1%} "
        f"{'v' if m_t1 > b_t1 else 'x'}"
    )


# ---------------------------------------------------------------
# backfill:stats-json
# ---------------------------------------------------------------
@app.command("backfill:stats-json")
def backfill_stats_json_cmd() -> None:
    """Backfill stats_json from disk program snapshots (no API calls)."""
    from scripts.backfill_stats_json import backfill_stats_json

    backfill_stats_json()


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
