#!/usr/bin/env bash
set -euo pipefail

# ====== 設定 ======
REPO_DIR="${REPO_DIR:-$HOME/autorace-exacta}"   # リポジトリ配置先に合わせて変更
TRACK="${TRACK:-kawaguchi}"

# 予測に使うモデル（無ければ predict はスキップする）
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/model.pkl}"
MODEL_VERSION="${MODEL_VERSION:-v0}"

# 日付解決モード（Codexで実装した前提）
# today: 今日開催が無ければスキップ（無駄アクセスを避けたいcron向け）
# auto : 今日が無ければ過去N日遡って最新開催日（深夜補完向け）
DATE_MODE_FAST="${DATE_MODE_FAST:-today}"
DATE_MODE_BACKFILL="${DATE_MODE_BACKFILL:-auto}"

# cronログ
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
mkdir -p "$LOG_DIR"

# ロック（cron多重起動防止）
LOCK_FILE="${LOCK_FILE:-/tmp/autorace-exacta.lock}"

# Docker Compose（v2想定）
DC="${DC:-docker compose}"

# ====== 共通関数 ======
ts() { date "+%Y-%m-%d %H:%M:%S"; }

run_worker() {
  # 例: run_worker python -m app.cli fetch:odds --track ... --date today --skip-if-no-meet
  (cd "$REPO_DIR" && $DC run --rm worker "$@")
}

# ====== 引数 ======
JOB="${1:-}"
if [[ -z "$JOB" ]]; then
  echo "Usage: $0 {sync|program_fast|odds_predict_fast|results_fast|train_weekly|backfill_daily}" >&2
  exit 2
fi

# ====== ロック（flock） ======
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(ts) already running, skip: $JOB" >> "$LOG_DIR/cron.log"
  exit 0
fi

echo "$(ts) start job=$JOB track=$TRACK" >> "$LOG_DIR/cron.log"

case "$JOB" in
  sync)
    # today対象（開催が無ければ内部でスキップさせる運用）
    run_worker python -m app.cli sync:race-days --track "$TRACK" --from "$(date +%F)" --to "$(date +%F)" \
      >> "$LOG_DIR/sync.log" 2>&1
    ;;

  program_fast)
    # 開催が無い日はスキップ（exit 0想定）
    run_worker python -m app.cli fetch:program --track "$TRACK" --date "$DATE_MODE_FAST" --skip-if-no-meet \
      >> "$LOG_DIR/program.log" 2>&1
    ;;

  odds_predict_fast)
    # 5分頻度で回す本体。開催が無ければスキップ。
    # fetch:odds 側で “fresh-skip（例: 3分以内に取得済みならskip）” が効く前提。
    run_worker python -m app.cli fetch:odds --track "$TRACK" --date "$DATE_MODE_FAST" --skip-if-no-meet \
      >> "$LOG_DIR/odds.log" 2>&1

    if [[ -f "$MODEL_PATH" ]]; then
      run_worker python -m app.cli predict:exacta --track "$TRACK" --date "$DATE_MODE_FAST" --skip-if-no-meet \
        --model "$MODEL_PATH" --model-version "$MODEL_VERSION" \
        >> "$LOG_DIR/predict.log" 2>&1
    else
      echo "$(ts) model not found, skip predict: $MODEL_PATH" >> "$LOG_DIR/predict.log"
    fi
    ;;

  results_fast)
    # 結果は当日夜に回す。開催なしならスキップ。
    run_worker python -m app.cli fetch:results --track "$TRACK" --date "$DATE_MODE_FAST" --skip-if-no-meet \
      >> "$LOG_DIR/results.log" 2>&1
    ;;

  train_weekly)
    # 週次学習（直近30日など）
    TODAY="$(date +%F)"
    FROM="$(date -d "$TODAY -30 days" +%F)"
    TO="$TODAY"
    mkdir -p "$REPO_DIR/models"
    run_worker python -m app.cli train:model --from "$FROM" --to "$TO" --out "$MODEL_PATH" \
      >> "$LOG_DIR/train.log" 2>&1
    ;;

  backfill_daily)
    # 深夜補完：最新開催日を自動選択（auto）で program/odds/results を一通り回して取りこぼしを埋める
    run_worker python -m app.cli fetch:program --track "$TRACK" --date "$DATE_MODE_BACKFILL" --skip-if-no-meet \
      >> "$LOG_DIR/program.log" 2>&1

    run_worker python -m app.cli fetch:odds --track "$TRACK" --date "$DATE_MODE_BACKFILL" --skip-if-no-meet \
      >> "$LOG_DIR/odds.log" 2>&1

    run_worker python -m app.cli fetch:results --track "$TRACK" --date "$DATE_MODE_BACKFILL" --skip-if-no-meet \
      >> "$LOG_DIR/results.log" 2>&1

    if [[ -f "$MODEL_PATH" ]]; then
      run_worker python -m app.cli predict:exacta --track "$TRACK" --date "$DATE_MODE_BACKFILL" --skip-if-no-meet \
        --model "$MODEL_PATH" --model-version "$MODEL_VERSION" \
        >> "$LOG_DIR/predict.log" 2>&1
    fi
    ;;
  *)
    echo "Unknown job: $JOB" >&2
    exit 2
    ;;
esac

echo "$(ts) done job=$JOB" >> "$LOG_DIR/cron.log"
