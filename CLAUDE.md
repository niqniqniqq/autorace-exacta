# CLAUDE.md — autorace-exacta

## プロジェクト概要
オートレース（川口）の公開データを収集し、2連単（Exacta）確率をPlackett-Luceモデルで予測するMVP。

## スタック
- Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, PostgreSQL 16
- 収集: requests (autorace.jp JSON API, CSRF付きPOST)
- ML: scikit-learn (LogisticRegression + Plackett-Luce)
- CLI: Typer, コンテナ: docker-compose

## よく使うコマンド
```bash
docker compose up -d --build           # 起動
docker compose build worker            # workerのみ再ビルド (profiles: [tools]のため)
docker compose run --rm worker pytest tests/ -v  # テスト
docker compose run --rm worker python -m app.cli --help  # CLI
alembic upgrade head                   # マイグレーション
```

## コード規約
- 型ヒント必須 (from __future__ import annotations)
- パーサは app/scraping/parsers/ に集中管理
- DB操作は app/services/upsert.py のupsert関数経由 (idempotent)
- raw APIレスポンスは data/snapshots/ にgzip保存、DBには数値のみ

## autorace.jp API
- POST /race_info/Program — 出走表 (要CSRF)
- POST /race_info/Odds — オッズ (要CSRF)
- POST /race_info/RaceResult — 結果 (要CSRF)
- placeCode: kawaguchi=2, isesaki=3, hamamatsu=4, iizuka=5, sanyou=6
- ナイト開催 (kawaguchi2, isesaki2) は同じplaceCodeを使用、Track別管理

## CLI日付解決
- `--date` は `YYYY-MM-DD` / `auto` / `latest` / `today` を受け付ける
- 日付解決ロジックは app/services/date_resolver.py に集約
- オッズ鮮度チェックは app/services/odds_freshness.py (デフォルト3分)
- cron運用時は `--skip-if-no-meet` でexit 0スキップ

## アクティブレース解決
- app/services/meet_resolver.py の `resolve_active_race_nos()` でレース番号を動的取得
- Program APIを1〜14までプローブし、3連続空でストップ
- 全CLIコマンド (fetch:program, fetch:odds, fetch:results, predict:exacta) で自動適用
- max_race_no のデフォルトは14（川口ナイトは12レース、昼開催も可変）

## 特徴量 (app/services/features.py)
### 基本特徴量
- handicap_m, trial_time, deviation, quinella_rate, trio_rate

### 相対特徴量 (レース内での相対位置)
- relative_handicap: 最小ハンデとの差
- relative_trial_time: 最速試走との差
- car_position: 車番位置 (1番車=1.0, 8番車=0.0)
- handicap_advantage: ハンデ有利度 (0mが最有利=1.0)

### 選手履歴特徴量 (app/services/racer_stats.py)
- hist_win_rate: 過去90日の勝率
- hist_place_rate: 過去90日の2連率
- hist_show_rate: 過去90日の3連率
- hist_avg_finish: 過去90日の平均着順
- hist_race_count: 経験値 (正規化済み)

## モデル
- models/model.pkl — 基本モデル (v1)
- models/model_v2.pkl — 相対特徴量追加
- models/model_v3.pkl — 選手履歴追加 (推奨)

## 予測のベストプラクティス
1. **発走直前にプログラム再取得** — 試走タイムは発走前に発表される
2. **オッズ確定後に予測** — 全56組揃ってから実行
3. **選手履歴が効く** — 過去実績のある選手ほど予測精度が高い

```bash
# 予測フロー例
docker compose run --rm worker python -m app.cli fetch:program --track kawaguchi --date today
docker compose run --rm worker python -m app.cli fetch:odds --track kawaguchi --date today --race-no 12
docker compose run --rm worker python -m app.cli predict:exacta --track kawaguchi --date today --model models/model_v3.pkl
```

## 重要な制約
- 公開ページのみ使用。ログイン・課金壁の回避禁止
- 低負荷: 1-3秒ジッタ、指数バックオフ、User-Agent明示
- ページ本文の転載禁止。DBには数値データのみ
