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

## 重要な制約
- 公開ページのみ使用。ログイン・課金壁の回避禁止
- 低負荷: 1-3秒ジッタ、指数バックオフ、User-Agent明示
- ページ本文の転載禁止。DBには数値データのみ
