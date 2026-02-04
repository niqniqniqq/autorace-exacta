# autorace-exacta

オートレース 2連単（Exacta）確率予測 MVP。川口オートレースの公開データを収集し、Plackett-Luce モデルで2連単の確率を算出する。

**kawaguchi と kawaguchi2（ナイト）は別 track として DB 上も完全に分離されます。**

## 前提条件

- Docker & Docker Compose
- (開発時) Python 3.12

## セットアップ

```bash
# .env を作成
cp .env.example .env

# コンテナ起動
docker compose up -d --build

# DB マイグレーション（api 起動時に自動実行される）
```

## 実行デモ

```bash
# 1. コンテナ起動
docker compose up -d --build

# 2. race_day を同期
docker compose run --rm worker python -m app.cli sync:race-days \
  --track kawaguchi --from 2026-01-01 --to 2026-02-14

# 3. 出走表（Program）を取得
docker compose run --rm worker python -m app.cli fetch:program \
  --track kawaguchi --date auto --skip-if-no-meet

# 4. オッズ（Exacta Odds）を取得
docker compose run --rm worker python -m app.cli fetch:odds \
  --track kawaguchi --date auto --skip-if-no-meet

# 5. 結果・払戻を取得
docker compose run --rm worker python -m app.cli fetch:results \
  --track kawaguchi --date auto --skip-if-no-meet

# 6. モデル学習
docker compose run --rm worker python -m app.cli train:model \
  --from 2026-01-01 --to 2026-02-14 --out models/model.pkl

# 7. 予測
docker compose run --rm worker python -m app.cli predict:exacta \
  --track kawaguchi --date auto --skip-if-no-meet \
  --model models/model.pkl --model-version v0

# 8. API ヘルスチェック
curl http://localhost:8000/health
```

## API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/race-days?track=kawaguchi&from=YYYY-MM-DD&to=YYYY-MM-DD` | 開催日一覧 |
| GET | `/races/{race_id}` | レース詳細（出走表含む） |
| GET | `/races/{race_id}/predictions?top=10&min_ev=0` | 2連単予測 |

## CLI コマンド

| コマンド | 説明 |
|----------|------|
| `sync:race-days` | 指定日付範囲の race_day レコードを作成 |
| `fetch:program` | 出走表を取得・格納 |
| `fetch:odds` | 2連単オッズを取得・格納 |
| `fetch:results` | 着順と払戻を取得・格納 |
| `train:model` | 過去データからモデルを学習 |
| `predict:exacta` | 予測を実行・格納 |

## 日付解決とスキップ動作

- `--date` は `YYYY-MM-DD` に加えて `auto` / `latest` / `today` を指定可能
  - `auto` / `latest`: 今日が開催なら今日、開催でなければ過去 14 日まで遡って開催日を探索
  - `today`: 今日が開催でなければ `None` としてスキップ
- `--skip-if-no-meet` が有効な場合、開催なし/解決失敗は何もせず exit 0（ログに理由を出力）
- `fetch:odds` は同日で直近 3 分以内に取得済みのオッズがあれば `skip (already fresh)`

## cron 運用例 (5分)

```bash
*/5 * * * * cd /app && python -m app.cli fetch:odds --track kawaguchi --date today --skip-if-no-meet
*/5 * * * * cd /app && python -m app.cli predict:exacta --track kawaguchi --date today --skip-if-no-meet --model models/model.pkl --model-version v0
```

## ナイトレース (kawaguchi2) の運用例

```bash
TRACK=kawaguchi2

docker compose run --rm worker python -m app.cli fetch:program \
  --track $TRACK --date 2026-02-04 --skip-if-no-meet

docker compose run --rm worker python -m app.cli fetch:odds \
  --track $TRACK --date 2026-02-04 --skip-if-no-meet

docker compose run --rm worker python -m app.cli fetch:results \
  --track $TRACK --date 2026-02-04 --skip-if-no-meet

docker compose run --rm worker python -m app.cli predict:exacta \
  --track $TRACK --date 2026-02-04 --skip-if-no-meet \
  --model models/model.pkl --model-version v0
```

> **注意:** `fetch:program` / `fetch:odds` / `fetch:results` / `predict:exacta` は同一 track を揃えて実行してください。kawaguchi と kawaguchi2 は別 track として DB 上も完全に分離されます。

## テスト

```bash
# コンテナ内で実行
docker compose run --rm worker pytest tests/ -v

# ローカル（要 pip install -e ".[dev]"）
pytest tests/ -v
```

## 予測アルゴリズム

Plackett-Luce 風のモデル:

1. 各車 i の特徴量からスコア u_i を算出（LogisticRegression）
2. 1着確率: `p1(i) = exp(u_i) / Σ exp(u_k)`
3. 2着確率: `p2(j|i) = exp(u_j) / Σ_{k≠i} exp(u_k)`
4. Exacta 確率: `prob(i→j) = p1(i) × p2(j|i)`

特徴量: handicap_m, trial_time, deviation, quinella_rate, trio_rate

## データ保存方針

- API 応答の raw JSON は `data/snapshots/` 配下に gzip 圧縮保存
- DB には数値データとメタ情報のみ格納（本文転載なし）
- `content_hash`（SHA-256）で同一内容の重複保存をスキップ

## リポジトリ構成

```
autorace-exacta/
  app/
    config.py           # 設定 (pydantic-settings)
    cli.py              # Typer CLI
    __main__.py
    db/
      base.py           # DeclarativeBase
      session.py        # Engine / Session
      models.py         # 全テーブル定義
      migrations/       # Alembic
    scraping/
      http.py           # HTTP クライアント（レート制限・CSRF）
      sources/          # データ取得
      parsers/          # JSON パーサ
    services/
      storage.py        # スナップショット保存
      upsert.py         # Idempotent upsert
      features.py       # 特徴量抽出
      modeling.py       # Plackett-Luce モデル
      evaluation.py     # LogLoss / Brier
    api/
      main.py           # FastAPI app
      routes.py         # エンドポイント
      schemas.py        # Pydantic スキーマ
  tests/
  docker/
  docker-compose.yml
  data/                 # スナップショット (gitignore)
  models/               # 学習済みモデル (gitignore)
```

## AWS 移行メモ

ローカル MVP から AWS への移行計画:

| コンポーネント | ローカル | AWS |
|---------------|---------|-----|
| DB | PostgreSQL (docker) | **RDS PostgreSQL 16** (db.t4g.micro → small) |
| API | FastAPI (docker) | **ECS Fargate** (API タスク) + ALB |
| Worker | docker compose run | **ECS Fargate** (タスク定義) + **EventBridge Scheduler** で定期実行 |
| スナップショット | `data/` ローカル | **S3** (`s3://bucket/snapshots/...`). storage_uri を `s3://` に切替 |
| モデル | `models/` ローカル | **S3** (`s3://bucket/models/...`) |
| ログ | stderr | **CloudWatch Logs** (ECS 統合) |
| シークレット | `.env` | **Secrets Manager** or **SSM Parameter Store** |
| 監視 | なし | **CloudWatch Metrics** + アラーム (API latency, scraping failure) |
| CI/CD | なし | **GitHub Actions** → ECR push → ECS deploy |

移行手順:
1. RDS インスタンス作成、Alembic で migration 実行
2. ECR リポジトリ作成、Docker イメージ push
3. ECS クラスタ・サービス (API) 作成、ALB 設定
4. EventBridge Scheduler で `fetch:program` / `fetch:odds` / `fetch:results` を日次実行
5. S3 バケット作成、`storage.py` の保存先を S3 対応に拡張
6. CloudWatch ダッシュボード作成
