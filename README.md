# autorace-exacta

オートレース 2連単（Exacta）確率予測。川口・山陽など複数場の公開データを収集し、LightGBM + Plackett-Luce モデルで2連単の確率を算出する。

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

# 6. stats_json バックフィル (v16 学習前に必要、ディスクのみ)
docker compose run --rm worker python3 scripts/backfill_stats_json.py

# 7. モデル学習 (v16 推奨)
docker compose run --rm worker python -m app.cli train:model-v16 \
  --from 2025-06-01 --to 2026-01-31 --out models/model_v16_lgb.pkl

# 8. 予測
docker compose run --rm worker python -m app.cli predict:exacta \
  --track kawaguchi --date auto --skip-if-no-meet \
  --model models/model_v16_lgb.pkl --model-version v16

# 9. API ヘルスチェック
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
| `train:model` | 過去データからモデルを学習 (legacy) |
| `train:model-v12` | v12 モデル学習 (9特徴量, 時系列CV) |
| `train:model-v13` | v13 モデル学習 (12特徴量 + Platt + market blend) |
| `train:model-v14` | v14 モデル学習 (15特徴量 + Platt + market blend) |
| `train:model-v15` | v15 モデル学習 (37特徴量, ペアワイズ) |
| `train:model-v16` | v16 モデル学習 (21特徴量 + Platt + market blend) **推奨** |
| `predict:exacta` | 予測を実行・格納 (モデル自動検出) |
| `backtest:exacta` | バックテスト (全既存レース対象) |
| `evaluate:exacta` | Walk-forward 評価 (市場ベースライン比較, v16) |
| `backfill:stats-json` | stats_json をディスクスナップショットから充填 |
| `recommend:purchase` | Kelly Criterion に基づく購入推奨 |

## 日付解決とスキップ動作

- `--date` は `YYYY-MM-DD` に加えて `auto` / `latest` / `today` を指定可能
  - `auto` / `latest`: 今日が開催なら今日、開催でなければ過去 14 日まで遡って開催日を探索
  - `today`: 今日が開催でなければ `None` としてスキップ
- `--skip-if-no-meet` が有効な場合、開催なし/解決失敗は何もせず exit 0（ログに理由を出力）
- `fetch:odds` は同日で直近 3 分以内に取得済みのオッズがあれば `skip (already fresh)`

## cron 運用例 (5分)

```bash
*/5 * * * * cd /app && python -m app.cli fetch:odds --track kawaguchi --date today --skip-if-no-meet
*/5 * * * * cd /app && python -m app.cli predict:exacta --track kawaguchi --date today --skip-if-no-meet --model models/model_v16_lgb.pkl --model-version v16
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
  --model models/model_v16_lgb.pkl --model-version v16
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

1. 各車 i の特徴量から勝率 p_i を算出（LightGBM）
2. Platt calibration で確率を補正（オプション）
3. 1着確率: `p1(i) = p_i / Σ p_k`
4. 2着確率: `p2(j|i) = p_j / Σ_{k≠i} p_k`
5. Exacta 確率: `prob(i→j) = p1(i) × p2(j|i)`
6. Market blend: `P = alpha * P_model + (1-alpha) * P_market`（alpha は val data で最適化）

### モデルバージョン

| バージョン | 特徴量数 | 追加要素 | ファイル |
|-----------|---------|---------|---------|
| **v16 (推奨)** | 21 | v14 + API stats + race context | `modeling_v16.py` |
| v15 | 37 (pair) | ペアワイズスコアリング | `modeling_v15.py` |
| v14 | 15 | v13 + 選手戦績 (win_rate, place_rate, race_count) | `modeling_v14.py` |
| v13 | 12 | v12 + オッズ由来 + Platt calibration + market blend | `modeling_v13.py` |
| v12 | 9 | v11 + age | `modeling_v12.py` |
| v11 | 8 | 生特徴量ベースライン | `modeling_v11.py` |

### 特徴量一覧 (v16)
```
handicap_m, trial_time, start_avg, deviation,       # 出走表 (4)
quinella_rate, trio_rate, rank_class, car_no,        # 出走表 (4)
age,                                                  # 選手属性 (1)
implied_win_prob, log_implied_win_odds, odds_rank,    # オッズ由来 (3)
win_rate, place_rate, race_count,                     # 選手戦績90日 (3)
good_track_trial_avg, good_track_race_avg,            # 良走路実績 (2)
career_win_rate, career_place_rate,                   # 通算戦績 (2)
race_no, n_runners                                    # レースコンテキスト (2)
```

### Walk-Forward 評価結果 (v16, 2025-10〜2026-01, 8 splits)
| 指標 | Model | Baseline (市場) | Delta |
|------|-------|----------------|-------|
| LogLoss | 2.701 | 2.698 | +0.003 |
| Brier | 0.8716 | 0.8703 | +0.0013 |
| Top-1 | 22.2% | 21.9% | +0.3% |

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
      features.py       # 特徴量抽出 (legacy)
      modeling.py       # Plackett-Luce モデル (legacy)
      modeling_v11.py   # v11 LightGBM (8特徴量)
      modeling_v12.py   # v12 LightGBM (9特徴量)
      modeling_v13.py   # v13 LightGBM (12特徴量 + Platt + blend)
      modeling_v14.py   # v14 LightGBM (15特徴量 + Platt + blend)
      modeling_v15.py   # v15 LightGBM (37特徴量, ペアワイズ)
      modeling_v16.py   # v16 LightGBM (21特徴量 + Platt + blend)
      training.py       # 学習パイプライン (v12〜v16, 時系列CV)
      walkforward.py    # Walk-forward 評価エンジン
      backtest.py       # バックテスト
      racer_stats.py    # 選手戦績 (90日集計)
      evaluation.py     # LogLoss / Brier
      betting.py        # Kelly Criterion 購入推奨
    api/
      main.py           # FastAPI app
      routes.py         # エンドポイント
      schemas.py        # Pydantic スキーマ
  scripts/
    backfill_stats_json.py  # stats_json バックフィル (ディスクから)
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

## 用語

⏺ 用語解説
  適正オッズ (Fair Odds)
   
  モデルが算出した確率から計算した「本来あるべきオッズ」です。

  適正オッズ = 1 ÷ 確率

  例: 7-5 の確率が 2.09% (0.0209) の場合
  適正オッズ = 1 ÷ 0.0209 = 47.9

  つまり「モデルの計算では47.9倍が妥当」という意味です。

  ---
  市場オッズ (Market Odds)

  実際に投票所で提示されているオッズです。これは実際の投票金額の分布で決まります。

  例: 7-5 の市場オッズが 534.1 倍
  → 100円賭けて当たれば 53,410円 の払戻し

  ---
  EV (Expected Value / 期待値)

  「長期的に見て得か損か」を示す指標です。

  EV = (確率 × 市場オッズ) - 1

  例: 7-5 の場合
  EV = (0.0209 × 534.1) - 1 = 11.16 - 1 = +10.16 (+1016%)
  ┌───────┬───────────────────────────────────────────────────┐
  │  EV   │                       意味                        │
  ├───────┼───────────────────────────────────────────────────┤
  │ +100% │ 100円賭けると平均200円返ってくる（100円の利益）   │
  ├───────┼───────────────────────────────────────────────────┤
  │ 0%    │ トントン（損も得もない）                          │
  ├───────┼───────────────────────────────────────────────────┤
  │ -50%  │ 100円賭けると平均50円しか返ってこない（50円の損） │
  └───────┴───────────────────────────────────────────────────┘
  ---
