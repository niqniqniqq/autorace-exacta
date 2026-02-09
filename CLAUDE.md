# CLAUDE.md — autorace-exacta

## プロジェクト概要
オートレース（川口）の公開データを収集し、2連単（Exacta）確率をPlackett-Luceモデルで予測するMVP。

## スタック
- Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, PostgreSQL 16
- 収集: requests (autorace.jp JSON API, CSRF付きPOST)
- ML: LightGBM (推奨), scikit-learn
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
- GET /race_info/Live/{track_code} — CSRF取得用
- POST /race_info/Program — 出走表 (要CSRF)
- POST /race_info/Odds — オッズ (要CSRF)
- POST /race_info/RaceResult — 結果 (要CSRF)
- POST /race_info/RaceRefund — 返還情報 (要CSRF)

### placeCode一覧
| コード | 場名 | 備考 |
|--------|------|------|
| 2 | 川口 | 昼開催 |
| 3 | 伊勢崎 | |
| 4 | 浜松 | |
| 5 | 飯塚 | |
| 6 | 山陽 | |
| 12 | 川口ナイター | base+10 |

### 注意事項
- リクエスト過多でSSLハンドシェイク拒否される場合あり（IP制限）
- 1-3秒のジッタ、指数バックオフで対策済み

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

## 特徴量

### 推奨: 生特徴量8個 (v11)
LightGBMで非線形関係を学習させる。独自補正は不要。
```
handicap_m, trial_time, start_avg, deviation,
quinella_rate, trio_rate, rank_class, car_no
```

### 非推奨: 加工特徴量 (v10以前)
以下は多重共線性・誤学習の原因となった:
- `relative_*` 系: 生値と重複、係数が矛盾
- `adjusted_time`: 独自補正が逆効果
- `handicap_advantage`: 正規化で情報損失

### 実データの傾向 (学習時の参考)
- **試走1位が44%勝利** — 最重要シグナル
- **40mハンデが最多勝(26%)** — 0mではない
- 線形モデルでは非線形関係を捉えられない

## モデル

### 推奨: v11 (LightGBM)
```
models/model_v11_lgb.pkl — 8生特徴量, 1着的中率64%
```
設定:
- n_estimators=100, max_depth=4, num_leaves=15
- min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0

### 非推奨 (過去バージョン)
| モデル | 問題点 |
|--------|--------|
| v10 (LogisticRegression) | 23特徴量で多重共線性、0/6的中 |
| v7 (LightGBM) | 正則化不足で過学習 |
| v2_separate | 効果不明確 |

### 特徴量エンジニアリングの教訓
1. **生データを使う**: 相対値・正規化は多重共線性を生む
2. **非線形はモデルに任せる**: 独自補正より木モデルの方が賢い
3. **少ない特徴量で十分**: 8個 > 23個

## 予測のベストプラクティス
1. **発走直前にプログラム再取得** — 試走タイムは発走前に発表される
2. **オッズ確定後に予測** — 全組合せ揃ってから (昼8頭=56組, ナイト7頭=42組)
3. **選手履歴が効く** — 過去実績のある選手ほど予測精度が高い
4. **placeCode注意** — ナイト(kawaguchi2)は12、昼(kawaguchi)は2

### 購入判断基準 (EV-based)
予測結果から購入すべきかを判断：

| 条件 | 判断 |
|------|------|
| トップ組合せのオッズ < 3.0 | **スキップ** (低オッズでBOX負け) |
| EV+ (prob × odds > 1) の組合せがない | **スキップ** |
| EV+ の組合せあり | **EV+のみ購入** |

```bash
# 予測フロー例 (v11モデル)
docker compose run --rm worker python -m app.cli fetch:program --track sanyou --date today
docker compose run --rm worker python -m app.cli fetch:odds --track sanyou --date today
docker compose run --rm worker python -m app.cli predict:exacta --track sanyou --date today \
  --model models/model_v11_lgb.pkl --model-version v11
```

## 重要な制約
- 公開ページのみ使用。ログイン・課金壁の回避禁止
- 低負荷: 1-3秒ジッタ、指数バックオフ、User-Agent明示
- ページ本文の転載禁止。DBには数値データのみ
