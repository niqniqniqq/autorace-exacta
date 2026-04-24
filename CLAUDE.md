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

### v18/v19/v20 特徴量 (22個, 現行最新 — レース内相対 + 非線形交互作用)
```
# v17 base (16): オッズフリー
handicap_m, trial_time, start_avg, deviation,
quinella_rate, trio_rate, rank_class, car_no, age,
win_rate, place_rate, race_count,
good_track_trial_avg, good_track_race_avg, career_win_rate, career_place_rate

# レース内相対 (4): NEW
trial_time_rel,         # trial_time - mean(全選手) → 負=場内最速寄り
deviation_rel,          # deviation - mean(全選手) → 正=場内強い
field_strength,         # mean(全選手のdeviation) → 高=ハイレベル戦
trial_time_best_diff    # trial_time - min(全選手) → 0=最速

# 非線形交互作用 (2): NEW
form_delta,             # trial_time - good_track_trial_avg → 負=調子上昇
handicap_deviation_ratio # handicap_m / (deviation + 1) → 低=能力の割にハンデ有利
```
仮説: 市場は個人の絶対値は織り込み済みだが、レース内の相対位置やフォーム変化は見落としがち

### v17 特徴量 (16個 — オッズフリー)
```
# v12 base (9): 生特徴量 + 選手属性
handicap_m, trial_time, start_avg, deviation,
quinella_rate, trio_rate, rank_class, car_no, age

# v14 追加 (3): 選手戦績 (90日)
win_rate, place_rate, race_count

# v16 API stats (4): API由来
good_track_trial_avg, good_track_race_avg,   # latest90List
career_win_rate, career_place_rate            # winList
```
除去: implied_win_prob, log_implied_win_odds, odds_rank (市場と直交するシグナルを生成するため)

### v16 特徴量 (21個)
```
# v11 base (8): 生特徴量
handicap_m, trial_time, start_avg, deviation,
quinella_rate, trio_rate, rank_class, car_no

# v12 追加 (1): 選手属性
age

# v13 追加 (3): オッズ由来
implied_win_prob, log_implied_win_odds, odds_rank

# v14 追加 (3): 選手戦績 (90日)
win_rate, place_rate, race_count

# v16 追加 (6): API未活用データ + レースコンテキスト
good_track_trial_avg, good_track_race_avg,   # latest90List
career_win_rate, career_place_rate,           # winList
race_no, n_runners                            # レースコンテキスト
```

### 非推奨: 加工特徴量 (v10以前)
以下は多重共線性・誤学習の原因となった:
- `relative_*` 系: 生値と重複、係数が矛盾
- `adjusted_time`: 独自補正が逆効果
- `handicap_advantage`: 正規化で情報損失

### アブレーション結果 (不採用)
| 候補 | 内容 | 結果 |
|------|------|------|
| race ranks | trial_time_rank, handicap_rank, deviation_rank (15→18) | LogLoss delta 変化なし (+0.002)、revert |
| v16 API stats | good_track_*, career_*, race_no, n_runners (15→21) | LogLoss delta +0.003 (v14同等)、要ablation |

### 実データの傾向 (学習時の参考)
- **試走1位が44%勝利** — 最重要シグナル
- **40mハンデが最多勝(26%)** — 0mではない
- 線形モデルでは非線形関係を捉えられない

## モデル

### 推奨: v20 (Multi-Track LightGBM)
```
models/model_v20_lgb.pkl — track_codeごとに独立したv19モデル + フォールバック
```
- ファイル: `app/services/modeling_v20.py`
- **設計**: 各場ごとに独立した ExactaModelV19 を学習。場の特性（外枠有利度・ハンデ効果）を個別に捉える
- **フォールバック**: min_track_races(=100)未満の場は全場合算モデルを使用
- **predict_exacta**: `track_code` パラメータが必要
- 場別 alpha_map 例 (2026-04-23学習):
  - iizuka: alpha=0.95〜1.00（モデル全信頼）
  - kawaguchi: low=0.15（低オッズは市場優先）、high=0.70
  - sanyou: mid=0.35（中間帯はバランス）
- Backtest ROI: **233.7%** (全期間, min_ev=0.15) / 226.8% (2026-02以降)

### v19 (LightGBM, Isotonic Calibration + Conditional Alpha)
```
models/model_v19_lgb.pkl — 22特徴量 (v18同一) + Isotonic calibration + Conditional alpha
```
- ファイル: `app/services/modeling_v19.py`
- **Isotonic regression**: Plattシグモイドを置換、全fold OOFデータでフィット
- **Conditional alpha**: オッズ区間別にブレンド比を最適化（low/mid/high/extreme）
- Backtest ROI: 86.2% (min_ev=0.15, 全期間)

### v18 (LightGBM, Race-Relative + Interactions)
```
models/model_v18_lgb.pkl — 22特徴量 (レース内相対 + 交互作用) + Platt calibration + market blend
```
- v17ベース (オッズフリー16) + レース内相対4 + 非線形交互作用2 = 22特徴量
- alpha=0.35, Backtest ROI=60.6%

### モデルバージョン一覧
| バージョン | 特徴量数 | 主な追加要素 | model_type | ファイル |
|-----------|---------|-------------|------------|---------|
| **v20** | 22 | 場別独立モデル (multi-track) | `v20_lgb` | `modeling_v20.py` |
| v19 | 22 | Isotonic cal + Conditional alpha | `v19_lgb` | `modeling_v19.py` |
| v18 | 22 | レース内相対 + 非線形交互作用 | `v18_lgb` | `modeling_v18.py` |
| v17 | 16 | オッズフリー (市場直交) | `v17_lgb` | `modeling_v17.py` |
| v16 | 21 | API stats + race context | `v16_lgb` | `modeling_v16.py` |
| v15 | 37 (pair) | pairwise scoring | `v15_pair` | `modeling_v15.py` |
| v14 | 15 | 選手戦績 (win_rate, place_rate, race_count) | `v14_lgb` | `modeling_v14.py` |
| v13 | 12 | オッズ由来 + Platt + market blend | `v13_lgb` | `modeling_v13.py` |
| v12 | 9 | age | `v12_lgb` | `modeling_v12.py` |
| v11 | 8 | 生特徴量ベースライン | `v11_lgb` | `modeling_v11.py` |

### 非推奨 (過去バージョン)
| モデル | 問題点 |
|--------|--------|
| v10 (LogisticRegression) | 23特徴量で多重共線性、0/6的中 |
| v7 (LightGBM) | 正則化不足で過学習 |
| v2_separate | 効果不明確 |

### モデル自動検出
pickle内の `model_type` フィールドで自動判別:
- 検出順: v20 → v19 → v18 → v17 → v16 → v15 → v14 → v13 → v12 → v11 → legacy
- predict:exacta / backtest:exacta / evaluate:exacta すべて対応

### 特徴量エンジニアリングの教訓
1. **生データを使う**: 相対値・正規化は多重共線性を生む
2. **非線形はモデルに任せる**: 独自補正より木モデルの方が賢い
3. **少ない特徴量で十分**: 8個 > 23個
4. **保守的ハイパーパラメータ**: max_depth=4, num_leaves=15 が 15特徴量でも安定
5. **レース内順位特徴は効かない**: LightGBMが木分割で既に相対比較を学習済み
6. **オッズ特徴量は市場と相関**: モデル入力にオッズ→ブレンド時に同じ信号の平均→エッジゼロ。v17で除去

## 予測のベストプラクティス
1. **発走直前にプログラム再取得** — 試走タイムは発走前に発表される
2. **オッズ確定後に予測** — 全組合せ揃ってから (昼8頭=56組, ナイト7頭=42組)
3. **選手履歴が効く** — 過去実績のある選手ほど予測精度が高い
4. **placeCode注意** — ナイト(kawaguchi2)は12、昼(kawaguchi)は2

### 購入判断基準 (EV-based)
予測結果から購入すべきかを判断：

| 条件 | 判断 |
|------|------|
| EV+なし | **スキップ** |
| 【本命帯EV+】あり | 本命帯を購入 |
| 【穴帯EV+】あり | 穴帯を購入（リスク高・高配当狙い） |

predict:exacta の出力フォーマット（v20〜）:
```
【本命帯EV+】 オッズ<20倍 かつ EV>0.15 をEV降順で最大3件
【穴帯EV+】   オッズ≥20倍 かつ EV>0.15 をEV降順で最大3件
【EV+なし/トップ3】 該当なしの場合は確率上位3件
```

```bash
# 予測フロー例 (v20モデル, 試走後に再取得)
docker compose run --rm worker python -m app.cli fetch:program --track iizuka --date today
docker compose run --rm worker python -m app.cli fetch:odds --track iizuka --date today [--force]
docker compose run --rm worker python -m app.cli predict:exacta --track iizuka --date today \
  --model models/model_v20_lgb.pkl --model-version v20
```

## Walk-Forward 評価
ローリングウィンドウで学習→検証→テストを繰り返し、市場ベースラインと比較:

```bash
# v18 (デフォルト)
docker compose run --rm worker python3 -m app.cli evaluate:exacta \
  --from 2025-10-01 --to 2026-01-31 --train-days 60 --test-days 7

# v17
docker compose run --rm worker python3 -m app.cli evaluate:exacta \
  --from 2025-10-01 --to 2026-01-31 --train-days 60 --test-days 7 --version v17

# v16
docker compose run --rm worker python3 -m app.cli evaluate:exacta \
  --from 2025-10-01 --to 2026-01-31 --train-days 60 --test-days 7 --version v16
```

出力指標:
- **LogLoss**: ペアレベルの対数損失 (市場ベースラインとの差分)
- **Brier**: ペアレベルのBrierスコア
- **Top-1**: 予測1位が的中した割合

### 最新 Walk-Forward 結果 (v16, 2025-10〜2026-01, 8 splits)
| 指標 | Model | Baseline | Delta |
|------|-------|----------|-------|
| LogLoss | 2.701 | 2.698 | +0.003 |
| Brier | 0.8716 | 0.8703 | +0.0013 |
| Top-1 | 22.2% | 21.9% | +0.3% |

市場ベースラインとほぼ同等。market_alpha は 0.00〜0.10 で推移し、モデル独自シグナルはまだ弱い。

### 過去 Walk-Forward 結果
| モデル | LogLoss (Model) | LogLoss (Baseline) | Delta |
|--------|----------------|-------------------|-------|
| v16 (21特徴量) | 2.701 | 2.698 | +0.003 |
| v14 (15特徴量) | 2.699 | 2.698 | +0.002 |

## バックテスト
オッズと結果が両方揃っているレースで収益シミュレーション:

```bash
docker compose run --rm worker python -m app.cli backtest:exacta \
  --model models/model_v16_lgb.pkl
```

出力:
- Top-1的中率: 予測1位が的中した割合
- EV+的中率: EV+ベットのいずれかが的中した割合
- ROI: 回収率 (total_return / total_invested)
- 各レースの予測 vs 実際の結果

データ蓄積: `fetch:odds` と `fetch:results` を継続実行してバックテストデータを増やす

## 学習コマンド

```bash
# stats_json backfill (学習前に必要)
docker compose run --rm worker python3 scripts/backfill_stats_json.py

# v20 モデル学習 (推奨, 場別独立モデル)
docker compose run --rm worker python -m app.cli train:model-v20 \
  --from 2025-06-01 --to 2026-04-23 --out models/model_v20_lgb.pkl

# v19 モデル学習 (単一モデル, Isotonic + Conditional Alpha)
docker compose run --rm worker python -m app.cli train:model-v19 \
  --from 2025-06-01 --to 2026-04-23 --out models/model_v19_lgb.pkl

# バックテスト (min_ev=0.15推奨)
docker compose run --rm worker python -m app.cli backtest:exacta \
  --model models/model_v20_lgb.pkl --min-ev 0.15
```

## 重要な制約
- 公開ページのみ使用。ログイン・課金壁の回避禁止
- 低負荷: 1-3秒ジッタ、指数バックオフ、User-Agent明示
- ページ本文の転載禁止。DBには数値データのみ
