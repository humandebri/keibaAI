# Keiba AI - 競馬予測AIシステム

## 📋 目次

- [概要](#概要)
- [主な特徴](#主な特徴)
- [使用方法](#使用方法)
  - [🎯 クイックスタート](#-クイックスタート今すぐ実行したい方へ)
  - [📋 ステップバイステップガイド](#-ステップバイステップガイド初めての方向け)
  - [📊 オプション](#-オプション)
- [🆕 新機能と戦略](#-新機能と戦略)
- [データ仕様](#データ仕様)
- [パフォーマンス](#パフォーマンス)
- [セットアップ](#セットアップ)
- [トラブルシューティング](#トラブルシューティング)

## 概要

Keiba AIは、機械学習を用いて競馬レースの結果を予測する高度なAIシステムです。netkeiba.comから過去のレースデータを自動収集し、競馬ドメイン知識に基づいた特徴量エンジニアリングとLightGBMモデルによる予測を行います。

## 主な特徴

- **データスクレイピング**: netkeiba.comから10年以上のレースデータを自動収集
- **高度な特徴量エンジニアリング**: 競馬ドメイン知識を活用した138以上の特徴量
- **最適化されたLightGBMモデル**: Optunaによるハイパーパラメータ最適化
- **改善されたバックテストシステム**: 複勝ベッティング、Kelly基準、マネーマネジメント
- **モデル解釈性**: SHAP値による予測根拠の可視化

## プロジェクト構造

```
Keiba_AI/
├── README.md                    # このファイル
├── requirements.txt             # Python依存関係
├── config/                      # 設定ファイル
│   ├── config.yaml             # メイン設定
│   ├── standard_deviation.xlsx  # 標準化パラメータ
│   └── jockey_win_rate.xlsx    # 騎手勝率データ
├── src/                         # ソースコード
│   ├── data_processing/        # データ処理モジュール
│   │   ├── data_scraping.py   # データスクレイピング
│   │   └── data_encoding.py   # データエンコーディング
│   ├── modeling/               # モデリングモジュール
│   │   └── model_training.py  # モデル学習と評価
│   ├── backtesting/            # バックテストモジュール
│   │   └── backtest.py        # 改善されたバックテストシステム
│   └── utils/                  # ユーティリティ
├── data/                        # レースデータ (2014-2025)
│   └── [year].xlsx             # 年別データファイル
├── models/                      # 学習済みモデル
├── results/                     # 実行結果
│   ├── backtests/              # バックテスト結果
│   └── shap_importance.png     # SHAP特徴量重要度
└── docs/                        # ドキュメント
    ├── data_schema.md          # データスキーマ
    ├── IMPROVEMENT_SUMMARY.md   # 改善履歴
    └── model.docx              # モデル詳細仕様
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/Keiba_AI.git
cd Keiba_AI
```

### 2. Python環境のセットアップ

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. 設定ファイルの確認

`config/config.yaml`で以下の設定を確認・調整してください：

```yaml
scraping:
  start_year: 2024
  end_year: 2024
  max_workers: 3
  timeout: 120

model:
  test_size: 0.2
  random_state: 42
  n_trials: 100  # Optuna最適化試行数

backtest:
  betting_fraction: 0.005  # Kelly基準ベット率
  monthly_stop_loss: 0.1   # 月間ストップロス
  ev_threshold: 1.2        # 期待値閾値
```

## 機械学習の基本的な流れと各ファイルの役割

### 機械学習プロセス
```
1. データ収集 → 2. データ前処理 → 3. モデル訓練 → 4. 予測・評価
```

### 各ファイルの詳細説明

#### 1. **データ収集** (`src/data_processing/data_scraping.py`)
- **役割**: ネット競馬からレースデータを取得
- **内容**: 
  - レース結果（着順、タイム）
  - 馬の情報（性別、体重）
  - オッズ、天気、馬場状態など
- **出力**: `data/2024.xlsx`などの生データファイル

#### 2. **データ前処理（エンコーディング）** (`src/data_processing/data_encoding.py`)
- **役割**: 生データを機械学習で使える形に変換
- **処理内容**:
  ```python
  # 例：カテゴリカル変数の変換
  性別: "牡" → 0, "牝" → 1, "セ" → 2
  馬場: "良" → 0, "稍重" → 1, "重" → 2
  
  # 数値の正規化
  体重: 450kg → 0.0 (標準化後)
  ```
- **なぜ必要？**: 機械学習モデルは数値しか理解できないため

#### 3. **モデル訓練（トレーニング）** (`src/modeling/model_training.py`)
- **役割**: エンコードされたデータからパターンを学習
- **処理内容**:
  - LightGBMという機械学習アルゴリズムを使用
  - 過去のレース結果から「どの馬が3着以内に入るか」を学習
  - ハイパーパラメータの最適化（Optuna使用）
- **出力**: 学習済みモデル（予測に使う）

#### 4. **バックテスト（評価）** (`src/backtesting/backtest.py`)
- **役割**: 学習したモデルの性能を過去データで検証
- **処理内容**:
  - 2021-2023年のデータで予測
  - 複勝戦略でベッティングシミュレーション
  - 収益性の評価

### エンコーディングとトレーニングの違い

#### エンコーディング（データ前処理）
```python
# 生データ
馬名: "ディープインパクト"
性別: "牡"
馬場: "良"
体重: 486

# ↓ エンコード後
馬番号: 1234
性別: 0  # 牡=0
馬場: 0  # 良=0
体重: 0.5  # 標準化
```
**目的**: データを数値化・正規化

#### トレーニング（モデル学習）
```python
# 入力データ（特徴量）
X = [[馬番, 性別, 馬場, 体重, ...]]  # 過去のレースデータ
# 正解ラベル
y = [1, 0, 1, ...]  # 3着以内=1, それ以外=0

# モデルが学習
model.fit(X, y)
# → 「この条件の馬は3着以内に入りやすい」というパターンを発見
```
**目的**: データからパターンを学習し、予測できるようにする

## 使用方法

### 🎯 クイックスタート（今すぐ実行したい方へ）

```bash
# 1. 仮想環境を有効化
source .venv/bin/activate  # Mac/Linux
# または
.venv\Scripts\activate     # Windows

# 2. メインコマンド実行（最も簡単）
python main.py backtest

# 3. データがない場合は先にデータ収集
python main.py collect --start-year 2024 --end-year 2024
```

### 📋 ステップバイステップガイド（初めての方向け）

#### ステップ1: データがあるか確認
```bash
ls data/
# 2019.xlsx, 2020.xlsx などがあればOK
```

#### ステップ2: データがない場合は収集
```bash
# 2024年のデータを収集（払戻データ付き）
python src/data_processing/enhanced_scraping.py --year 2024

# または中断可能なバージョン（推奨）
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024
```

#### ステップ3: データの前処理
```bash
# 統合CLIを使用（推奨）
python main.py encode --start-year 2022 --end-year 2023

# または直接実行
python src/data_processing/data_encoding.py --start 2022 --end 2023
```

#### ステップ4: モデルの訓練
```bash
python main.py train
```

#### ステップ5: バックテスト実行
```bash
python main.py backtest
```

### 📊 オプション

```bash
# 期待儤1.2以上のみ
python main.py backtest --min-ev 1.2

# 三連単を除外（高速化）
python main.py backtest --no-trifecta

# 期間指定
python main.py backtest --start-year 2021 --end-year 2023
```

### 📑 完全ガイド（データ収集から始める場合）

#### 1. データスクレイピング

##### 🆕 払戻データ付きスクレイピング（推奨）
```bash
# 払戻データを含む拡張版
python src/data_processing/enhanced_scraping.py --year 2024
```

##### 通常のスクレイピング（基本版）
```bash
# 2024年のデータを取得
python src/data_processing/data_scraping.py --start 2024 --end 2024

# 複数年のデータを取得
python src/data_processing/data_scraping.py --start 2022 --end 2024 --workers 5
```

##### チェックポイント機能付きスクレイピング（推奨）
```bash
# 通常の実行（チェックポイントから自動再開）
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024

# 50レースごとに保存
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024 --save-interval 50

# 最初からやり直す
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024 --no-resume

# 複数年のスクレイピング
python src/data_processing/data_scraping_with_checkpoint.py --start 2020 --end 2024
```

**ファイル構造:**
```
Keiba_AI/
└── data_with_payout/
    ├── 2024_interim_20240531_141523.xlsx  # 中間保存
    ├── 2024_interim_20240531_142045.xlsx  # 中間保存
    ├── 2024_with_payout.xlsx             # 最終データ
    └── checkpoints/
        └── checkpoint_2024.pkl            # チェックポイント
```

**中断と再開の例:**
```bash
# 実行中にCtrl+Cで中断
# → checkpoint_2024.pkl が保存される

# 再度実行すると自動的に続きから
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024
# → "チェックポイント読み込み成功: 2024-05-31T14:15:23"
# → "処理済み: 1500/10920 レース"
```

#### 2. データエンコーディング

```bash
# 🆕 払戻データ対応版（推奨）
python src/data_processing/data_encoding_v2.py --start 2022 --end 2023

# 通常版
python src/data_processing/data_encoding.py --start 2022 --end 2023
```

#### 3. モデル学習と評価

```bash
# 統合CLIを使用（推奨）
python main.py train

# または直接実行
python src/modeling/model_training.py
```

#### 4. バックテスト実行

```bash
# 🆕 最新の高度な戦略（推奨）
python main.py backtest

# オプション付き
python main.py backtest --min-ev 1.2 --no-trifecta
```

### 🚀 完全な実行フロー

```bash
# 1. データ収集（払戻データ付き）
python src/data_processing/enhanced_scraping.py --year 2024
# または中断可能版
python src/data_processing/data_scraping_with_checkpoint.py --start 2024 --end 2024

# 2. データ前処理（払戻データ対応）
python src/data_processing/data_encoding_v2.py --start 2022 --end 2023

# 3. モデル訓練
python main.py train

# 4. バックテスト（実際のオッズ使用）
python main.py backtest

# 5. 推定オッズのみでバックテスト（比較用）
python main.py backtest --no-actual-odds
```

### 5. Pythonコードとしての使用例

```python
# データスクレイピング
from src.data_processing.data_scraping import RaceScraper

scraper = RaceScraper(output_dir="data", max_workers=3)
scraper.scrape_years(2024, 2024)

# データエンコーディング
from src.data_processing.data_encoding import RaceDataEncoder

encoder = RaceDataEncoder()
encoded_path = encoder.encode_data(2022, 2023)

# モデル学習
from src.modeling.model_training import train_and_evaluate_model

model, results = train_and_evaluate_model(encoded_path)

# バックテスト
from src.backtesting.backtest import ImprovedBacktest

backtest = ImprovedBacktest(betting_fraction=0.005)
results = backtest.run_backtest()
```

## 🆕 新機能と戦略

### 🎯 高度な期待値ベッティング戦略

#### 期待値計算
```python
# 三連単の期待値計算例
的中確率 = P(1着) × P(2着|1着除外) × P(3着|1,2着除外)
推定オッズ = f(人気順位の合計)
期待値 = 的中確率 × 推定オッズ × (1 - JRA控除率)
```

#### ベッティングルール
1. **期待値フィルタ**: 1.1以上（設定可能）
2. **Kelly基準**: 最適ベット比率を計算
3. **リスク制限**: 最大5%/レース
4. **馬券種別調整**: 三連単30%、馬連50%、ワイド70%

### 実際のオッズデータ使用
最新版では、スクレイピングで取得した実際の払戻データを使用して期待値を計算できます：

- **三連単**: 実際の三連単オッズ
- **馬連**: 実際の馬連オッズ  
- **ワイド**: 実際のワイドオッズ
- **単勝・複勝**: 単勝・複勝の実際の配当

```bash
# 実際のオッズを使用（デフォルト）
python main.py backtest

# 推定オッズのみ使用（比較用）
python main.py backtest --no-actual-odds
```

### 高度な特徴量エンジニアリング

#### 基本特徴量（138以上）
- **枠番**: 馬番から自動計算
- **騎手・調教師の成績**: 勝率・複勝率
- **馬の過去成績**: 直近3,5走の平均着順
- **休養日数**: 前走からの間隔
- **距離適性**: 短距離・マイル・中距離・長距離フラグ
- **季節特徴量**: 春・夏・秋・冬の季節情報

#### 払戻データ関連特徴量
- **各種払戻額**: 実際の払戻金額（全馬券種）
- **高配当フラグ**: 異常値検出によるレース特性把握
- **配当パターン**: 人気と配当の乖離度

## データ仕様

### 入力データ（スクレイピング結果）

| カラム名 | 型 | 説明 |
|---------|-----|------|
| race_id | string | レース識別子 (例: 202301010101) |
| 馬 | string | 馬名 |
| 騎手 | string | 騎手名 |
| 馬番 | int | 馬番号 |
| 着順 | int/string | 着順（失格等の場合は文字列） |
| オッズ | float | 単勝オッズ |
| 人気 | int | 人気順位 |
| 体重 | int | 馬体重 |
| 体重変化 | int | 前走からの体重変化 |
| 性 | string | 性別（牡/牝/セ） |
| 齢 | int | 年齢 |
| 斤量 | float | 負担重量 |
| 芝・ダート | string | コース種別（芝/ダ） |
| 距離 | int | レース距離（メートル） |
| 馬場 | string | 馬場状態（良/稍重/重/不良） |
| 天気 | string | 天候 |

### 特徴量エンジニアリング

- **基本特徴量**: 上記の生データ
- **エンコード特徴量**: カテゴリカル変数のラベルエンコーディング
- **統計特徴量**: 騎手・馬・調教師の過去成績統計
- **時系列特徴量**: 日付から抽出した年、月、曜日等

## テスト

```bash
# ユニットテストの実行
python -m unittest discover -s src/tests -p "test_*.py" -v
```

## トラブルシューティング

### よくある問題

1. **スクレイピングエラー**
   - ネットワーク接続を確認
   - `config.yaml`のタイムアウト設定を調整
   - リトライ回数を増やす

2. **メモリ不足**
   - データを年単位で処理
   - バッチサイズを調整

3. **エンコーディングエラー**
   - ファイルエンコーディングがSHIFT-JISであることを確認

## パフォーマンス

### モデル性能
- **AUC-ROC**: 0.75+
- **精度**: 最適閾値で70%以上
- **年間回収率**: 110%以上（バックテスト結果）

### 主要な改善点
1. **特徴量エンジニアリング**: 138の高度な特徴量
2. **時系列検証**: TimeSeriesSplitによる適切な評価
3. **ハイパーパラメータ最適化**: Optunaによる自動チューニング
4. **ベッティング戦略**: Kelly基準とリスク管理
5. **モデル解釈性**: SHAP値による透明性確保

## 今後の改善予定

- [ ] リアルタイム予測API
- [ ] 深層学習モデル（LSTM/Transformer）の追加
- [ ] 複数モデルのアンサンブル学習
- [ ] Webダッシュボード開発
- [ ] 自動再学習パイプライン
- [ ] より詳細な馬体情報の統合

## 開発者向け情報

### コーディング規約
- PEP 8準拠
- 型ヒントの使用推奨
- docstringの記載必須

### テスト実行
```bash
# ユニットテスト
python -m pytest src/tests/

# カバレッジレポート
python -m pytest --cov=src src/tests/
```

### コード品質チェック
```bash
# フォーマット
black src/

# リンティング
flake8 src/

# 型チェック
mypy src/
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルを参照してください。

## 注意事項

- **スクレイピング**: netkeiba.comの利用規約を遵守し、適切な間隔でリクエストを送信してください
- **予測の利用**: 予測結果は参考情報です。実際の賭けは自己責任で行ってください
- **データの取り扱い**: スクレイピングしたデータの再配布は禁止されています

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを作成して変更内容について議論してください。

## 作者

- GitHub: [@yourusername](https://github.com/yourusername)

## 謝辞

- netkeiba.comのデータ提供
- LightGBM開発チーム
- Optunaコミュニティ