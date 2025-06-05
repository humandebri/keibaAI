# 🏇 競馬AI予測システム - Advanced ML-based Horse Racing Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)](https://catboost.ai/)
[![Data](https://img.shields.io/badge/Data-2020--2025-green.svg)](data/)

> **高精度競馬予測システム** - 6年分の競馬データと最新の機械学習技術を活用した実践的な予測システム

## 🎯 システム概要

本システムは、JRAの競馬データを機械学習で分析し、各馬の勝率を予測する高度な予測システムです。CatBoostを中心とした機械学習モデルと、拡張された特徴量エンジニアリングにより、高精度な予測を実現しています。

### 🚀 **最新の改善点 (2024年12月)**
- **6年分のデータ活用** - 2020-2025年の全レースデータ（216,858件）
- **拡張騎手統計** - 16種類の騎手関連特徴量による精度向上
- **中間日数特徴量** - 休養期間と調子の関係を学習
- **確率正規化** - 勝率の合計が正確に100%になるよう改善
- **リアルタイム予測** - 実レースに対応した予測システム

## 📊 システム性能

### 🤖 **機械学習モデル性能**

| 指標 | 値 | 説明 |
|------|-----|------|
| **AUC Score** | 0.858 | 高い判別性能 |
| **データ規模** | 216,858件 | 6年分の全レースデータ |
| **特徴量数** | 50+ | 基本・騎手・中間日数特徴量 |
| **訓練時間** | 約10分 | M1 MacBook Proでの実測 |

### 📈 **予測精度の特徴**

- **確率正規化**: レース単位で勝率の合計が100%
- **現実的な勝率**: 2.1%〜8.2%の適切な範囲
- **期待値計算**: オッズと勝率から自動計算
- **高速予測**: リアルタイムでの予測が可能

### 🎯 **特徴量カテゴリ**

#### 基本特徴量
- 体重、体重変化、斤量、上がり
- 出走頭数、距離、クラス、性別
- 芝・ダート、回り、馬場、天気
- 場id、枠番

#### 騎手統計（拡張版）
- 基本統計: 勝率、複勝率、騎乗数、平均着順、ROI
- 時系列統計: 30日/60日勝率、連続不勝、最後勝利日数
- コンテキスト統計: 芝/ダート別、距離カテゴリ別
- シナジー統計: 騎手×調教師の相性

#### 中間日数関連
- 前走からの日数、放牧区分（休養カテゴリ）
- 平均中間日数、中間日数標準偏差
- 最近3走の具体的な間隔

#### 過去成績
- 過去5走の着順、距離、通過順、走破時間
- 平均着順、最高着順、勝利/複勝経験

## 🛠️ インストール & セットアップ

### 📋 **必要要件**
- Python 3.8+
- 8GB RAM (推奨: 16GB)
- 10GB ディスク容量

### ⚡ **クイックスタート**

```bash
# 1. リポジトリクローン
git clone https://github.com/yourusername/Keiba_AI.git
cd Keiba_AI

# 2. 自動セットアップ（推奨）
./setup_env.sh

# 3. MLシステム実行
./run_venv.sh python clean_full_data_ml_system.py
```

### 🔧 **手動セットアップ**

```bash
# 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt

# データ準備（必要な場合）
python encode_2020_2025_data.py  # データエンコーディング
```

## 🎮 使用方法

### 🚀 **基本的な使い方**

```bash
# MLシステムの実行（訓練と予測）
python clean_full_data_ml_system.py

# 統一CLIでのバックテスト
python main.py backtest --train-years 2020 2021 2022 --test-years 2024 2025

# データ収集（スクレイピング）
python main.py collect --start-year 2024 --end-year 2024

# データエンコーディング
python encode_2020_2025_data.py

# リアルタイム予測システム
python jra_realtime_system.py
```

### 🎯 **高精度モデルの訓練（train_model_2020_2025.py）**

詳細な特徴量（騎手統計、中間日数等）を使用した改良版モデルの訓練：

```bash
# 1. データエンコーディング（必須）
python src/data_processing/data_encoding_v2.py --start 2020 --end 2025

# 2. 改良版モデルの訓練
source .venv/bin/activate
python train_model_2020_2025.py
```

#### 処理フロー:
1. **エンコード済みデータ読み込み**: `encoded/2020_2025encoded_data_v2.csv`
2. **馬・騎手データベース構築**: 6年分の生データから統計を計算（キャッシュ対応）
3. **詳細特徴量の追加**:
   - 騎手統計: 基本、時系列、コンテキスト、シナジー（16種類）
   - 中間日数: 前走からの日数、放牧区分、平均・標準偏差
   - 馬の過去成績: 過去10走の詳細データ
4. **モデル訓練**: 2020-2023年で訓練、2024-2025年で検証
5. **結果保存**: `model_2020_2025/`ディレクトリに保存

#### 出力ファイル:
- `model_2020_2025/model_2020_2025.pkl` - 訓練済みLightGBMモデル
- `model_2020_2025/feature_cols_2020_2025.pkl` - 特徴量リスト
- `model_2020_2025/model_info_2020_2025.json` - モデル情報
- `cache/horse_database.pkl` - 馬・騎手データベースのキャッシュ

### 📊 **プログラマティック利用**

```python
# clean_full_data_ml_system.py の使用例
from clean_full_data_ml_system import ImprovedMLSystem

# システム初期化
system = ImprovedMLSystem()

# モデル訓練（6年分のデータを使用）
system.train()

# ライブレース予測
system.predict_race("live_race_data_202505021212.csv")
```

### 🏇 **予測結果の例**

```
🎯 予測結果:
================================================================================
 順位  馬番             馬名     オッズ      勝率     期待値
================================================================================
  1.  3番 エリキング            17.0倍   8.2%   1.40
  2. 18番 サトノシャイニング        12.3倍   7.5%   0.92
  3. 13番 クロワデュノール          2.1倍   7.4%   0.16

📊 予測統計:
   勝率合計: 100.0%
   期待値1.0以上: 13頭
```

### ⚙️ **MLConfigパラメータ**

```python
@dataclass
class MLConfig:
    """機械学習設定"""
    random_state: int = 42
    test_size: float = 0.15
    n_folds: int = 5
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0
    scale_pos_weight: float = 17  # クラスバランス調整
```

## 🏗️ システムアーキテクチャ

### 📁 **プロジェクト構造**

```
Keiba_AI/
├── 🎯 clean_full_data_ml_system.py     # メインMLシステム
├── 📊 main.py                          # 統一CLIエントリーポイント
├── 📚 data_with_payout/                # 払戻データ付きレースデータ
│   ├── 2020_with_payout.xlsx
│   ├── 2021_with_payout.xlsx
│   └── ...
├── 🔢 encoded/                         # エンコード済みデータ
│   └── 2020_2025encoded_data_v2.csv
├── 🤖 model_2020_2025/                 # 訓練済みモデル
│   ├── model_2020_2025.pkl
│   └── feature_cols_2020_2025.pkl
├── 💾 cache/                           # キャッシュデータ
│   └── horse_database.pkl              # 馬・騎手統計DB
├── 📊 src/                             # ソースコード
│   ├── core/                           # コアシステム
│   ├── features/                       # 特徴量エンジニアリング
│   ├── strategies/                     # ベッティング戦略
│   └── data_processing/                # データ処理
├── 🧪 tests/                           # テストスイート
├── 📋 config/                          # 設定ファイル
├── 📈 results/                         # 結果データ
└── 🚀 *.sh                            # 実行スクリプト
```

### 🔄 **データフロー**

```
1. データ収集（data_with_payout/）
   ↓
2. エンコーディング（encode_2020_2025_data.py）
   ↓
3. データベース構築（HorseDatabase）
   ↓
4. 特徴量生成（50+特徴量）
   ↓
5. モデル訓練（CatBoost）
   ↓
6. 予測生成（確率正規化）
   ↓
7. 期待値計算（勝率×オッズ）
```

## 🔬 技術詳細

### 🤖 **機械学習モデル**

```python
# CatBoostClassifier with balanced weights
self.model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    scale_pos_weight=17  # 正例と負例の比率調整
)
```

### 📊 **データベースシステム**

- **HorseDatabase**: 31,869頭の詳細成績を管理
- **騎手統計**: 369人の騎手の詳細統計
- **キャッシュ機能**: 高速データアクセス
- **拡張統計**: 時系列・コンテキスト別・シナジー分析

### 🎯 **予測の特徴**

- **確率正規化**: Softmax正規化でレース内合計100%
- **グループ学習**: GroupKFoldで時系列リーク防止
- **特徴量重要度**: CatBoostの内蔵機能で解釈可能

## 🧪 主要ファイルの説明

### 📄 **clean_full_data_ml_system.py**
メインのMLシステム。以下の機能を統合：
- HorseDatabase: 馬・騎手の過去成績データベース
- ImprovedMLSystem: 機械学習の訓練と予測（CatBoost）
- 拡張特徴量エンジニアリング
- リアルタイム予測機能（勝率予測）

### 📄 **train_model_2020_2025.py**
改良版モデル訓練スクリプト：
- HorseDatabase: 6年分のデータから詳細統計を構築
- 騎手統計: 基本、時系列、コンテキスト、シナジー統計を追加
- 中間日数特徴量: 休養期間と調子の関係を学習
- LightGBMモデル: 複勝（3着以内）予測に特化

### 📄 **encode_2020_2025_data.py** / **src/data_processing/data_encoding_v2.py**
2020-2025年のレースデータをエンコード：
- Excel形式のデータをCSVに変換
- カテゴリ変数の数値化
- 過去5走の成績を特徴量として追加
- 払戻データ対応版（v2）

### 📄 **main.py**
統一CLIインターフェース：
- バックテスト実行
- データ収集・エンコーディング
- モデル訓練
- 結果表示

### 📄 **jra_realtime_system.py**
リアルタイム予測システム：
- JRA公式サイトからレース情報取得
- 出馬表の自動解析
- リアルタイム予測実行

## 🧪 テスト実行

```bash
# 全テスト実行
./run_tests.sh

# 個別テスト
python -m pytest tests/test_config.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_unified_system.py -v
```

## 🌟 今後の改善計画

### 📈 **短期的改善**
- [ ] 調教データの追加
- [ ] 馬場指数の統合
- [ ] 血統情報の活用
- [ ] パドック評価の数値化

### 🚀 **中長期的改善**
- [ ] ニューラルネットワークモデルの追加
- [ ] リアルタイムオッズ変動の活用
- [ ] 地方競馬への対応
- [ ] WebUI/モバイルアプリの開発

## 🚨 注意事項 & 免責事項

> ⚠️ **重要**: このシステムは教育・研究目的で開発されています

### 📋 **利用にあたって**

- **投資リスク**: 過去の成績は将来の結果を保証しません
- **自己責任**: 実際の投資判断は自己責任で行ってください
- **法令遵守**: 公営競技の規約に従って利用してください
- **資金管理**: 適切な資金管理を心がけてください

### 🛡️ **推奨事項**

1. **少額から開始**: 全資産の1-2%以下で始める
2. **定期的見直し**: 月次でパフォーマンスを評価
3. **システマティック**: 感情的判断を避け、システムに従う
4. **継続的学習**: モデルの定期的な再訓練

## 🤝 貢献について

### 🔧 **開発環境**

```bash
# 開発用セットアップ
pip install -r requirements.txt
pip install black flake8 mypy pytest

# コード品質チェック
black src/
flake8 src/
mypy src/
```

### 📝 **貢献方法**

1. **Issue作成**: バグ報告・機能要望
2. **Fork**: リポジトリをフォーク
3. **Branch**: feature/your-feature-name
4. **Commit**: 明確なコミットメッセージ
5. **Pull Request**: 詳細な説明と共に提出

## 📜 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- **JRA**: 競馬データの提供
- **netkeiba.com**: 包括的な競馬データベース
- **CatBoost**: 高性能な勾配ブースティングライブラリ
- **scikit-learn**: 機械学習フレームワーク
- **pandas/numpy**: データ処理ライブラリ

---

## 📚 関連ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [CLAUDE.md](CLAUDE.md) | AI開発アシスタント向けガイド |
| [QUICKSTART.md](QUICKSTART.md) | クイックスタートガイド |
| [data_schema.md](docs/data_schema.md) | データスキーマ仕様 |

## 🔗 Quick Links

- [📊 サンプルデータ](data_with_payout/)
- [🧪 テストコード](tests/)
- [📓 Jupyterノートブック](notebooks/)
- [⚙️ 設定ファイル](config/)

---

<div align="center">

### 🏇 **競馬AI予測システム** 🏇
**機械学習による高精度競馬予測プラットフォーム**

Built with ❤️ by the Keiba AI Team

**⚡ May the odds be ever in your favor! 🐴**

</div>