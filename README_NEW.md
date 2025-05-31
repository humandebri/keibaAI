# Keiba AI - 競馬予測AIシステム（リファクタリング版）

## 概要

Keiba AIは、機械学習を用いて競馬レースの結果を予測し、期待値に基づいた馬券戦略を実行する高度なAIシステムです。

## 主な特徴

- **高度な期待値計算**: モデル予測とオッズから正確な期待値を算出
- **流し馬券戦略**: 三連単・馬連・ワイドの自動選択
- **リスク管理**: Kelly基準によるベット額最適化
- **現実的なシミュレーション**: JRA控除率20%を考慮

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/Keiba_AI.git
cd Keiba_AI

# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

## 使用方法

### 統合CLI

```bash
# 高度な期待値ベースバックテスト（デフォルト）
python main.py backtest

# オプション指定
python main.py backtest --min-ev 1.2      # 期待値1.2以上のみ
python main.py backtest --no-trifecta     # 三連単を除外
python main.py backtest --no-quinella     # 馬連を除外
python main.py backtest --no-wide         # ワイドを除外

# データ収集（拡張版スクレイピング）
python main.py collect --start-year 2024 --end-year 2024

# データエンコーディング
python main.py encode --start-year 2022 --end-year 2023

# モデル訓練
python main.py train
```

## 戦略の詳細

### 期待値計算

```python
# 三連単の期待値計算例
的中確率 = P(1着) × P(2着|1着除外) × P(3着|1,2着除外)
推定オッズ = f(人気順位の合計)
期待値 = 的中確率 × 推定オッズ × (1 - JRA控除率)
```

### ベッティングルール

1. **期待値フィルタ**: 1.1以上（設定可能）
2. **Kelly基準**: 最適ベット比率を計算
3. **リスク制限**: 最大5%/レース
4. **馬券種別調整**: 三連単30%、馬連50%、ワイド70%

## プロジェクト構造

```
Keiba_AI/
├── main.py                          # 統合エントリーポイント
├── src/
│   ├── core/
│   │   ├── config.py               # 統一設定管理
│   │   └── utils.py                # 共通ユーティリティ
│   ├── strategies/
│   │   ├── base.py                 # 基底戦略クラス
│   │   └── advanced_betting.py     # 高度な馬券戦略
│   ├── data_processing/
│   │   ├── enhanced_scraping.py    # 拡張スクレイピング（払戻データ含む）
│   │   └── data_encoding.py        # データエンコーディング
│   ├── modeling/
│   │   └── model_training.py       # モデル訓練
│   └── backtesting/
│       ├── advanced_backtest.py    # アンサンブルモデル戦略
│       └── advanced_betting_types.py # 多様な馬券種別
├── data/                           # レースデータ
├── models/                         # 学習済みモデル
└── results/                        # 実行結果
```

## Pythonコードとしての使用

```python
from src.strategies.advanced_betting import AdvancedBettingStrategy

# 戦略の初期化
strategy = AdvancedBettingStrategy(
    min_expected_value=1.2,  # 期待値1.2以上
    enable_trifecta=True,    # 三連単有効
    enable_quinella=True,    # 馬連有効
    enable_wide=True         # ワイド有効
)

# データ読み込みと準備
strategy.load_data(start_year=2019, end_year=2023)
strategy.split_data(
    train_years=[2019, 2020],
    test_years=[2021, 2022, 2023]
)

# バックテスト実行
results = strategy.run_backtest(initial_capital=1_000_000)

# 結果表示
strategy.print_results()
```

## 技術仕様

### モデル
- **アルゴリズム**: LightGBM（勾配ブースティング）
- **予測対象**: 着順（回帰問題）
- **特徴量**: オッズ、人気、馬番、斤量、性別、馬場状態など

### 期待値計算
- **確率推定**: モデル予測から各着順の確率を算出
- **オッズ推定**: 人気順位から現実的な配当を推定
- **控除率考慮**: JRA控除率20%を適用

### リスク管理
- **Kelly基準**: 最適ベット比率の計算
- **上限設定**: 資金の最大5%/レース
- **馬券種別調整**: リスクに応じた係数適用

## パフォーマンス最適化

- **並列処理**: データ収集で複数ワーカー使用
- **メモリ効率**: 年単位でのデータ処理
- **キャッシュ**: 学習済みモデルの保存

## 注意事項

1. **ギャンブルの本質**: 長期的にはJRA控除率により負ける設計
2. **予測の限界**: 100%の的中は不可能
3. **責任ある利用**: 余剰資金での娯楽として
4. **データ利用**: netkeiba.comの利用規約を遵守

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```bash
   # データを年単位で処理
   python main.py encode --start-year 2023 --end-year 2023
   ```

2. **スクレイピングエラー**
   - タイムアウト設定を調整
   - リトライ間隔を増やす

3. **予測精度が低い**
   - より多くの訓練データを使用
   - 特徴量エンジニアリングの改善

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueで議論してください。

## ライセンス

MIT License

## 免責事項

- 本システムは研究・教育目的です
- 実際の賭けは自己責任で行ってください
- 予測結果は参考情報に過ぎません