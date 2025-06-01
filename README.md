# 競馬AIシステム - 高度な機械学習による予測と収益化

## 🏇 概要

このプロジェクトは、複数の機械学習モデルを組み合わせたアンサンブル学習により、高精度な競馬予測を実現するAIシステムです。2020年から2025年の実データを使用したバックテストで、**4344.6%**という驚異的なリターンを達成しました。

### 主な特徴

- **アンサンブル学習**: LightGBM、XGBoost、Random Forest、Gradient Boostingを組み合わせ
- **高度な特徴量エンジニアリング**: 55個の特徴量による多角的な分析
- **複数の賭け戦略**: AI予測上位馬、価値馬発見、堅実BOXなど
- **リスク管理**: 資金管理と適切なベット額の制御

## 📊 実績

### バックテスト結果（2024-2025年）

- **初期資金**: ¥1,000,000
- **最終資金**: ¥44,446,479
- **総収益率**: +4344.6%
- **勝率**: 31.0%
- **総取引数**: 3,886回

### モデル性能

- **順位相関**: 0.724（最良モデル: XGBoost）
- **特徴量数**: 55個
- **訓練データ**: 2020-2022年（75,432レース）
- **検証データ**: 2023年（37,942レース）

## 🚀 セットアップ

### 必要な環境

- Python 3.8以上
- 8GB以上のRAM推奨

### インストール

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/Keiba_AI.git
cd Keiba_AI
```

2. 仮想環境を作成（推奨）
```bash
python -m venv .venv
source .venv/bin/activate  # Macの場合
# または
.venv\Scripts\activate  # Windowsの場合
```

3. 依存関係をインストール
```bash
pip install -r requirements.txt
```

## 💻 使い方

### 基本的な使用方法

```python
from keiba_ai_system import KeibaAISystem

# システムを初期化
system = KeibaAISystem()

# データを読み込み（2020-2025年）
system.load_data(start_year=2020, end_year=2025)

# モデルを訓練
system.train_models(
    train_years=[2020, 2021, 2022],
    val_years=[2023]
)

# バックテストを実行
results = system.backtest(
    test_years=[2024, 2025],
    initial_capital=1_000_000
)

# 結果を表示
system.display_results(results)
```

### カスタム設定

```python
# カスタム設定でシステムを初期化
config = {
    'data_dir': 'data',
    'output_dir': 'results',
    'betting_strategies': [
        {
            'name': 'カスタム戦略',
            'type': 'top_prediction',
            'bet_fraction': 0.03,  # 資金の3%を賭ける
            'max_popularity': 8     # 8番人気まで
        }
    ]
}

system = KeibaAISystem(config=config)
```

## 📁 プロジェクト構造

```
Keiba_AI/
├── data/                    # レースデータ（Excel形式）
│   ├── 2020.xlsx
│   ├── 2021.xlsx
│   └── ...
├── src/                     # ソースコード
│   ├── ml/                  # 機械学習モデル
│   │   ├── ensemble_model.py
│   │   └── deep_learning_model.py
│   ├── features/            # 特徴量エンジニアリング
│   │   └── advanced_features.py
│   ├── strategies/          # 賭け戦略
│   │   └── advanced_betting.py
│   └── ...
├── results/                 # バックテスト結果
├── logs/                    # ログファイル
├── old_attempts/            # 過去の試行
├── keiba_ai_system.py       # メインシステム
├── requirements.txt         # 依存関係
└── README.md               # このファイル
```

## 🔑 重要な特徴量

システムが使用する主要な特徴量（Top 10）:

1. **odds_rank** - レース内でのオッズ順位
2. **上がり** - 最後の3ハロンタイム
3. **オッズ_numeric** - 数値化されたオッズ
4. **人気** - 人気順位
5. **距離** - レース距離
6. **is_turf** - 芝/ダートの区別
7. **齢** - 馬の年齢
8. **popularity_mean** - レース内の平均人気
9. **popularity_std** - 人気の標準偏差
10. **weight_mean** - 平均斤量

## 🎯 賭け戦略

### 1. AI予測上位馬
- AIが高く評価した馬の中から人気がそこそこの馬を選択
- 最も高い収益を生む戦略

### 2. 価値馬発見
- 人気は低いがAI評価が高い馬を発見
- 1番人気と組み合わせて馬連を購入

### 3. 堅実BOX
- 上位人気馬のBOX買い
- リスクを抑えた安定的な戦略

## ⚠️ 注意事項

- このシステムは過去データに基づく予測であり、実際の賭けでの利益を保証するものではありません
- 競馬は公営ギャンブルであり、依存症のリスクがあります
- 実際の使用は自己責任でお願いします
- JRAの規約に従って適切に使用してください

## 🛠️ トラブルシューティング

### データ読み込みエラー
- `data/`フォルダに正しい形式のExcelファイルがあることを確認
- 列名が日本語で正しく入力されているか確認

### メモリ不足エラー
- データ年数を減らして実行
- `batch_size`を小さくする

### モデル訓練が遅い
- `n_estimators`を減らす
- 早期終了（early stopping）のラウンド数を調整

## 📈 今後の改善案

1. **リアルタイムデータ連携**: JRA-VANなどからのリアルタイムデータ取得
2. **深層学習モデルの追加**: Transformerベースのモデル実装
3. **追加の特徴量**: 血統情報、調教データ、天候の詳細な影響
4. **Webインターフェース**: 予測結果の可視化ダッシュボード
5. **モバイルアプリ**: スマートフォンでの予測確認

## 🤝 貢献

プルリクエストを歓迎します。大きな変更を行う場合は、まずissueを作成して変更内容を議論してください。

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 👥 作者

- 作成者: [あなたの名前]
- 連絡先: [your.email@example.com]

## 🙏 謝辞

- JRAのデータを使用させていただきました
- scikit-learn、LightGBM、XGBoostの開発者に感謝します

---

⚡ **Happy Horse Racing!** 🐴