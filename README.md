# 競馬予測AIシステム

## 概要

このプロジェクトは、機械学習を用いて競馬レースの結果を予測するシステムです。過去のレースデータをスクレイピングし、様々な特徴量を抽出して、LightGBMモデルで勝馬を予測します。

## 主な機能

- **データスクレイピング**: netkeiba.comから過去のレースデータを自動収集
- **データ前処理**: カテゴリカル変数のエンコーディング、特徴量エンジニアリング
- **予測モデル**: LightGBMを使用した二値分類（勝馬予測）
- **評価システム**: 回収率を考慮した閾値最適化

## プロジェクト構造

```
Keiba_AI/
├── config.yaml              # 設定ファイル
├── data/                    # レースデータ（CSV/Excel）
├── models/                  # 学習済みモデル
├── output/                  # 出力ファイル
├── logs/                    # ログファイル
├── src/                     # ソースコード
│   ├── utils/              # ユーティリティ関数
│   │   ├── scraping_utils.py    # スクレイピング用関数
│   │   ├── data_utils.py        # データ処理用関数
│   │   └── config_loader.py     # 設定読み込み
│   ├── tests/              # テストコード
│   └── data_quality_check.py    # データ品質チェック
├── docs/                    # ドキュメント
└── ノートブック/
    ├── 00.data_scraping.ipynb   # データ収集
    ├── 01.encode.ipynb          # データエンコーディング
    └── 02.model.ipynb           # モデル構築・評価
```

## セットアップ

### 1. 依存関係のインストール

```bash
# Pipfileを使用する場合
pipenv install

# または requirements.txt を使用する場合
pip install -r requirements.txt
```

### 2. 設定ファイルの確認

`config.yaml`で以下の設定を確認・調整してください：

- データ取得年範囲
- スクレイピング設定（並行処理数、タイムアウト等）
- モデルパラメータ
- 出力ディレクトリ

## 使用方法

### 1. データスクレイピング

```python
# Jupyter Notebookで実行
# 00.data_scraping.ipynb を開いて実行
```

または、モジュール化されたコードを使用：

```python
from src.utils.scraping_utils import RaceScraper
from src.utils.config_loader import get_config

config = get_config()
scraper = RaceScraper()

# レースデータの取得
race_data = scraper.fetch_race_data(url)
```

### 2. データ前処理

```python
# 01.encode.ipynb を開いて実行
```

### 3. モデル学習・評価

```python
# 02.model.ipynb を開いて実行
```

### 4. データ品質チェック

```bash
# 全データファイルをチェック
python src/data_quality_check.py

# 特定のファイルをチェック
python src/data_quality_check.py --file data/2023.csv

# レポートを指定場所に出力
python src/data_quality_check.py --output reports/quality_check.json
```

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

## 今後の改善予定

- [ ] リアルタイム予測機能
- [ ] より高度な特徴量エンジニアリング
- [ ] 複数の予測モデルのアンサンブル
- [ ] Webインターフェースの追加
- [ ] 自動再学習システム

## ライセンス

このプロジェクトは個人利用を目的としています。商用利用の際はご相談ください。

## 注意事項

- スクレイピングは対象サイトの利用規約を遵守してください
- 過度なリクエストは避け、適切な間隔を設けてください
- 予測結果は参考情報であり、実際の賭けは自己責任で行ってください