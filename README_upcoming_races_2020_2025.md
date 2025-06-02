# 競馬予測システム アップデート（2025年6月）

## 実装した変更内容

### 1. 今後のレース検索機能の追加

#### jra_realtime_system.py の変更
- `get_upcoming_races()` メソッドを追加
  - `days_ahead`: 何日先から取得するか（0=今日、1=明日）
  - `max_days`: 最大何日先まで取得するか
- `get_today_races()` は内部的に `get_upcoming_races(days_ahead=0, max_days=1)` を呼ぶように変更

#### NetkeibaDataCollector クラスの変更
- `get_upcoming_race_list()` メソッドを追加
- 同様に日付指定でレース情報を取得可能

### 2. 統合ベッティングシステムの更新

#### integrated_betting_system.py の変更
- `_get_upcoming_races()` メソッドを追加
- 設定項目を追加：
  - `days_ahead`: 明日から検索開始（デフォルト: 1）
  - `max_days_to_analyze`: 最大3日先まで分析（デフォルト: 3）
- 日付ごとにグループ化してレース処理

### 3. モデル訓練期間の変更（2020-2025年）

#### 新規作成ファイル
1. **encode_2020_2025_data.py**
   - 2020-2025年のデータをエンコード
   - `python encode_2020_2025_data.py` で実行

2. **train_model_2020_2025.py**
   - 2020-2025年データでモデルを訓練
   - 2020-2023年を訓練データ、2024-2025年を検証データとして使用
   - コロナ期間の特徴量を追加
   - `python train_model_2020_2025.py` で実行

3. **test_upcoming_races.py**
   - 今後のレース取得機能のテスト
   - `python test_upcoming_races.py` で実行

## 使用方法

### 1. データの準備（既にdata/フォルダに2020-2025年のXLSXファイルがあることを確認）

### 2. データのエンコード
```bash
python encode_2020_2025_data.py
```

### 3. モデルの訓練
```bash
python train_model_2020_2025.py
```

### 4. 今後のレース分析の実行
```bash
python integrated_betting_system.py
```

## 主な特徴

1. **未来のレース予測**
   - 明日、明後日、週末のレースを事前に分析
   - 計画的な投資戦略の立案が可能

2. **最新データでの訓練**
   - 2020-2025年の最新データを使用
   - コロナ期間の特殊な状況も考慮

3. **柔軟な設定**
   - 何日先まで分析するか設定可能
   - 分析対象期間を自由に調整

## 設定のカスタマイズ

integrated_betting_system.py のmain関数内で設定を変更できます：

```python
config = {
    'enable_auto_betting': False,  # 自動投票の有効化（注意が必要）
    'min_expected_value': 1.2,    # 最小期待値
    'kelly_fraction': 0.025,       # Kelly基準の係数
    'max_bet_per_race': 5000,     # レースあたり最大投票額
    'max_daily_loss': 30000,      # 日次最大損失額
    'days_ahead': 1,              # 何日先から検索（0=今日、1=明日）
    'max_days_to_analyze': 3      # 最大何日先まで分析
}
```

## 注意事項

1. **レート制限**
   - JRAやnetkeibaのサーバーに負荷をかけないよう、適切な遅延を設定
   - 大量のリクエストは避ける

2. **データの鮮度**
   - オッズは時間とともに変動するため、レース直前の再取得を推奨
   - キャッシュ機能を活用して効率化

3. **投票の確認**
   - 自動投票機能は必ず手動確認が必要
   - 実際の投票は慎重に行うこと