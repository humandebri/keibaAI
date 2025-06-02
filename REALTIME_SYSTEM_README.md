# 🏇 JRA競馬リアルタイムシステム 実装ガイド

## 📋 概要

このドキュメントは、競馬予測モデルとリアルタイムデータ取得・自動投票機能の統合システムの使用方法を説明します。

## 🚀 クイックスタート

### 1. 環境確認

```bash
# 必要なパッケージのインストール
pip install requests beautifulsoup4 pandas numpy asyncio aiohttp

# ディレクトリ構造の確認
mkdir -p logs cache/jra_data test_results demo_logs
```

### 2. システムテスト

```bash
# リアルタイムデータ取得のテスト
python test_realtime_system.py

# 5分間のクイックデモ
python demo_runner.py
```

## 📁 ファイル構成

### コアシステム
- **jra_realtime_system.py**: JRA/netkeibaからのデータ取得
- **integrated_betting_system.py**: 予測モデルとの統合
- **keiba_ai_improved_system_fixed.py**: 改善された予測モデル

### テスト・デモ
- **test_realtime_system.py**: システムテストスクリプト
- **demo_runner.py**: 安全なデモ実行環境

### ドキュメント
- **JRA_IMPLEMENTATION_PLAN.md**: 5週間の実装計画
- **REALTIME_SYSTEM_README.md**: このファイル

## 🔧 主要機能

### 1. リアルタイムデータ取得

```python
from jra_realtime_system import JRARealTimeSystem

# システム初期化
jra = JRARealTimeSystem()

# 本日のレース取得
races = jra.get_today_races()

# レース詳細取得
details = jra.get_race_details(race_id)
```

### 2. 統合システム

```python
from integrated_betting_system import IntegratedKeibaSystem

# 設定
config = {
    'max_bet_per_race': 10000,
    'min_expected_value': 1.2,
    'kelly_fraction': 0.05,
    'enable_auto_betting': False  # 安全のため
}

# システム起動
system = IntegratedKeibaSystem(config)
await system.start()
```

## ⚙️ 設定オプション

### 基本設定
```python
{
    'max_bet_per_race': 10000,      # レースあたり最大ベット額
    'max_daily_loss': 50000,        # 日次最大損失額
    'min_expected_value': 1.1,      # 最小期待値（1.1 = 10%）
    'kelly_fraction': 0.05,         # Kelly基準の係数（5%）
    'data_refresh_interval': 300,   # データ更新間隔（秒）
    'enable_auto_betting': False    # 自動投票（推奨: False）
}
```

### レート制限設定
```python
{
    'min_delay': 1.0,   # 最小遅延（秒）
    'max_delay': 3.0,   # 最大遅延（秒）
    'timeout': 10,      # タイムアウト（秒）
    'max_retries': 3    # 最大リトライ回数
}
```

## 🏃 実行モード

### 1. テストモード
```bash
# 全機能のテスト
python test_realtime_system.py
```

### 2. ペーパートレード
```bash
# 実際の資金を使わない仮想取引
python demo_runner.py --mode paper --hours 2
```

### 3. ライブモード（手動確認）
```bash
# 実資金を使用（手動確認必須）
python demo_runner.py --mode live --hours 1
```

## 📊 データフロー

```
1. データ収集
   ├── JRA公式サイト → 基本情報
   └── netkeiba.com → 詳細情報

2. データ統合
   └── 重複排除・フォーマット統一

3. 予測実行
   ├── 特徴量生成
   └── モデル予測

4. ベッティング判断
   ├── 期待値計算
   ├── Kelly基準
   └── リスク管理

5. 投票処理
   ├── ログ記録（ペーパー）
   └── IPAT連携（ライブ）※手動確認
```

## ⚠️ 重要な注意事項

### 法的遵守事項
1. **robots.txt遵守**: 各サイトのクローリングルールを守る
2. **レート制限**: 過度なアクセスは厳禁
3. **利用規約**: 各サイトの利用規約を確認
4. **手動確認**: 自動投票は必ず手動で最終確認

### セキュリティ
1. **認証情報**: 環境変数で管理
   ```bash
   export JRA_MEMBER_ID="your_id"
   export JRA_PIN="your_pin"
   export JRA_PARS="your_pars"
   ```

2. **ログ管理**: 個人情報を含まないよう注意

### リスク管理
1. **小額スタート**: 最初は最小額でテスト
2. **損失制限**: 日次・月次の上限設定
3. **記録保持**: 全ての取引を記録
4. **定期見直し**: 週次でパフォーマンス評価

## 🔍 トラブルシューティング

### よくある問題

1. **レースが取得できない**
   - レース開催日か確認
   - ネットワーク接続を確認
   - サイト構造の変更をチェック

2. **予測モデルエラー**
   - モデルファイルの存在確認
   - 必要な特徴量の確認
   - メモリ不足でないか確認

3. **レート制限エラー**
   - 遅延時間を増やす
   - リトライ間隔を調整
   - キャッシュを活用

## 📈 パフォーマンス最適化

1. **キャッシュ活用**
   ```python
   config['enable_cache'] = True
   config['cache_ttl'] = 300  # 5分
   ```

2. **並列処理**
   ```python
   # 非同期処理でデータ取得を高速化
   async with aiohttp.ClientSession() as session:
       tasks = [fetch_race(session, race_id) for race_id in race_ids]
       results = await asyncio.gather(*tasks)
   ```

3. **メモリ管理**
   - 不要なデータは即座に削除
   - 大きなDataFrameは分割処理
   - ガベージコレクションの活用

## 🚦 次のステップ

1. **Week 1**: 基本機能のテスト
2. **Week 2**: データ収集の安定化
3. **Week 3**: 予測精度の向上
4. **Week 4**: リスク管理の強化
5. **Week 5**: 本番環境への移行

## 📞 サポート

問題が発生した場合：
1. ログファイルを確認（`logs/`ディレクトリ）
2. テストスクリプトを実行
3. エラーメッセージを記録
4. 設定を見直す

---

**免責事項**: このシステムは研究・教育目的で作成されています。実際の投票は自己責任で行ってください。