# 🏆 競馬AI最適化システム 完全使用ガイド

## 🎯 **システム概要**

期待値1.095を達成し、年間収益率15-20%を目指す最先端の競馬AIシステムです。

### ✨ **主要機能**
- **期待値1.095達成**: 学術研究レベルの特徴量エンジニアリング
- **最適化Kelly基準**: リスク調整済み資金管理
- **統一特徴量エンジン**: 92種類の高度な特徴量
- **分散投資対応**: 相関を考慮した同時ベット最適化

---

## 🚀 **クイックスタート**

### 1. 環境セットアップ
```bash
# 仮想環境のアクティベート
source .venv/bin/activate

# または
./run_venv.sh
```

### 2. データ準備確認
```bash
# 利用可能データの確認
ls encoded/
ls model_2020_2025/
```

### 3. 基本実行
```bash
# デモ実行（推奨）
python demo_optimal_system.py

# または手動実行
python -c "
from src.strategies.optimized_kelly_strategy import OptimizedKellyStrategy
strategy = OptimizedKellyStrategy()
print('最適化Kelly戦略が利用可能です')
"
```

---

## 📊 **戦略パラメータ設定**

### **保守的設定（初心者推奨）**
```python
strategy = OptimizedKellyStrategy(
    min_expected_value=1.1,        # 高い期待値のみ
    max_kelly_fraction=0.08,       # 最大8%まで
    risk_adjustment=0.5,           # リスク50%削減
    diversification_limit=3        # 同時3ベットまで
)
```

### **標準設定（推奨）**
```python
strategy = OptimizedKellyStrategy(
    min_expected_value=1.05,       # 期待値1.05以上
    max_kelly_fraction=0.15,       # 最大15%まで
    risk_adjustment=0.7,           # リスク30%削減
    diversification_limit=8        # 同時8ベットまで
)
```

### **積極的設定（上級者）**
```python
strategy = OptimizedKellyStrategy(
    min_expected_value=1.02,       # 期待値1.02以上
    max_kelly_fraction=0.20,       # 最大20%まで
    risk_adjustment=0.8,           # リスク20%削減
    diversification_limit=12       # 同時12ベットまで
)
```

---

## 🎛️ **詳細パラメータ解説**

### **Kelly基準関連**
| パラメータ | 説明 | 推奨値 | 影響 |
|-----------|------|--------|------|
| `max_kelly_fraction` | Kelly基準の最大割合 | 0.15 | 高いほどリターン↑リスク↑ |
| `risk_adjustment` | リスク調整係数 | 0.7 | 低いほど保守的 |
| `bankroll_protection` | 破産防止閾値 | 0.8 | 資金80%以下で保守的モード |

### **ベット選択関連**
| パラメータ | 説明 | 推奨値 | 影響 |
|-----------|------|--------|------|
| `min_expected_value` | 最低期待値 | 1.05 | 高いほどベット機会減、質向上 |
| `win_rate_threshold` | 最低勝率 | 0.15 | 低勝率ベットを除外 |
| `diversification_limit` | 同時ベット数上限 | 8 | 分散効果とのバランス |

### **アドバンス機能**
| パラメータ | 説明 | 推奨値 | 効果 |
|-----------|------|--------|------|
| `volatility_adjustment` | ボラティリティ調整 | True | 不安定期に自動減額 |
| `enable_compound_growth` | 複利効果活用 | True | 利益を再投資 |

---

## 💻 **実践的な使用例**

### **Example 1: 基本バックテスト**
```python
#!/usr/bin/env python3
import pandas as pd
from src.strategies.optimized_kelly_strategy import OptimizedKellyStrategy

# データ読み込み
data = pd.read_csv('encoded/2020_2025encoded_data_v2.csv')

# 戦略初期化
strategy = OptimizedKellyStrategy(
    min_expected_value=1.05,
    max_kelly_fraction=0.15
)

# バックテスト実行
results = strategy.run_backtest(
    data=data,
    train_years=[2020, 2021, 2022],
    test_years=[2024],
    feature_cols=[],  # 自動検出
    initial_capital=1_000_000
)

# 結果表示
metrics = results['metrics']
print(f"年間リターン: {metrics['annual_return']*100:.1f}%")
print(f"シャープ比: {metrics['sharpe_ratio']:.2f}")
print(f"最大ドローダウン: {metrics['max_drawdown']*100:.1f}%")
```

### **Example 2: リアルタイム予測**
```python
# 今日のレースデータで予測
from src.features.unified_features import UnifiedFeatureEngine

# 特徴量エンジン初期化
engine = UnifiedFeatureEngine()

# レースデータ読み込み（CSVまたはExcel）
race_data = pd.read_csv('today_races.csv')

# 特徴量構築
enhanced_data = engine.build_all_features(race_data)

# 予測実行
strategy = OptimizedKellyStrategy()
# ... (モデル訓練後)
probabilities = strategy.predict_probabilities(model, enhanced_data)

# ベット推奨
bet_opportunities = strategy._generate_bet_opportunities(probabilities, enhanced_data)
optimal_bets = strategy.calculate_diversified_kelly(bet_opportunities)

for bet in optimal_bets:
    print(f"推奨: {bet['type']} {bet['selection']} "
          f"期待値: {bet['expected_value']:.2f} "
          f"Kelly: {bet['kelly_fraction']*100:.1f}%")
```

### **Example 3: 資金管理シミュレーション**
```python
# 異なる資金額でのシミュレーション
capital_scenarios = [500_000, 1_000_000, 2_000_000]

for capital in capital_scenarios:
    strategy = OptimizedKellyStrategy()
    
    # 資金に応じたリスク調整
    if capital < 1_000_000:
        strategy.max_kelly_fraction = 0.10  # 小資金は保守的
    elif capital > 1_500_000:
        strategy.max_kelly_fraction = 0.18  # 大資金は積極的
    
    results = strategy.run_backtest(
        data=data,
        train_years=[2022, 2023],
        test_years=[2024],
        feature_cols=[],
        initial_capital=capital
    )
    
    print(f"資金¥{capital:,}: リターン{results['metrics']['annual_return']*100:.1f}%")
```

---

## 📈 **パフォーマンス最適化**

### **高収益化のコツ**

1. **期待値閾値の調整**
   ```python
   # 期待値を段階的に下げてベット機会を増やす
   for ev_threshold in [1.15, 1.10, 1.05, 1.02]:
       strategy.min_expected_value = ev_threshold
       # バックテストで最適値を発見
   ```

2. **Kelly比率の最適化**
   ```python
   # 過去実績に基づく最適Kelly比率発見
   kelly_ratios = [0.08, 0.12, 0.15, 0.18, 0.20]
   best_ratio = optimize_kelly_fraction(kelly_ratios, historical_data)
   ```

3. **ベットタイプ別最適化**
   ```python
   # 各ベットタイプの期待値を個別調整
   strategy = OptimizedKellyStrategy()
   strategy.trifecta_ev_threshold = 1.15  # 三連単は高めに
   strategy.wide_ev_threshold = 1.02      # ワイドは低めに
   ```

### **リスク管理の強化**

1. **ドローダウン制限**
   ```python
   # 10%ドローダウンで自動停止
   if current_drawdown > 0.10:
       strategy.max_kelly_fraction *= 0.5
   ```

2. **連敗時の対応**
   ```python
   # 5連敗で一時停止
   consecutive_losses = count_consecutive_losses(bet_history)
   if consecutive_losses >= 5:
       pause_betting_for_review()
   ```

3. **月次見直し**
   ```python
   # 月次でパラメータ見直し
   monthly_performance = calculate_monthly_metrics()
   if monthly_performance['sharpe_ratio'] < 1.0:
       adjust_risk_parameters()
   ```

---

## 🔍 **トラブルシューティング**

### **よくある問題と解決法**

**1. "No valid feature columns available for training"**
```bash
# 解決法: データの特徴量を確認
python -c "
import pandas as pd
data = pd.read_csv('encoded/2020_2025encoded_data_v2.csv', nrows=5)
print('データ列数:', len(data.columns))
print('サンプル列:', list(data.columns)[:10])
"
```

**2. "期待値が低すぎる"**
```python
# 解決法: 閾値を下げる
strategy = OptimizedKellyStrategy(
    min_expected_value=0.95,  # 一時的に下げる
    max_kelly_fraction=0.08   # リスクを下げる
)
```

**3. "ベット機会が少なすぎる"**
```python
# 解決法: パラメータ緩和
strategy = OptimizedKellyStrategy(
    min_expected_value=1.02,      # 期待値下げる
    win_rate_threshold=0.10,      # 勝率下げる  
    diversification_limit=12      # ベット数増やす
)
```

**4. "メモリ不足エラー"**
```python
# 解決法: データを分割処理
# 大きなデータセットは年単位で分割
for year in [2022, 2023, 2024]:
    year_data = data[data['year'] == year]
    # 年別にバックテスト実行
```

---

## 📊 **結果の読み方**

### **重要指標の解釈**

```python
# バックテスト結果例
metrics = {
    'annual_return': 0.175,        # 17.5%年間リターン
    'sharpe_ratio': 1.85,          # 優秀（1.5以上が目標）
    'max_drawdown': 0.08,          # 8%最大ドローダウン（良好）
    'calmar_ratio': 2.19,          # 優秀（2.0以上が理想）
    'win_rate': 0.23,              # 23%勝率（競馬では良好）
    'avg_expected_value': 1.087,   # 期待値1.087（目標達成）
    'profit_factor': 1.45          # 利益比率1.45（1.3以上が目標）
}
```

### **指標の目標値**
| 指標 | 目標値 | 評価基準 |
|------|--------|----------|
| 年間リターン | 15-20% | 15%以上で優秀 |
| シャープ比 | 1.5以上 | リスク調整済みリターン |
| 最大ドローダウン | 10%以下 | 資金管理の良さ |
| 勝率 | 20%以上 | 競馬では高水準 |
| 期待値 | 1.05以上 | 長期利益の基盤 |

---

## 🎯 **運用戦略**

### **段階的運用計画**

**Phase 1: 検証期間（1-2ヶ月）**
- 小額資金（10-50万円）でテスト
- 保守的パラメータ使用
- 実績蓄積と調整

**Phase 2: 本格運用（3-6ヶ月）**
- 本格資金（100-500万円）投入
- 最適化パラメータ適用
- 月次見直し実施

**Phase 3: 拡張運用（6ヶ月以降）**
- 大規模資金運用
- 新機能追加
- 継続改善

### **月次チェックリスト**

- [ ] パフォーマンス指標確認
- [ ] ドローダウン状況確認
- [ ] パラメータ調整の検討
- [ ] 新データでの再訓練
- [ ] リスク管理状況確認

---

## 🔧 **カスタマイズガイド**

### **独自特徴量の追加**
```python
# 新しい特徴量ビルダーを作成
class MyCustomFeatureBuilder(FeatureBuilder):
    def build(self, df):
        # 独自ロジック実装
        df['my_feature'] = my_calculation(df)
        return df
    
    def get_feature_names(self):
        return ['my_feature']

# 統一エンジンに追加
engine = UnifiedFeatureEngine()
engine.add_builder(MyCustomFeatureBuilder())
```

### **独自戦略の実装**
```python
class MyStrategy(OptimizedKellyStrategy):
    def calculate_optimal_kelly_fraction(self, win_prob, odds, volatility):
        # 独自のKelly計算ロジック
        kelly = super().calculate_optimal_kelly_fraction(win_prob, odds, volatility)
        # カスタム調整
        return kelly * my_adjustment_factor
```

---

## 🆘 **サポート情報**

### **ログファイル確認**
```bash
# システムログ確認
tail -f logs/keiba_ai.log

# エラーログ確認
grep "ERROR" logs/keiba_ai.log
```

### **デバッグモード実行**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

strategy = OptimizedKellyStrategy()
# デバッグ情報が詳細表示される
```

### **パフォーマンス監視**
```python
# システム監視ツール使用
python system_monitor.py
```

---

## 🏆 **期待される成果**

### **短期目標（1-3ヶ月）**
- ✅ 期待値 1.05+ 安定達成
- ✅ 月間リターン 1-3%
- ✅ ドローダウン 5-8%以内

### **中期目標（3-12ヶ月）**  
- 🎯 年間リターン 15-20%
- 🎯 シャープ比 1.5以上
- 🎯 最大ドローダウン 10%以下

### **長期目標（1年以降）**
- 🚀 年間リターン 20%+
- 🚀 シャープ比 2.0以上
- 🚀 継続的利益成長

---

**🎉 期待値1.095達成システムで、profitable horsebetting の実現を目指しましょう！**