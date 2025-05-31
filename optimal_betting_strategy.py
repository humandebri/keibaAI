#!/usr/bin/env python3
"""
オッズ帯別分析に基づく最適賭け戦略
"""

import pandas as pd
import numpy as np

class OptimalBettingStrategy:
    def __init__(self):
        # 実際のデータから得られたオッズ帯別統計
        self.odds_stats = {
            '1-2': {'place_rate': 0.795, 'place_ev': 0.611},
            '2-5': {'place_rate': 0.561, 'place_ev': 0.709},
            '5-10': {'place_rate': 0.366, 'place_ev': 0.798},
            '10-20': {'place_rate': 0.236, 'place_ev': 0.861},
            '20-50': {'place_rate': 0.128, 'place_ev': 0.832},
            '50-100': {'place_rate': 0.062, 'place_ev': 0.667},
            '100+': {'place_rate': 0.020, 'place_ev': 0.648}
        }
        
    def get_odds_category(self, odds):
        """オッズをカテゴリに分類"""
        if odds <= 2:
            return '1-2'
        elif odds <= 5:
            return '2-5'
        elif odds <= 10:
            return '5-10'
        elif odds <= 20:
            return '10-20'
        elif odds <= 50:
            return '20-50'
        elif odds <= 100:
            return '50-100'
        else:
            return '100+'
    
    def calculate_optimal_ev_threshold(self, odds):
        """オッズに応じた最適な期待値閾値を計算"""
        category = self.get_odds_category(odds)
        base_ev = self.odds_stats[category]['place_ev']
        
        # 期待値が低いオッズ帯ほど高い閾値を要求
        if category == '1-2':
            return 1.5  # 期待値0.611なので高い閾値が必要
        elif category == '2-5':
            return 1.3  # 期待値0.709
        elif category in ['5-10', '10-20', '20-50']:
            return 1.15  # 期待値0.798-0.861（最も効率的）
        else:
            return 1.4  # 期待値が低いので慎重に
    
    def calculate_kelly_fraction(self, odds, win_prob, bankroll_fraction=0.25):
        """Kelly基準に基づくベット額の計算（改良版）"""
        category = self.get_odds_category(odds)
        
        # オッズ帯別の基本ベット率
        base_fractions = {
            '1-2': 0.002,    # 低オッズは控えめに
            '2-5': 0.003,    
            '5-10': 0.005,   # 最適ゾーン
            '10-20': 0.006,  # 最も期待値が高い
            '20-50': 0.004,  # やや控えめに
            '50-100': 0.002, # リスク管理
            '100+': 0.001    # 最小限に
        }
        
        base_fraction = base_fractions[category]
        
        # Kelly基準での調整
        # f = (p * b - q) / b
        # p: 勝率, q: 負け率(1-p), b: オッズ-1
        p = win_prob
        q = 1 - p
        b = odds - 1
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # 安全のため、Kelly基準の25%のみ使用
        kelly_fraction = max(0, kelly_fraction * bankroll_fraction)
        
        # 基本ベット率とKelly基準の平均を取る
        final_fraction = (base_fraction + kelly_fraction) / 2
        
        # 上限設定（資金の1%まで）
        return min(final_fraction, 0.01)
    
    def should_bet(self, odds, predicted_prob, model_confidence=1.0):
        """賭けるべきかどうかの判断"""
        category = self.get_odds_category(odds)
        
        # 複勝オッズの推定
        place_odds = self.estimate_place_odds(odds)
        
        # 期待値の計算
        ev = predicted_prob * place_odds
        
        # オッズ帯別の閾値
        threshold = self.calculate_optimal_ev_threshold(odds)
        
        # モデルの信頼度で調整
        adjusted_threshold = threshold / model_confidence
        
        # 追加条件：期待値が極端に低いオッズ帯は避ける
        if category in ['1-2', '100+'] and ev < 1.0:
            return False, 0, ev
        
        # 最適ゾーン（5-20倍）は条件を緩和
        if category in ['5-10', '10-20'] and ev > 1.1:
            return True, self.calculate_kelly_fraction(odds, predicted_prob), ev
        
        # 通常の判定
        if ev > adjusted_threshold:
            fraction = self.calculate_kelly_fraction(odds, predicted_prob)
            return True, fraction, ev
        
        return False, 0, ev
    
    def estimate_place_odds(self, win_odds):
        """複勝オッズの推定（実データに基づく改良版）"""
        if win_odds <= 1.5:
            return 1.1
        elif win_odds <= 2.0:
            return win_odds * 0.45
        elif win_odds <= 3.0:
            return win_odds * 0.40
        elif win_odds <= 5.0:
            return win_odds * 0.35
        elif win_odds <= 10.0:
            return win_odds * 0.30
        elif win_odds <= 20.0:
            return win_odds * 0.25
        elif win_odds <= 50.0:
            return win_odds * 0.20
        else:
            return win_odds * 0.15
    
    def get_strategy_summary(self):
        """戦略のサマリーを返す"""
        return {
            "最適オッズ帯": "5-20倍（特に10-20倍）",
            "回避すべきオッズ帯": "1-2倍、100倍以上",
            "推奨ベット率": {
                "5-10倍": "資金の0.5%",
                "10-20倍": "資金の0.6%",
                "その他": "資金の0.2-0.4%"
            },
            "期待値閾値": {
                "5-20倍": "1.15以上",
                "それ以外": "1.3-1.5以上"
            }
        }

# 使用例
def demonstrate_strategy():
    strategy = OptimalBettingStrategy()
    
    # テストケース
    test_cases = [
        {"odds": 1.5, "predicted_prob": 0.8, "name": "大本命"},
        {"odds": 3.5, "predicted_prob": 0.6, "name": "本命"},
        {"odds": 8.0, "predicted_prob": 0.4, "name": "中穴"},
        {"odds": 15.0, "predicted_prob": 0.25, "name": "穴馬"},
        {"odds": 35.0, "predicted_prob": 0.15, "name": "大穴"},
        {"odds": 120.0, "predicted_prob": 0.05, "name": "超大穴"}
    ]
    
    print("=== 最適賭け戦略デモンストレーション ===\n")
    
    for case in test_cases:
        should_bet, fraction, ev = strategy.should_bet(
            case["odds"], 
            case["predicted_prob"]
        )
        
        print(f"{case['name']} (オッズ: {case['odds']}, 予測確率: {case['predicted_prob']:.0%})")
        print(f"  期待値: {ev:.3f}")
        print(f"  賭けるべきか: {'YES' if should_bet else 'NO'}")
        if should_bet:
            print(f"  推奨ベット率: {fraction:.1%}")
        print()
    
    print("\n=== 戦略サマリー ===")
    summary = strategy.get_strategy_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    demonstrate_strategy()