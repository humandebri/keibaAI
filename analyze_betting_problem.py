#!/usr/bin/env python3
"""
期待値問題の分析と解決策
"""

import pandas as pd
import numpy as np

def analyze_betting_problem():
    """期待値が1.0未満である問題を分析"""
    
    # 実データ
    odds_data = {
        'オッズ帯': ['1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+'],
        '複勝率': [0.795073, 0.561327, 0.365861, 0.235511, 0.128378, 0.062087, 0.019867],
        '平均オッズ': [1.708684, 3.607282, 7.269251, 14.620773, 32.414821, 71.569501, 217.513703],
        '複勝期待値': [0.611338, 0.708703, 0.797860, 0.860837, 0.832270, 0.666530, 0.648209]
    }
    
    df = pd.DataFrame(odds_data)
    
    print("=== 現状の問題分析 ===\n")
    print("全てのオッズ帯で期待値 < 1.0 （最高でも0.861）")
    print("\nこれは以下を意味します：")
    print("1. 控除率（テラ銭）の影響で、そもそも期待値が1.0を超えることは困難")
    print("2. 現在の複勝オッズ推定が実際より低い可能性")
    print("3. 予測精度を上げるだけでは限界がある")
    
    print("\n=== 各オッズ帯の詳細分析 ===")
    for _, row in df.iterrows():
        # 期待値を1.0にするために必要な複勝率を計算
        current_place_rate = row['複勝率']
        current_ev = row['複勝期待値']
        
        # 複勝オッズの推定（現在の期待値から逆算）
        estimated_place_odds = current_ev / current_place_rate
        
        # 期待値1.0に必要な複勝率
        required_rate_for_ev1 = 1.0 / estimated_place_odds
        
        # 必要な改善率
        improvement_needed = (required_rate_for_ev1 / current_place_rate - 1) * 100
        
        print(f"\n{row['オッズ帯']}倍:")
        print(f"  現在の複勝率: {current_place_rate:.1%}")
        print(f"  推定複勝オッズ: {estimated_place_odds:.2f}")
        print(f"  現在の期待値: {current_ev:.3f}")
        print(f"  期待値1.0に必要な複勝率: {required_rate_for_ev1:.1%}")
        print(f"  必要な的中率改善: +{improvement_needed:.0f}%")
    
    print("\n=== 解決策の提案 ===")
    print("\n1. 【現実的】期待値が高いオッズ帯に集中")
    print("   - 10-20倍に特化（期待値0.861）")
    print("   - 5-20倍のゾーンのみで勝負")
    print("   - 期待値0.85以上の馬のみに賭ける")
    
    print("\n2. 【高度な戦略】複合的アプローチ")
    print("   - 出走取消や競走中止を考慮")
    print("   - レース選択（条件の良いレースのみ）")
    print("   - 時期選択（控除率が低い時期）")
    
    print("\n3. 【現実的な目標設定】")
    print("   - 期待値0.95を目指す（年率-5%の損失）")
    print("   - ベット数を減らして高確率の勝負のみ")
    print("   - 資金管理で長期継続を重視")
    
    return df

def calculate_realistic_strategy():
    """現実的な戦略の計算"""
    
    print("\n\n=== 現実的な戦略シミュレーション ===")
    
    # 10-20倍ゾーンに特化した場合
    odds_10_20 = {
        'rate': 0.235511,
        'ev': 0.860837,
        'avg_odds': 14.620773
    }
    
    # より厳しい選別を行った場合（上位20%のみ）
    selected_rate = odds_10_20['rate'] * 1.5  # 予測精度向上を仮定
    selected_ev = selected_rate * (odds_10_20['avg_odds'] * 0.25)  # 複勝オッズ
    
    print(f"\n厳選戦略（10-20倍の上位馬のみ）:")
    print(f"  想定複勝率: {selected_rate:.1%}")
    print(f"  想定期待値: {selected_ev:.3f}")
    
    # 年間シミュレーション
    annual_bets = 500  # 厳選して年間500レース
    bet_amount = 1000  # 1レース1000円
    
    expected_return = bet_amount * selected_ev
    expected_profit = (expected_return - bet_amount) * annual_bets
    roi = expected_profit / (bet_amount * annual_bets)
    
    print(f"\n年間シミュレーション:")
    print(f"  年間ベット数: {annual_bets}")
    print(f"  総投資額: ¥{bet_amount * annual_bets:,}")
    print(f"  期待リターン: ¥{expected_return * annual_bets:,.0f}")
    print(f"  期待損益: ¥{expected_profit:,.0f}")
    print(f"  ROI: {roi:.1%}")
    
    print("\n=== 結論 ===")
    print("完全にプラスにすることは困難だが、以下で損失を最小化可能：")
    print("1. 10-20倍のオッズ帯に集中")
    print("2. モデルの予測上位のみに厳選")
    print("3. 年間-10%程度の損失に抑える")
    print("4. エンターテイメントとして楽しむ")

if __name__ == "__main__":
    df = analyze_betting_problem()
    calculate_realistic_strategy()