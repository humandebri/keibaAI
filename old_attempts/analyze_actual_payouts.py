#!/usr/bin/env python3
"""
実際の配当データを詳細に分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def extract_quinella_payouts(payout_str):
    """馬連の配当データを抽出"""
    if pd.isna(payout_str) or payout_str == '':
        return []
    
    payouts = []
    # パターン: "1-2 1,234円" のような形式
    matches = re.findall(r'(\d+)-(\d+)\s*([0-9,]+)円', str(payout_str))
    
    for h1, h2, amount in matches:
        amount_int = int(amount.replace(',', ''))
        payouts.append({
            'horses': (int(h1), int(h2)),
            'payout': amount_int
        })
    
    return payouts


def analyze_payout_distribution():
    """配当分布を分析"""
    print("配当分布分析")
    print("=" * 60)
    
    # 2024年のデータを分析
    df = pd.read_excel('data_with_payout/2024_with_payout.xlsx')
    
    # 人気順組み合わせ別の配当統計
    popularity_payouts = {}
    
    for _, row in df.iterrows():
        # 馬連払戻列を探す
        quinella_col = None
        for col in df.columns:
            if '馬連' in col and '払戻' in col:
                quinella_col = col
                break
        
        if not quinella_col:
            continue
            
        # 実際の1-2着の人気を取得
        if pd.notna(row['着順']) and row['着順'] == 1:
            race_data = df[df['race_id'] == row['race_id']]
            actual_result = race_data.sort_values('着順')
            
            if len(actual_result) >= 2:
                pop1 = int(actual_result.iloc[0]['人気'])
                pop2 = int(actual_result.iloc[1]['人気'])
                combo = tuple(sorted([pop1, pop2]))
                
                # 配当データを抽出
                payouts = extract_quinella_payouts(row[quinella_col])
                
                # 最初の配当を使用（通常は的中した配当）
                if payouts:
                    payout = payouts[0]['payout']
                    
                    if combo not in popularity_payouts:
                        popularity_payouts[combo] = []
                    popularity_payouts[combo].append(payout)
    
    # 統計を表示
    print("\n人気順組み合わせ別の配当統計:")
    print("組み合わせ  件数   平均配当   中央値   最小   最大")
    print("-" * 60)
    
    sorted_combos = sorted(popularity_payouts.items(), 
                          key=lambda x: len(x[1]), reverse=True)[:20]
    
    for combo, payouts in sorted_combos:
        if len(payouts) >= 5:  # 5件以上のデータがある組み合わせのみ
            avg_payout = np.mean(payouts)
            median_payout = np.median(payouts)
            min_payout = np.min(payouts)
            max_payout = np.max(payouts)
            
            print(f"{combo[0]:2d}-{combo[1]:2d}番人気  {len(payouts):4d}  "
                  f"¥{avg_payout:7,.0f}  ¥{median_payout:7,.0f}  "
                  f"¥{min_payout:6,.0f}  ¥{max_payout:8,.0f}")
    
    return popularity_payouts


def simulate_with_actual_payouts(popularity_payouts):
    """実際の配当データを使用してシミュレーション"""
    print("\n\n実際の配当を使用したシミュレーション")
    print("=" * 60)
    
    strategies = [
        {'name': '1-2番人気固定', 'combos': [(1, 2)], 'bet_pct': 0.05},
        {'name': '1-3番人気固定', 'combos': [(1, 3)], 'bet_pct': 0.04},
        {'name': '1番人気軸流し', 'combos': [(1, 2), (1, 3), (1, 4), (1, 5)], 'bet_pct': 0.02},
        {'name': '高配当狙い', 'combos': [(1, 8), (1, 9), (1, 10)], 'bet_pct': 0.01},
        {'name': '2-3-4BOX', 'combos': [(2, 3), (2, 4), (3, 4)], 'bet_pct': 0.03},
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n戦略: {strategy['name']}")
        
        # 期待値を計算
        total_races = sum(len(payouts) for payouts in popularity_payouts.values())
        hit_count = 0
        total_payout = 0
        
        for combo in strategy['combos']:
            if combo in popularity_payouts:
                hit_count += len(popularity_payouts[combo])
                total_payout += sum(popularity_payouts[combo])
        
        if hit_count > 0:
            hit_rate = hit_count / total_races
            avg_payout = total_payout / hit_count
            expected_value = hit_rate * avg_payout / 100
            
            print(f"  的中率: {hit_rate*100:.2f}%")
            print(f"  平均配当: ¥{avg_payout:,.0f}")
            print(f"  期待値: {expected_value:.3f}")
            
            # 簡易シミュレーション
            capital = 1_000_000
            bet_amount = capital * strategy['bet_pct']
            num_bets = int(capital / bet_amount * 0.8)  # 80%まで賭ける
            
            wins = int(num_bets * hit_rate)
            total_return = wins * avg_payout
            total_cost = num_bets * 100
            profit = total_return - total_cost
            
            print(f"  推定収支: ¥{profit:,.0f} ({profit/capital*100:.1f}%)")
            
            results.append({
                'strategy': strategy['name'],
                'expected_value': expected_value,
                'hit_rate': hit_rate,
                'avg_payout': avg_payout,
                'estimated_profit_rate': profit/capital
            })
    
    # 最良の戦略
    if results:
        best = max(results, key=lambda x: x['expected_value'])
        print(f"\n\n最良の戦略: {best['strategy']}")
        print(f"期待値: {best['expected_value']:.3f}")
        print(f"推定収益率: {best['estimated_profit_rate']*100:.1f}%")


def find_profitable_patterns():
    """利益が出るパターンを探す"""
    print("\n\n利益が出るパターンの探索")
    print("=" * 60)
    
    # 2024年のデータを使用
    df = pd.read_excel('data_with_payout/2024_with_payout.xlsx')
    
    # レースごとの分析
    profitable_patterns = []
    
    for race_id, race_data in df.groupby('race_id'):
        if len(race_data) < 10:  # 10頭以上のレースのみ
            continue
            
        # オッズ情報から期待値の高い馬を探す
        race_data = race_data.copy()
        race_data['implied_prob'] = 100 / (race_data['オッズ'] + 100)
        
        # 人気と実力の乖離を見つける
        actual_result = race_data.sort_values('着順')
        
        for i in range(min(3, len(actual_result))):
            horse = actual_result.iloc[i]
            if horse['人気'] >= 6:  # 6番人気以下が3着以内
                pattern = {
                    'popularity': int(horse['人気']),
                    'finish': int(horse['着順']),
                    'odds': horse['オッズ'],
                    'race_horses': len(race_data)
                }
                profitable_patterns.append(pattern)
    
    # パターンの統計
    print("中穴・大穴馬の好走パターン:")
    print("人気  平均オッズ  3着内率")
    print("-" * 40)
    
    for pop in range(6, 13):
        patterns = [p for p in profitable_patterns if p['popularity'] == pop]
        if patterns:
            avg_odds = np.mean([p['odds'] for p in patterns])
            rate = len(patterns) / 100  # 仮の出現率
            print(f"{pop:2d}番人気  {avg_odds:6.1f}倍  {rate*100:5.1f}%")


if __name__ == "__main__":
    # 配当分布を分析
    popularity_payouts = analyze_payout_distribution()
    
    # 実際の配当でシミュレーション
    simulate_with_actual_payouts(popularity_payouts)
    
    # 利益が出るパターンを探す
    find_profitable_patterns()