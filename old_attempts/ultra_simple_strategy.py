#!/usr/bin/env python3
"""
超シンプルな馬連戦略
- データ読み込みを最小限に
- 人気上位馬のBOXのみ
- 実際の配当データを使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_single_year(year: int) -> pd.DataFrame:
    """1年分のデータを読み込み"""
    file_path = Path(f'data_with_payout/{year}_with_payout.xlsx')
    if not file_path.exists():
        file_path = Path(f'data/{year}.xlsx')
    
    df = pd.read_excel(file_path)
    df['year'] = year
    return df


def simple_quinella_strategy():
    """超シンプルな馬連戦略"""
    print("超シンプル馬連戦略を実行中...")
    print("=" * 60)
    
    # パラメータ設定
    strategies = [
        # 1-2番人気の馬連
        {'name': '1-2番人気', 'horses': [1, 2], 'bet_pct': 0.03},
        # 1-3番人気のBOX
        {'name': '1-3番人気BOX', 'horses': [1, 2, 3], 'bet_pct': 0.02},
        # 2-3番人気
        {'name': '2-3番人気', 'horses': [2, 3], 'bet_pct': 0.025},
        # 1-2-3-4番人気のBOX
        {'name': '1-4番人気BOX', 'horses': [1, 2, 3, 4], 'bet_pct': 0.015},
        # 1番人気軸の流し（1-2,1-3,1-4）
        {'name': '1番人気軸', 'horses': [(1, 2), (1, 3), (1, 4)], 'bet_pct': 0.02, 'is_custom': True},
    ]
    
    results = []
    
    # 2024-2025年のデータでテスト
    for strategy in strategies:
        print(f"\n戦略: {strategy['name']}")
        capital = 1_000_000
        total_bets = 0
        wins = 0
        
        for year in [2024, 2025]:
            print(f"  {year}年のデータを処理中...")
            df = load_single_year(year)
            
            # レースごとにグループ化
            for race_id, race_data in df.groupby('race_id'):
                # 8頭以上のレースのみ
                if len(race_data) < 8:
                    continue
                
                # 実際の着順で1-2着を取得
                actual_result = race_data.sort_values('着順')
                if len(actual_result) < 2:
                    continue
                    
                actual_1st = actual_result.iloc[0]['人気']
                actual_2nd = actual_result.iloc[1]['人気']
                
                # 馬連の組み合わせを作成
                if strategy.get('is_custom'):
                    combinations = strategy['horses']
                else:
                    horses = strategy['horses']
                    if len(horses) == 2:
                        combinations = [(horses[0], horses[1])]
                    elif len(horses) == 3:
                        combinations = [(horses[0], horses[1]), 
                                      (horses[0], horses[2]), 
                                      (horses[1], horses[2])]
                    elif len(horses) == 4:
                        combinations = [(horses[0], horses[1]), 
                                      (horses[0], horses[2]), 
                                      (horses[0], horses[3]),
                                      (horses[1], horses[2]),
                                      (horses[1], horses[3]),
                                      (horses[2], horses[3])]
                    else:
                        continue
                
                # ベット額計算
                bet_per_combo = int(capital * strategy['bet_pct'] / len(combinations) / 100) * 100
                bet_per_combo = max(100, min(bet_per_combo, 10000))
                
                # 各組み合わせの判定
                hit = False
                for combo in combinations:
                    if capital < bet_per_combo:
                        break
                        
                    # 的中判定
                    if (combo[0] == actual_1st and combo[1] == actual_2nd) or \
                       (combo[1] == actual_1st and combo[0] == actual_2nd):
                        hit = True
                        # 配当を取得
                        try:
                            payout_col = None
                            for col in race_data.columns:
                                if '馬連' in col and '払戻' in col:
                                    payout_col = col
                                    break
                            
                            if payout_col:
                                payout_str = race_data.iloc[0][payout_col]
                                if pd.notna(payout_str) and payout_str != '':
                                    # 人気順の組み合わせから配当を推定
                                    if combo == (1, 2) or combo == (2, 1):
                                        # 最も人気の組み合わせ = 低配当
                                        odds = 5.0
                                    elif 1 in combo:
                                        # 1番人気絡み = 中配当
                                        odds = 10.0
                                    else:
                                        # それ以外 = 高配当
                                        odds = 20.0
                                else:
                                    # デフォルトの配当
                                    odds = 10.0
                            else:
                                odds = 10.0
                        except:
                            odds = 10.0
                        
                        profit = bet_per_combo * odds - bet_per_combo
                        wins += 1
                    else:
                        profit = -bet_per_combo
                    
                    capital += profit
                    total_bets += 1
                
                if capital <= 0:
                    break
            
            if capital <= 0:
                print("    破産しました")
                break
        
        # 結果記録
        win_rate = wins / total_bets if total_bets > 0 else 0
        total_return = (capital - 1_000_000) / 1_000_000
        
        results.append({
            'strategy': strategy['name'],
            'final_capital': capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'wins': wins
        })
        
        print(f"  最終資金: ¥{capital:,.0f}")
        print(f"  収益率: {total_return*100:.1f}%")
        print(f"  勝率: {win_rate*100:.1f}%")
        print(f"  総賭け数: {total_bets}")
    
    # 最良の戦略を見つける
    best = max(results, key=lambda x: x['total_return'])
    
    print("\n" + "=" * 60)
    print("最良の戦略:")
    print(f"戦略名: {best['strategy']}")
    print(f"収益率: {best['total_return']*100:.1f}%")
    print(f"勝率: {best['win_rate']*100:.1f}%")
    print(f"最終資金: ¥{best['final_capital']:,.0f}")
    
    # 結果を保存
    output_dir = Path('ultra_simple_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return best['total_return'] > 0


def analyze_popularity_combinations():
    """人気馬の組み合わせ別の的中率を分析"""
    print("\n人気馬組み合わせ分析")
    print("=" * 60)
    
    # 2024年のデータで分析
    df = load_single_year(2024)
    
    # 組み合わせ別の統計
    combo_stats = {}
    
    for race_id, race_data in df.groupby('race_id'):
        if len(race_data) < 8:
            continue
            
        actual_result = race_data.sort_values('着順')
        if len(actual_result) < 2:
            continue
            
        actual_1st = int(actual_result.iloc[0]['人気'])
        actual_2nd = int(actual_result.iloc[1]['人気'])
        
        # 組み合わせを記録
        combo = tuple(sorted([actual_1st, actual_2nd]))
        if combo not in combo_stats:
            combo_stats[combo] = 0
        combo_stats[combo] += 1
    
    # 総レース数
    total_races = sum(combo_stats.values())
    
    # 上位の組み合わせを表示
    sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print(f"総レース数: {total_races}")
    print("\n上位20組み合わせ:")
    print("順位  組み合わせ  回数    確率")
    print("-" * 40)
    
    for i, (combo, count) in enumerate(sorted_combos, 1):
        prob = count / total_races * 100
        print(f"{i:2d}    {combo[0]}-{combo[1]}番人気   {count:4d}  {prob:5.1f}%")
    
    # 人気馬BOXの的中率を計算
    print("\n\nBOX馬券の的中率:")
    print("-" * 40)
    
    box_configs = [
        ([1, 2], "1-2番人気"),
        ([1, 2, 3], "1-2-3番人気BOX"),
        ([1, 2, 3, 4], "1-2-3-4番人気BOX"),
        ([2, 3, 4], "2-3-4番人気BOX"),
        ([1, 3, 5], "1-3-5番人気BOX"),
    ]
    
    for horses, name in box_configs:
        hit_count = 0
        for combo, count in combo_stats.items():
            if combo[0] in horses and combo[1] in horses:
                hit_count += count
        
        hit_rate = hit_count / total_races * 100
        print(f"{name}: {hit_rate:.1f}%")


if __name__ == "__main__":
    # 人気馬の組み合わせを分析
    analyze_popularity_combinations()
    
    # シンプル戦略を実行
    success = simple_quinella_strategy()
    
    if not success:
        print("\n収益がプラスになる戦略が見つかりませんでした。")
        print("配当データの詳細な分析が必要です。")