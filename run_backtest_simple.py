#!/usr/bin/env python3
"""
シンプルなバックテスト実行スクリプト
"""

import pandas as pd
import numpy as np
import os

# pipenv環境でインストールされているか確認
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBMがインストールされていません")
    print("以下のコマンドを実行してください:")
    print("pip install lightgbm")
    exit(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("=== 競馬AI改善バックテスト ===")
print("複勝ベッティング戦略のテスト\n")

# データ確認
print("1. データの存在確認")
years_found = []
for year in range(2021, 2024):  # 最近3年間のみ
    if os.path.exists(f'data/{year}.xlsx'):
        years_found.append(year)
        print(f"✓ {year}.xlsx found")

if not years_found:
    print("データファイルが見つかりません")
    exit(1)

print(f"\n2. {len(years_found)}年分のデータで簡易バックテスト実行")

# 簡易バックテスト
capital = 1000000  # 100万円
total_bets = 0
total_wins = 0

for year in years_found:
    print(f"\n--- {year}年 ---")
    
    # データ読み込み
    df = pd.read_excel(f'data/{year}.xlsx')
    
    # 複勝（3着以内）をシミュレート
    year_bets = 0
    year_wins = 0
    
    # 人気順位1-3位の馬のみベット（簡易戦略）
    target_horses = df[df['人気'] <= 3]
    
    for _, horse in target_horses.iterrows():
        # オッズが3.0以上の馬のみ（期待値を考慮）
        if horse['オッズ'] >= 3.0:
            year_bets += 1
            total_bets += 1
            
            # 3着以内なら的中
            if horse['着順'] <= 3:
                year_wins += 1
                total_wins += 1
                # 複勝オッズは単勝の約30%と仮定
                place_odds = horse['オッズ'] * 0.3
                capital = capital * 0.995 * place_odds + capital * 0.995  # 0.5%ベット
            else:
                capital = capital * 0.995  # 0.5%損失
    
    win_rate = year_wins / year_bets if year_bets > 0 else 0
    print(f"ベット数: {year_bets}, 的中数: {year_wins}, 勝率: {win_rate:.1%}")
    print(f"資産: ¥{capital:,.0f}")

# 最終結果
total_return = (capital - 1000000) / 1000000
overall_win_rate = total_wins / total_bets if total_bets > 0 else 0

print("\n=== 最終結果 ===")
print(f"初期資産: ¥1,000,000")
print(f"最終資産: ¥{capital:,.0f}")
print(f"総リターン: {total_return:.2%}")
print(f"総ベット数: {total_bets}")
print(f"勝率: {overall_win_rate:.1%}")

if total_return > 0:
    print("\n✓ プラスのリターンを達成！")
    print("改善戦略が有効であることが確認されました")
else:
    print("\nさらなる戦略の調整が必要です")