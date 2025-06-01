#!/usr/bin/env python3
"""
期待値計算の問題を分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


def analyze_expectation_accuracy():
    """期待値と実際の結果の乖離を分析"""
    logger = setup_logger('analyze_problem')
    
    # 元のバックテスト結果を読み込み
    trades_file = Path('backtest_kelly25/backtest_trades.csv')
    if not trades_file.exists():
        logger.error("取引履歴ファイルが見つかりません")
        return
        
    trades_df = pd.read_csv(trades_file)
    
    # 期待値を抽出
    expected_values = []
    win_probs = []
    actual_wins = []
    bet_types = []
    
    for _, row in trades_df.iterrows():
        bet_info = eval(row['bet'])  # 文字列をdictに変換
        expected_values.append(float(bet_info['expected_value']))
        win_probs.append(float(bet_info['win_probability']))
        actual_wins.append(row['is_win'])
        bet_types.append(bet_info['type'])
    
    # 分析結果
    print("\n" + "="*60)
    print("期待値分析結果")
    print("="*60)
    
    # 期待値別の実際の勝率
    ev_bins = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 10.0]
    for i in range(len(ev_bins)-1):
        mask = (np.array(expected_values) >= ev_bins[i]) & (np.array(expected_values) < ev_bins[i+1])
        if mask.sum() > 0:
            actual_win_rate = np.array(actual_wins)[mask].mean()
            avg_win_prob = np.array(win_probs)[mask].mean()
            count = mask.sum()
            print(f"\n期待値 {ev_bins[i]:.1f}～{ev_bins[i+1]:.1f}:")
            print(f"  賭け回数: {count}回")
            print(f"  予測勝率: {avg_win_prob*100:.1f}%")
            print(f"  実際勝率: {actual_win_rate*100:.1f}%")
            print(f"  乖離: {(actual_win_rate - avg_win_prob)*100:+.1f}%ポイント")
    
    # 馬券種別の分析
    print("\n\n馬券種別の分析:")
    for bet_type in ['三連単', '馬連', 'ワイド']:
        mask = np.array(bet_types) == bet_type
        if mask.sum() > 0:
            actual_win_rate = np.array(actual_wins)[mask].mean()
            avg_win_prob = np.array(win_probs)[mask].mean()
            avg_ev = np.array(expected_values)[mask].mean()
            count = mask.sum()
            print(f"\n{bet_type}:")
            print(f"  賭け回数: {count}回")
            print(f"  平均期待値: {avg_ev:.2f}")
            print(f"  予測勝率: {avg_win_prob*100:.1f}%")
            print(f"  実際勝率: {actual_win_rate*100:.1f}%")
            print(f"  乖離: {(actual_win_rate - avg_win_prob)*100:+.1f}%ポイント")
    
    # 的中確率別の分析
    print("\n\n的中確率別の分析:")
    prob_bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
    for i in range(len(prob_bins)-1):
        mask = (np.array(win_probs) >= prob_bins[i]) & (np.array(win_probs) < prob_bins[i+1])
        if mask.sum() > 0:
            actual_win_rate = np.array(actual_wins)[mask].mean()
            avg_win_prob = np.array(win_probs)[mask].mean()
            count = mask.sum()
            print(f"\n予測確率 {prob_bins[i]*100:.0f}%～{prob_bins[i+1]*100:.0f}%:")
            print(f"  賭け回数: {count}回")
            print(f"  平均予測勝率: {avg_win_prob*100:.1f}%")
            print(f"  実際勝率: {actual_win_rate*100:.1f}%")
            print(f"  乖離: {(actual_win_rate - avg_win_prob)*100:+.1f}%ポイント")


def suggest_realistic_strategy():
    """現実的な戦略を提案"""
    print("\n\n" + "="*60)
    print("現実的な戦略の提案")
    print("="*60)
    
    print("\n1. 単勝・複勝への切り替え")
    print("   - 的中率が高く、モデルの予測が活きやすい")
    print("   - 期待値1.1程度でも長期的にプラスになる可能性")
    
    print("\n2. 人気馬の馬連BOX")
    print("   - 上位3頭のBOX馬券")
    print("   - 的中率20-30%を狙う")
    
    print("\n3. オッズの歪みを利用")
    print("   - モデル予測とオッズの乖離が大きい馬を狙う")
    print("   - 過小評価されている馬を見つける")
    
    print("\n4. 資金管理の見直し")
    print("   - 固定額ベット（資金の2%など）")
    print("   - 連敗時の自動停止ルール")
    
    print("\n5. データの見直し")
    print("   - 直近1-2年のデータで再訓練")
    print("   - 重賞レースのみに絞る")
    print("   - 馬場状態、距離適性を重視")


if __name__ == "__main__":
    analyze_expectation_accuracy()
    suggest_realistic_strategy()