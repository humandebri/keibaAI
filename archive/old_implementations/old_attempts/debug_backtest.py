#!/usr/bin/env python3
"""
デバッグ用バックテストスクリプト
実際のベット判定と結果を詳しく確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger
import json


def analyze_payout_data():
    """払戻データの分析"""
    print("\n=== 払戻データの分析 ===")
    
    # サンプルレースの払戻データを確認
    df = pd.read_excel("data_with_payout/2025_with_payout.xlsx", nrows=100)
    
    if '払戻データ' in df.columns:
        # 最初のレースの払戻データを詳しく見る
        for idx, row in df.iterrows():
            if pd.notna(row['払戻データ']):
                try:
                    payout = json.loads(row['払戻データ'])
                    race_id = row['race_id']
                    print(f"\nレース {race_id} の払戻データ:")
                    
                    # 各馬券種のデータを確認
                    for bet_type in ['win', 'place', 'quinella', 'wide', 'trifecta']:
                        if bet_type in payout and payout[bet_type]:
                            print(f"\n{bet_type}:")
                            items = list(payout[bet_type].items())[:3]  # 最初の3つ
                            for k, v in items:
                                print(f"  {k}: {v}円")
                    
                    # 最初のレースだけ確認
                    break
                except:
                    pass


def test_win_checking():
    """勝敗判定のテスト"""
    print("\n\n=== 勝敗判定のテスト ===")
    
    # 戦略の初期化
    strategy = AdvancedBettingStrategy(
        min_expected_value=1.0,  # 低く設定してベットを増やす
        enable_trifecta=False,
        enable_quinella=True,
        enable_wide=True,
        use_actual_odds=True
    )
    
    # テスト用のレース結果
    test_result = pd.DataFrame({
        '馬番': [5, 3, 1, 7, 2],
        '着順': [1, 2, 3, 4, 5]
    })
    
    print("テストレース結果:")
    print(test_result[['馬番', '着順']].head())
    
    # 各馬券種のテスト
    test_bets = [
        {'type': '馬連', 'selection': (3, 5)},      # 1-2着 → 勝ち
        {'type': '馬連', 'selection': (1, 5)},      # 1-3着 → 負け
        {'type': 'ワイド', 'selection': (1, 3)},    # 2-3着 → 勝ち
        {'type': 'ワイド', 'selection': (2, 7)},    # 4-5着 → 負け
        {'type': '三連単', 'selection': (5, 3, 1)}, # 1-2-3着 → 勝ち
    ]
    
    print("\n判定結果:")
    for bet in test_bets:
        is_win, odds = strategy._check_result(bet, test_result)
        print(f"{bet['type']} {bet['selection']}: {'勝ち' if is_win else '負け'} (オッズ: {odds})")


def run_detailed_backtest():
    """詳細なバックテスト実行"""
    print("\n\n=== 詳細バックテスト ===")
    
    # 戦略の初期化
    strategy = AdvancedBettingStrategy(
        min_expected_value=1.1,
        enable_trifecta=True,
        enable_quinella=True,
        enable_wide=True,
        use_actual_odds=True
    )
    
    # データ読み込み
    strategy.load_data(start_year=2025, end_year=2025, use_payout_data=True)
    strategy.split_data(train_years=[2025], test_years=[2025])
    
    # 簡易モデル訓練
    print("\nモデル訓練中...")
    model = strategy.train_model()
    
    # 最初の10レースだけテスト
    unique_races = strategy.test_data['race_id'].unique()[:10]
    
    total_bets = 0
    wins = 0
    bet_details = []
    
    for race_id in unique_races:
        race_data = strategy.test_data[strategy.test_data['race_id'] == race_id]
        if len(race_data) < 8:
            continue
        
        # 払戻データ取得
        payout_data = strategy._get_payout_data(race_data)
        
        # 確率予測
        probs = strategy.predict_probabilities(model, race_data)
        if not probs:
            continue
        
        print(f"\n\nレース {race_id}:")
        print(f"出走頭数: {len(race_data)}")
        
        # 実際の結果
        actual_result = race_data.sort_values('着順')
        print(f"実際の着順: 1着={actual_result.iloc[0]['馬番']}, "
              f"2着={actual_result.iloc[1]['馬番']}, "
              f"3着={actual_result.iloc[2]['馬番']}")
        
        # 予測順位でソート
        sorted_horses = sorted(probs.items(), key=lambda x: x[1]['predicted_rank'])
        
        # ワイドベットの例
        if len(sorted_horses) >= 5:
            top5 = [h[0] for h in sorted_horses[:5]]
            print(f"\n予測上位5頭: {top5}")
            
            # 上位5頭から2頭の組み合わせ
            from itertools import combinations
            for h1, h2 in combinations(top5[:3], 2):  # 上位3頭のみ
                # 実際のオッズ取得
                actual_odds = strategy._get_actual_odds(
                    payout_data, 'wide', tuple(sorted([h1, h2]))
                )
                
                # 期待値計算
                ev, wp, odds = strategy.calculate_wide_ev(probs, h1, h2, actual_odds)
                
                if ev >= 1.1:  # 期待値1.1以上
                    bet = {
                        'type': 'ワイド',
                        'selection': tuple(sorted([h1, h2])),
                        'expected_value': ev,
                        'actual_odds': actual_odds
                    }
                    
                    # 結果判定
                    is_win, result_odds = strategy._check_result(bet, actual_result)
                    
                    print(f"\nワイド {h1}-{h2}:")
                    print(f"  期待値: {ev:.2f}")
                    print(f"  実際のオッズ: {actual_odds}")
                    print(f"  結果: {'勝ち' if is_win else '負け'}")
                    
                    total_bets += 1
                    if is_win:
                        wins += 1
                    
                    bet_details.append({
                        'race_id': race_id,
                        'type': 'ワイド',
                        'selection': f"{h1}-{h2}",
                        'ev': ev,
                        'actual_odds': actual_odds,
                        'is_win': is_win
                    })
    
    print(f"\n\n=== サマリー ===")
    print(f"総ベット数: {total_bets}")
    print(f"勝利数: {wins}")
    print(f"勝率: {wins/total_bets*100:.1f}%" if total_bets > 0 else "N/A")
    
    # 詳細をDataFrameで表示
    if bet_details:
        df_bets = pd.DataFrame(bet_details)
        print("\n\n詳細結果:")
        print(df_bets.to_string())


def main():
    """メイン処理"""
    print("=== バックテストデバッグ ===")
    
    # 1. 払戻データの確認
    analyze_payout_data()
    
    # 2. 勝敗判定のテスト
    test_win_checking()
    
    # 3. 実際のバックテスト（小規模）
    run_detailed_backtest()


if __name__ == "__main__":
    main()