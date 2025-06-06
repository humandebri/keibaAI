#!/usr/bin/env python3
"""
統合システムのシミュレーションテスト
"""

import pandas as pd
import numpy as np
from integrated_betting_system import IntegratedKeibaSystem
import asyncio
import warnings
warnings.filterwarnings('ignore')


async def test_simulation():
    # 設定
    config = {
        'model_path': 'model_2020_2025/model_2020_2025.pkl',
        'enable_auto_betting': False,
        'min_expected_value': 1.2,
        'kelly_fraction': 0.025,
        'max_bet_per_race': 5000,
        'max_daily_loss': 30000,
        'simulation_mode': True,
        'simulation_files': ['live_race_data_202505021212.csv'],
        'data_refresh_interval': 300
    }
    
    print("=" * 60)
    print("統合競馬予測・投票システム - シミュレーションテスト")
    print("=" * 60)
    
    # システム初期化
    system = IntegratedKeibaSystem(config)
    print(f"\n✅ モデル読み込み完了: {len(system.feature_cols)}個の特徴量")
    
    # CSVファイル読み込み
    csv_file = 'live_race_data_202505021212.csv'
    print(f"\n📄 テストデータ: {csv_file}")
    
    race_df = pd.read_csv(csv_file)
    print(f"   出走頭数: {len(race_df)}頭")
    
    # 予測用データ準備
    prediction_df = system._prepare_prediction_data_from_csv(race_df)
    print(f"   予測用データ準備完了: {prediction_df.shape}")
    
    # 予測実行
    print("\n🔮 予測実行中...")
    predictions = system._run_prediction(prediction_df)
    
    # 結果表示
    print("\n🏇 予測結果:")
    print("=" * 80)
    print(f"{'順位':^4} {'馬番':^4} {'馬名':^20} {'オッズ':^8} {'勝率':^8} {'期待値':^8}")
    print("=" * 80)
    
    # 勝率でソート
    sorted_predictions = predictions.sort_values('win_probability', ascending=False)
    
    for i, (_, row) in enumerate(sorted_predictions.head(10).iterrows(), 1):
        horse_name = row.get('馬名', 'Unknown')
        if len(horse_name) > 10:
            horse_name = horse_name[:10] + '...'
        
        print(f"{i:4d} {int(row['馬番']):4d} {horse_name:^20s} "
              f"{row.get('オッズ', 0):7.1f}倍 {row['win_probability']*100:6.1f}% "
              f"{row.get('オッズ', 0) * row['win_probability']:7.2f}")
    
    # 統計情報
    print("\n📊 統計情報:")
    print(f"   勝率合計: {predictions['win_probability'].sum()*100:.1f}%")
    print(f"   最高勝率: {predictions['win_probability'].max()*100:.1f}%")
    print(f"   最低勝率: {predictions['win_probability'].min()*100:.1f}%")
    print(f"   平均勝率: {predictions['win_probability'].mean()*100:.1f}%")
    
    # ベッティング機会の分析
    race_details = {'race_info': {'distance': 2500, 'surface': '芝'}}
    betting_opportunities = system._analyze_betting_opportunities(predictions, race_details)
    
    if betting_opportunities:
        print(f"\n💰 ベッティング機会: {len(betting_opportunities)}件")
        for opp in betting_opportunities[:3]:
            print(f"   馬番{opp['horse_number']:2d}: "
                  f"期待値{opp['expected_value']:.2f} "
                  f"(勝率{opp['win_probability']*100:.1f}% × {opp['odds']:.1f}倍)")
    else:
        print("\n💰 期待値1.2以上の馬はありませんでした")


if __name__ == "__main__":
    asyncio.run(test_simulation())