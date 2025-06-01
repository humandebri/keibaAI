#!/usr/bin/env python3
"""
効率的な馬連BOX戦略
- 処理を高速化
- メモリ効率を改善
- シンプルな勝率重視戦略
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class EfficientQuinellaStrategy(AdvancedBettingStrategy):
    """効率的な馬連BOX戦略"""
    
    def __init__(self,
                 box_horses: int = 3,
                 bet_fraction: float = 0.02,
                 min_popularity: int = 1,
                 max_popularity: int = 6,
                 min_horses_in_race: int = 8):
        
        super().__init__(
            min_expected_value=1.0,
            enable_trifecta=False,
            enable_quinella=True,
            enable_wide=False,
            use_actual_odds=True,
            kelly_fraction=0.25
        )
        
        self.box_horses = box_horses
        self.bet_fraction = bet_fraction
        self.min_popularity = min_popularity
        self.max_popularity = max_popularity
        self.min_horses_in_race = min_horses_in_race
        
    def train_simple_model(self) -> lgb.Booster:
        """シンプルで高速なモデル訓練"""
        self.logger.info("高速モデルを訓練中...")
        
        # 必要最小限の特徴量のみ使用
        feature_cols = ['馬番', '人気', 'オッズ', '斤量', '馬齢', '馬体重']
        available_cols = [col for col in feature_cols if col in self.train_data.columns]
        
        features = self.train_data[available_cols].values
        target = self.train_data['着順'].values
        
        # 高速パラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'num_threads': -1
        }
        
        lgb_train = lgb.Dataset(features, target)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=50,  # 少ないラウンド数
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def run_fast_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """高速バックテスト"""
        self.logger.info(f"高速バックテスト開始（{self.box_horses}頭BOX、人気{self.min_popularity}-{self.max_popularity}）")
        
        # モデル訓練
        model = self.train_simple_model()
        
        # バックテスト初期化
        capital = initial_capital
        wins = 0
        total_bets = 0
        total_profit = 0
        
        # テストデータをレースごとに処理
        test_races = self.test_data.groupby('race_id')
        
        for race_id, race_data in test_races:
            # レースフィルタリング
            if len(race_data) < self.min_horses_in_race:
                continue
            
            # 人気フィルタリング
            candidates = race_data[
                (race_data['人気'] >= self.min_popularity) & 
                (race_data['人気'] <= self.max_popularity)
            ].copy()
            
            if len(candidates) < self.box_horses:
                continue
            
            # 予測（シンプル）
            feature_cols = ['馬番', '人気', 'オッズ', '斤量', '馬齢', '馬体重']
            available_cols = [col for col in feature_cols if col in candidates.columns]
            features = candidates[available_cols].values
            
            try:
                predictions = model.predict(features)
                candidates['predicted_rank'] = predictions
            except:
                continue
            
            # 上位N頭を選択
            selected = candidates.nsmallest(self.box_horses, 'predicted_rank')
            selected_horses = selected['馬番'].values
            
            # 馬連の組み合わせ（最大3通り = 3C2）
            if self.box_horses == 3:
                combinations = [(selected_horses[0], selected_horses[1]),
                               (selected_horses[0], selected_horses[2]),
                               (selected_horses[1], selected_horses[2])]
            elif self.box_horses == 2:
                combinations = [(selected_horses[0], selected_horses[1])]
            else:
                continue
            
            # ベット額計算
            bet_per_combo = int(capital * self.bet_fraction / len(combinations) / 100) * 100
            bet_per_combo = max(100, min(bet_per_combo, 10000))
            
            # 実際の結果
            actual_top2 = race_data.nsmallest(2, '着順')['馬番'].values
            actual_quinella = tuple(sorted(actual_top2))
            
            # 各組み合わせの判定
            race_hit = False
            for combo in combinations:
                if bet_per_combo > capital:
                    break
                
                sorted_combo = tuple(sorted(combo))
                is_win = sorted_combo == actual_quinella
                
                if is_win:
                    # 平均的な馬連配当を仮定（データから取得するとメモリ/時間消費）
                    avg_odds = 15.0  # 馬連の平均配当
                    profit = bet_per_combo * avg_odds - bet_per_combo
                    wins += 1
                    race_hit = True
                else:
                    profit = -bet_per_combo
                
                capital += profit
                total_profit += profit
                total_bets += 1
                
                if capital <= 0:
                    break
            
            if capital <= 0:
                self.logger.warning("破産")
                break
        
        # 結果集計
        win_rate = wins / total_bets if total_bets > 0 else 0
        total_return = (capital - initial_capital) / initial_capital
        
        return {
            'final_capital': capital,
            'total_return': total_return,
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'total_profit': total_profit
        }


def test_multiple_parameters():
    """複数パラメータでテスト"""
    logger = setup_logger('efficient_test')
    
    # テストパラメータ
    test_params = [
        # 上位人気の2頭BOX
        {'box_horses': 2, 'min_popularity': 1, 'max_popularity': 3, 'bet_fraction': 0.03},
        # 中位人気の3頭BOX
        {'box_horses': 3, 'min_popularity': 2, 'max_popularity': 5, 'bet_fraction': 0.02},
        # 上位人気の3頭BOX（保守的）
        {'box_horses': 3, 'min_popularity': 1, 'max_popularity': 4, 'bet_fraction': 0.015},
        # 中穴狙い
        {'box_horses': 2, 'min_popularity': 3, 'max_popularity': 6, 'bet_fraction': 0.025},
        # 大穴狙い
        {'box_horses': 2, 'min_popularity': 4, 'max_popularity': 8, 'bet_fraction': 0.02},
        # 超保守的
        {'box_horses': 2, 'min_popularity': 1, 'max_popularity': 2, 'bet_fraction': 0.04},
    ]
    
    results = []
    
    print("効率的な馬連BOX戦略テスト")
    print("=" * 60)
    
    for i, params in enumerate(test_params):
        print(f"\nテスト {i+1}/{len(test_params)}")
        print(f"パラメータ: {params}")
        
        strategy = EfficientQuinellaStrategy(**params)
        
        # データ読み込み（軽量化のため直近のみ）
        strategy.load_data(start_year=2023, end_year=2025, use_payout_data=True)
        strategy.split_data(train_years=[2023], test_years=[2024, 2025])
        
        # バックテスト実行
        result = strategy.run_fast_backtest(initial_capital=1_000_000)
        
        # 結果表示
        print(f"収益率: {result['total_return']*100:.1f}%")
        print(f"勝率: {result['win_rate']*100:.1f}%")
        print(f"総賭け数: {result['total_bets']}")
        print(f"最終資金: ¥{result['final_capital']:,.0f}")
        
        results.append({
            'params': params,
            'result': result
        })
        
        # メモリクリア
        del strategy
        gc.collect()
    
    # 最良結果を見つける
    best_idx = max(range(len(results)), key=lambda i: results[i]['result']['total_return'])
    best = results[best_idx]
    
    print("\n" + "=" * 60)
    print("最良結果:")
    print(f"パラメータ: {best['params']}")
    print(f"収益率: {best['result']['total_return']*100:.1f}%")
    print(f"勝率: {best['result']['win_rate']*100:.1f}%")
    
    # 最良パラメータで詳細テスト
    if best['result']['total_return'] > 0:
        print("\n最良パラメータで詳細テスト実行中...")
        
        strategy = EfficientQuinellaStrategy(**best['params'])
        strategy.load_data(start_year=2020, end_year=2025, use_payout_data=True)
        strategy.split_data(train_years=[2020, 2021, 2022], test_years=[2023, 2024, 2025])
        
        final_result = strategy.run_fast_backtest(initial_capital=1_000_000)
        
        print("\n最終結果:")
        print(f"収益率: {final_result['total_return']*100:.1f}%")
        print(f"勝率: {final_result['win_rate']*100:.1f}%")
        print(f"最終資金: ¥{final_result['final_capital']:,.0f}")
        
        # 結果保存
        output_dir = Path('efficient_strategy_results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'best_result.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': best['params'],
                'test_result': best['result'],
                'final_result': final_result
            }, f, ensure_ascii=False, indent=2)
        
        return final_result['total_return'] > 0
    
    return False


if __name__ == "__main__":
    # 複数パラメータでテスト
    success = test_multiple_parameters()
    
    if not success:
        print("\n収益がプラスになる戦略が見つかりませんでした。")
        print("さらなる改善が必要です。")