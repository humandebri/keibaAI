#!/usr/bin/env python3
"""
シンプルな馬連BOX戦略
- 上位N頭の馬連BOX
- 固定額ベット
- 期待値フィルタなし（順位予測のみ）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class SimpleQuinellaBoxStrategy(AdvancedBettingStrategy):
    """シンプルな馬連BOX戦略"""
    
    def __init__(self,
                 box_horses: int = 3,              # BOXに含める頭数
                 bet_fraction: float = 0.02,       # 資金の2%
                 min_horses_in_race: int = 8,      # 最小出走頭数
                 max_popularity: int = 8,          # 最大人気（これ以下のみ）
                 use_only_top_races: bool = False, # 重賞のみ
                 avoid_new_horses: bool = True):   # 新馬戦を避ける
        
        super().__init__(
            min_expected_value=1.0,  # 期待値フィルタは使わない
            enable_trifecta=False,
            enable_quinella=True,
            enable_wide=False,
            use_actual_odds=True,
            kelly_fraction=0.25
        )
        
        self.box_horses = box_horses
        self.bet_fraction = bet_fraction
        self.min_horses_in_race = min_horses_in_race
        self.max_popularity = max_popularity
        self.use_only_top_races = use_only_top_races
        self.avoid_new_horses = avoid_new_horses
        
    def train_model(self) -> lgb.Booster:
        """シンプルなモデル訓練（着順予測に特化）"""
        self.logger.info("着順予測モデルを訓練中...")
        
        features = self._get_features(self.train_data)
        target = self.train_data['着順'].values
        
        # シンプルなパラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        lgb_train = lgb.Dataset(features, target)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def filter_race(self, race_data: pd.DataFrame) -> bool:
        """レースをフィルタリング"""
        # 出走頭数チェック
        if len(race_data) < self.min_horses_in_race:
            return False
            
        # 重賞のみ
        if self.use_only_top_races:
            # レース名やグレードでフィルタリング（データに含まれている場合）
            # ここでは簡易的に実装
            pass
            
        # 新馬戦を避ける
        if self.avoid_new_horses:
            # 出走回数が少ない馬が多い場合はスキップ
            if '出走回数' in race_data.columns:
                if (race_data['出走回数'] <= 1).sum() > len(race_data) * 0.5:
                    return False
                    
        return True
    
    def calculate_bet_amount(self, capital: float) -> float:
        """固定額ベット"""
        bet_amount = capital * self.bet_fraction
        
        # 最低1000円、最大50000円
        bet_amount = max(1000, min(bet_amount, 50000))
        
        # 100円単位に丸める
        return int(bet_amount / 100) * 100
    
    def run_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """シンプルなバックテスト"""
        self.logger.info("シンプルな馬連BOX戦略でバックテスト中...")
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト
        capital = initial_capital
        all_bets = []
        wins = 0
        total_bets = 0
        
        unique_races = self.test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if i % 500 == 0:
                self.logger.debug(f"Processing race {i}/{len(unique_races)}")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            
            # レースフィルタリング
            if not self.filter_race(race_data):
                continue
            
            # 払戻データを取得
            payout_data = self._get_payout_data(race_data)
            
            # 予測
            features = self._get_features(race_data)
            if features is None:
                continue
                
            predicted_positions = model.predict(features)
            
            # 予測順位でソート
            race_data_with_pred = race_data.copy()
            race_data_with_pred['predicted_rank'] = predicted_positions
            race_data_with_pred = race_data_with_pred.sort_values('predicted_rank')
            
            # 人気フィルタ
            race_data_with_pred = race_data_with_pred[race_data_with_pred['人気'] <= self.max_popularity]
            
            if len(race_data_with_pred) < self.box_horses:
                continue
            
            # 上位N頭を選択
            selected_horses = race_data_with_pred.iloc[:self.box_horses]['馬番'].values
            
            # 馬連の組み合わせを生成
            quinella_combinations = list(combinations(selected_horses, 2))
            
            # 各組み合わせに賭ける
            bet_amount_per_combination = self.calculate_bet_amount(capital) / len(quinella_combinations)
            bet_amount_per_combination = max(100, int(bet_amount_per_combination / 100) * 100)
            
            # 実際の結果
            actual_result = race_data.sort_values('着順')
            top2_actual = set(actual_result.iloc[:2]['馬番'].values)
            
            race_total_bet = 0
            race_total_profit = 0
            
            for h1, h2 in quinella_combinations:
                if bet_amount_per_combination > capital:
                    break
                    
                # 的中判定
                is_win = set([h1, h2]) == top2_actual
                
                if is_win:
                    # 実際のオッズを取得
                    actual_odds = self._get_actual_odds(
                        payout_data, '馬連', tuple(sorted([h1, h2]))
                    )
                    if actual_odds and actual_odds > 0:
                        profit = bet_amount_per_combination * (actual_odds / 100) - bet_amount_per_combination
                    else:
                        # オッズが取得できない場合は平均的な配当を仮定
                        profit = bet_amount_per_combination * 10 - bet_amount_per_combination
                    wins += 1
                else:
                    profit = -bet_amount_per_combination
                
                capital += profit
                race_total_bet += bet_amount_per_combination
                race_total_profit += profit
                total_bets += 1
                
                all_bets.append({
                    'race_id': race_id,
                    'horses': (h1, h2),
                    'amount': bet_amount_per_combination,
                    'profit': profit,
                    'capital': capital,
                    'is_win': is_win
                })
            
            # デバッグ情報
            if i < 10 and race_total_bet > 0:
                self.logger.debug(f"Race {race_id}: bet {race_total_bet}, profit {race_total_profit}")
            
            if capital <= 0:
                self.logger.warning("Bankrupt!")
                break
        
        # 結果集計
        total_return = (capital - initial_capital) / initial_capital
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        self.results = {
            'trades': all_bets,
            'metrics': {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'total_bets': total_bets,
                'total_wins': wins,
                'win_rate': win_rate,
                'box_horses': self.box_horses,
                'bet_fraction': self.bet_fraction
            }
        }
        
        return self.results


def optimize_parameters():
    """パラメータを最適化"""
    logger = setup_logger('parameter_optimization')
    
    best_result = None
    best_params = None
    best_return = -float('inf')
    
    # パラメータの組み合わせを試す
    param_combinations = [
        {'box_horses': 3, 'bet_fraction': 0.02, 'max_popularity': 6},
        {'box_horses': 3, 'bet_fraction': 0.03, 'max_popularity': 8},
        {'box_horses': 4, 'bet_fraction': 0.02, 'max_popularity': 8},
        {'box_horses': 4, 'bet_fraction': 0.01, 'max_popularity': 10},
        {'box_horses': 5, 'bet_fraction': 0.01, 'max_popularity': 10},
        {'box_horses': 3, 'bet_fraction': 0.025, 'max_popularity': 5},
        {'box_horses': 2, 'bet_fraction': 0.04, 'max_popularity': 4},
        {'box_horses': 3, 'bet_fraction': 0.02, 'max_popularity': 7, 'min_horses_in_race': 10},
    ]
    
    for i, params in enumerate(param_combinations):
        logger.info(f"\nテスト {i+1}/{len(param_combinations)}: {params}")
        
        # 戦略の初期化
        strategy = SimpleQuinellaBoxStrategy(**params)
        
        # データ読み込み
        strategy.load_data(start_year=2022, end_year=2025, use_payout_data=True)
        
        # 異なるデータ分割でテスト
        if i % 2 == 0:
            strategy.split_data(train_years=[2022, 2023], test_years=[2024, 2025])
        else:
            strategy.split_data(train_years=[2023, 2024], test_years=[2022, 2025])
        
        # バックテスト実行
        results = strategy.run_backtest(initial_capital=1_000_000)
        
        metrics = results['metrics']
        total_return = metrics['total_return']
        
        # 結果表示
        print(f"パラメータ: {params}")
        print(f"収益率: {total_return*100:.1f}%")
        print(f"勝率: {metrics['win_rate']*100:.1f}%")
        print(f"総賭け回数: {metrics['total_bets']}")
        print(f"最終資金: ¥{metrics['final_capital']:,.0f}")
        print("-" * 40)
        
        # 最良結果を更新
        if total_return > best_return:
            best_return = total_return
            best_result = results
            best_params = params
    
    # 最良結果を保存
    output_dir = Path('best_simple_strategy_result')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'best_result.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_params,
            'best_return': best_return,
            'best_metrics': best_result['metrics']
        }, f, ensure_ascii=False, indent=2)
    
    # 最良パラメータで詳細なバックテスト
    if best_return > 0:
        print("\n" + "="*60)
        print("最良パラメータで詳細バックテスト実行")
        print("="*60)
        
        final_strategy = SimpleQuinellaBoxStrategy(**best_params)
        final_strategy.load_data(start_year=2020, end_year=2025, use_payout_data=True)
        final_strategy.split_data(
            train_years=[2020, 2021, 2022, 2023],
            test_years=[2024, 2025]
        )
        
        final_results = final_strategy.run_backtest(initial_capital=1_000_000)
        
        with open(output_dir / 'final_backtest_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results['metrics'], f, ensure_ascii=False, indent=2)
        
        # 取引履歴を保存
        if final_results['trades']:
            trades_df = pd.DataFrame(final_results['trades'])
            trades_df.to_csv(output_dir / 'trades.csv', index=False, encoding='utf-8')
        
        print(f"\n最終結果:")
        print(f"最良パラメータ: {best_params}")
        print(f"収益率: {final_results['metrics']['total_return']*100:.1f}%")
        print(f"最終資金: ¥{final_results['metrics']['final_capital']:,.0f}")
    
    return best_return > 0


if __name__ == "__main__":
    # パラメータ最適化を実行
    success = optimize_parameters()
    
    if not success:
        print("\n収益がプラスになる戦略が見つかりませんでした。")
        print("さらなる改善が必要です。")