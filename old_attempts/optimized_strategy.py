#!/usr/bin/env python3
"""
最適化された競馬予測戦略
- Optunaでハイパーパラメータ最適化
- 単勝・複勝・馬連を組み合わせた現実的な戦略
- 過学習防止のための交差検証
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import warnings
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class OptimizedBettingStrategy(AdvancedBettingStrategy):
    """最適化された馬券戦略"""
    
    def __init__(self,
                 min_expected_value: float = 1.05,
                 enable_win: bool = True,         # 単勝
                 enable_place: bool = True,       # 複勝
                 enable_quinella: bool = True,    # 馬連
                 enable_wide: bool = True,        # ワイド
                 enable_trifecta: bool = False,   # 三連単（無効）
                 use_actual_odds: bool = True,
                 bet_fraction: float = 0.02,      # 固定2%ベット
                 max_horses: int = 3):            # 最大3頭まで
        
        super().__init__(
            min_expected_value=min_expected_value,
            enable_trifecta=enable_trifecta,
            enable_quinella=enable_quinella,
            enable_wide=enable_wide,
            use_actual_odds=use_actual_odds,
            kelly_fraction=0.25  # 使わないが互換性のため
        )
        
        self.enable_win = enable_win
        self.enable_place = enable_place
        self.bet_fraction = bet_fraction
        self.max_horses = max_horses
        self.optimized_params = None
        
    def optimize_hyperparameters(self, n_trials: int = 50) -> Dict:
        """Optunaでハイパーパラメータを最適化"""
        self.logger.info("ハイパーパラメータ最適化を開始...")
        
        features = self._get_features(self.train_data)
        target = self.train_data['着順'].values
        
        # 時系列分割で過学習を防ぐ
        tscv = TimeSeriesSplit(n_splits=3)
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1
            }
            
            # 交差検証
            cv_scores = []
            for train_idx, val_idx in tscv.split(features):
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = target[train_idx], target[val_idx]
                
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_val = lgb.Dataset(X_val, y_val)
                
                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_val],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                # 順位相関を評価指標として使用
                val_pred = model.predict(X_val)
                val_ranks = pd.Series(val_pred).rank()
                true_ranks = pd.Series(y_val).rank()
                correlation = val_ranks.corr(true_ranks, method='spearman')
                cv_scores.append(correlation)
            
            return np.mean(cv_scores)
        
        # 最適化実行
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.optimized_params = study.best_params
        self.logger.info(f"最適パラメータ: {self.optimized_params}")
        
        return self.optimized_params
    
    def train_model(self) -> lgb.Booster:
        """最適化されたパラメータでモデルを訓練"""
        self.logger.info("最適化されたモデルを訓練中...")
        
        # パラメータ最適化
        if self.optimized_params is None:
            self.optimize_hyperparameters()
        
        features = self._get_features(self.train_data)
        target = self.train_data['着順'].values
        
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            **self.optimized_params
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def calculate_win_odds_ev(self, probs: Dict[int, Dict], 
                             horse: int,
                             actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """単勝の期待値計算"""
        win_prob = probs[horse]['win_prob']
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            estimated_odds = actual_odds / 100
        else:
            popularity = probs[horse]['popularity']
            if popularity <= 3:
                base_odds = 2 + popularity * 1.5
            elif popularity <= 6:
                base_odds = 5 + popularity * 2.5
            else:
                base_odds = 10 + popularity * 5
            
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = win_prob * estimated_odds
        return expected_value, win_prob, estimated_odds
    
    def calculate_place_odds_ev(self, probs: Dict[int, Dict], 
                               horse: int,
                               actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """複勝の期待値計算"""
        place_prob = probs[horse]['place_prob']
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            estimated_odds = actual_odds / 100
        else:
            popularity = probs[horse]['popularity']
            # 複勝は単勝の約1/3のオッズ
            if popularity <= 3:
                base_odds = 1.1 + popularity * 0.2
            elif popularity <= 6:
                base_odds = 1.5 + popularity * 0.3
            else:
                base_odds = 2.5 + popularity * 0.5
            
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = place_prob * estimated_odds
        return expected_value, place_prob, estimated_odds
    
    def calculate_bet_amount_fixed(self, capital: float, bet_info: Dict) -> float:
        """固定比率でベット額計算"""
        # 資金の固定割合
        bet_amount = capital * self.bet_fraction
        
        # 最低1000円、最大50000円
        bet_amount = max(1000, min(bet_amount, 50000))
        
        # 100円単位に丸める
        return int(bet_amount / 100) * 100
    
    def run_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """最適化されたバックテスト"""
        self.logger.info("最適化されたバックテストを実行中...")
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト
        capital = initial_capital
        all_bets = []
        stats = {
            'win': {'count': 0, 'wins': 0, 'profit': 0},
            'place': {'count': 0, 'wins': 0, 'profit': 0},
            'quinella': {'count': 0, 'wins': 0, 'profit': 0},
            'wide': {'count': 0, 'wins': 0, 'profit': 0}
        }
        
        unique_races = self.test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races[:2000]):
            if i % 200 == 0:
                self.logger.debug(f"Processing race {i}/{min(len(unique_races), 2000)}")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            if len(race_data) < 6:
                continue
            
            # 払戻データを取得
            payout_data = self._get_payout_data(race_data)
            
            # 確率予測
            probs = self.predict_probabilities(model, race_data)
            if not probs:
                continue
            
            # 予測順位でソート
            sorted_horses = sorted(probs.items(), 
                                 key=lambda x: x[1]['predicted_rank'])[:self.max_horses]
            
            race_bets = []
            
            # 単勝（上位3頭）
            if self.enable_win:
                for horse, horse_probs in sorted_horses:
                    actual_odds = self._get_actual_odds(payout_data, '単勝', (horse,))
                    ev, wp, odds = self.calculate_win_odds_ev(probs, horse, actual_odds)
                    
                    if ev >= self.min_expected_value:
                        race_bets.append({
                            'type': '単勝',
                            'selection': (horse,),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds,
                            'actual_odds': actual_odds if actual_odds else odds
                        })
            
            # 複勝（上位3頭）
            if self.enable_place:
                for horse, horse_probs in sorted_horses:
                    actual_odds = self._get_actual_odds(payout_data, '複勝', (horse,))
                    ev, wp, odds = self.calculate_place_odds_ev(probs, horse, actual_odds)
                    
                    if ev >= self.min_expected_value * 0.95:  # 複勝は少し緩く
                        race_bets.append({
                            'type': '複勝',
                            'selection': (horse,),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds,
                            'actual_odds': actual_odds if actual_odds else odds
                        })
            
            # 馬連（上位3頭のBOX）
            if self.enable_quinella and len(sorted_horses) >= 2:
                for i in range(len(sorted_horses)):
                    for j in range(i+1, len(sorted_horses)):
                        h1, h2 = sorted_horses[i][0], sorted_horses[j][0]
                        actual_odds = self._get_actual_odds(
                            payout_data, '馬連', tuple(sorted([h1, h2]))
                        )
                        
                        ev, wp, odds = self.calculate_quinella_ev(probs, h1, h2, actual_odds)
                        
                        if ev >= self.min_expected_value:
                            race_bets.append({
                                'type': '馬連',
                                'selection': tuple(sorted([h1, h2])),
                                'expected_value': ev,
                                'win_probability': wp,
                                'estimated_odds': odds,
                                'actual_odds': actual_odds if actual_odds else odds
                            })
            
            # 期待値順にソート
            race_bets.sort(key=lambda x: x['expected_value'], reverse=True)
            
            # 最大3点まで購入
            for bet in race_bets[:3]:
                bet_amount = self.calculate_bet_amount_fixed(capital, bet)
                
                if bet_amount > capital:
                    continue
                
                # 結果判定
                actual_result = race_data.sort_values('着順')
                is_win, actual_odds = self._check_result_extended(bet, actual_result)
                
                if is_win:
                    profit = bet_amount * actual_odds - bet_amount
                    bet_type_key = self._get_bet_type_key(bet['type'])
                    stats[bet_type_key]['wins'] += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                
                # 統計更新
                bet_type_key = self._get_bet_type_key(bet['type'])
                stats[bet_type_key]['count'] += 1
                stats[bet_type_key]['profit'] += profit
                
                all_bets.append({
                    'race_id': race_id,
                    'bet': bet,
                    'amount': bet_amount,
                    'profit': profit,
                    'capital': capital,
                    'is_win': is_win
                })
                
                if capital <= 0:
                    self.logger.warning("Bankrupt!")
                    break
            
            if capital <= 0:
                break
        
        # 結果集計
        self.results = {
            'trades': all_bets,
            'metrics': self._calculate_metrics(initial_capital, capital, all_bets, stats)
        }
        
        return self.results
    
    def _check_result_extended(self, bet: Dict, result: pd.DataFrame) -> Tuple[bool, float]:
        """拡張された結果チェック（単勝・複勝対応）"""
        bet_type = bet['type']
        selection = bet['selection']
        
        if bet_type == '単勝':
            winner = int(result.iloc[0]['馬番'])
            if winner == selection[0]:
                actual = bet.get('actual_odds', bet['estimated_odds'])
                return True, actual if actual > 100 else actual / 100
                
        elif bet_type == '複勝':
            top3 = [int(result.iloc[i]['馬番']) for i in range(min(3, len(result)))]
            if selection[0] in top3:
                actual = bet.get('actual_odds', bet['estimated_odds'])
                return True, actual if actual > 100 else actual / 100
        
        else:
            # 既存の処理を使用
            return self._check_result(bet, result)
        
        return False, 0
    
    def _get_bet_type_key(self, bet_type: str) -> str:
        """日本語の馬券種別を英語キーに変換"""
        mapping = {
            '単勝': 'win',
            '複勝': 'place',
            '馬連': 'quinella',
            'ワイド': 'wide'
        }
        return mapping.get(bet_type, bet_type)


def run_optimization_loop(n_iterations: int = 5):
    """最適化ループを実行"""
    logger = setup_logger('optimization_loop')
    best_result = None
    best_params = None
    best_return = -float('inf')
    
    for i in range(n_iterations):
        logger.info(f"\n最適化イテレーション {i+1}/{n_iterations}")
        
        # 出力ディレクトリ
        output_dir = Path(f'backtest_optimized_iter{i+1}')
        output_dir.mkdir(exist_ok=True)
        
        # 戦略の初期化
        strategy = OptimizedBettingStrategy(
            min_expected_value=1.05 + i * 0.02,  # 徐々に厳しく
            enable_win=True,
            enable_place=True,
            enable_quinella=True,
            enable_wide=i < 3,  # 後半はワイドを無効化
            enable_trifecta=False,
            use_actual_odds=True,
            bet_fraction=0.02,  # 固定2%
            max_horses=3
        )
        
        # データ読み込み
        strategy.load_data(start_year=2022, end_year=2025, use_payout_data=True)
        
        # データ分割（過学習防止のため年を変える）
        if i % 2 == 0:
            strategy.split_data(train_years=[2022, 2023], test_years=[2024, 2025])
        else:
            strategy.split_data(train_years=[2023, 2024], test_years=[2022, 2025])
        
        # バックテスト実行
        results = strategy.run_backtest(initial_capital=1_000_000)
        
        # 結果を保存
        metrics = results['metrics']
        total_return = metrics['total_return']
        
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'iteration': i + 1,
                'parameters': {
                    'min_expected_value': 1.05 + i * 0.02,
                    'enable_wide': i < 3,
                    'optimized_params': strategy.optimized_params
                },
                'metrics': metrics
            }, f, ensure_ascii=False, indent=2)
        
        # 結果表示
        print(f"\nイテレーション {i+1} 結果:")
        print(f"収益率: {total_return*100:.1f}%")
        print(f"勝率: {metrics['win_rate']*100:.1f}%")
        print(f"総賭け回数: {metrics['total_bets']}")
        
        # 最良結果を更新
        if total_return > best_return:
            best_return = total_return
            best_result = results
            best_params = {
                'iteration': i + 1,
                'min_expected_value': 1.05 + i * 0.02,
                'optimized_params': strategy.optimized_params
            }
    
    # 最終結果
    print("\n" + "="*60)
    print("最適化完了！")
    print("="*60)
    print(f"最良の収益率: {best_return*100:.1f}%")
    print(f"最良のパラメータ: {best_params}")
    
    # 最良結果を保存
    with open('best_optimization_result.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_params,
            'best_return': best_return,
            'best_metrics': best_result['metrics'] if best_result else None
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 最適化ループを実行
    run_optimization_loop(n_iterations=5)