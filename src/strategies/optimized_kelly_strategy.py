#!/usr/bin/env python3
"""
最適化Kelly基準戦略 - 年間収益率15-20%を目指す高度な資金管理
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations, permutations
import json
from datetime import datetime
import os
from scipy.optimize import minimize_scalar

from .base import BaseStrategy
from ..core.config import config
from ..core.utils import setup_logger


class OptimizedKellyStrategy(BaseStrategy):
    """最適化Kelly基準戦略"""
    
    def __init__(self, 
                 min_expected_value: float = 1.05,
                 max_kelly_fraction: float = 0.15,  # Kelly基準の最大15%
                 risk_adjustment: float = 0.7,      # リスク調整係数
                 bankroll_protection: float = 0.8,  # 破産防止（残り80%以下では保守的に）
                 diversification_limit: int = 8,    # 同時ベット数制限
                 win_rate_threshold: float = 0.15,  # 最低勝率閾値
                 volatility_adjustment: bool = True, # ボラティリティ調整
                 enable_compound_growth: bool = True # 複利効果活用
                 ):
        super().__init__(name="OptimizedKelly")
        
        # Kelly基準パラメータ
        self.min_expected_value = min_expected_value
        self.max_kelly_fraction = max_kelly_fraction
        self.risk_adjustment = risk_adjustment
        self.bankroll_protection = bankroll_protection
        self.diversification_limit = diversification_limit
        self.win_rate_threshold = win_rate_threshold
        self.volatility_adjustment = volatility_adjustment
        self.enable_compound_growth = enable_compound_growth
        
        # ベットタイプ設定
        self.enable_trifecta = True
        self.enable_quinella = True
        self.enable_wide = True
        self.use_actual_odds = True
        
        # JRA控除率
        self.jra_take_rate = 0.2
        
        # 動的パラメータ
        self.recent_performance = []
        self.volatility_history = []
        self.current_drawdown = 0.0
        
        self.logger = setup_logger("OptimizedKelly")
        
    def calculate_optimal_kelly_fraction(self, win_prob: float, odds: float, 
                                       recent_volatility: float = 0.0) -> float:
        """最適化されたKelly基準の計算"""
        
        # 基本Kelly計算: f = (bp - q) / b
        # b = odds - 1 (利益倍率)
        # p = 勝率, q = 1 - p (負け率)
        
        if odds <= 1.0 or win_prob <= 0:
            return 0.0
            
        b = odds - 1
        p = win_prob
        q = 1 - p
        
        # 基本Kelly分数
        kelly_fraction = (b * p - q) / b
        
        if kelly_fraction <= 0:
            return 0.0
        
        # 1. リスク調整
        kelly_fraction *= self.risk_adjustment
        
        # 2. ボラティリティ調整
        if self.volatility_adjustment and recent_volatility > 0:
            volatility_factor = max(0.5, 1.0 - recent_volatility * 0.5)
            kelly_fraction *= volatility_factor
        
        # 3. ドローダウン調整
        if self.current_drawdown > 0.1:  # 10%以上のドローダウン
            drawdown_factor = max(0.3, 1.0 - self.current_drawdown * 2)
            kelly_fraction *= drawdown_factor
        
        # 4. 勝率による調整
        if win_prob < self.win_rate_threshold:
            win_rate_factor = win_prob / self.win_rate_threshold
            kelly_fraction *= win_rate_factor
        
        # 5. 最大値制限
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        return kelly_fraction
    
    def calculate_diversified_kelly(self, bet_opportunities: List[Dict]) -> List[Dict]:
        """分散投資を考慮したKelly最適化"""
        
        if not bet_opportunities:
            return []
        
        # 期待値でソート
        bet_opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # 上位N個まで選択
        selected_bets = bet_opportunities[:self.diversification_limit]
        
        # 相関を考慮した分散調整
        correlation_matrix = self._estimate_bet_correlations(selected_bets)
        
        optimized_bets = []
        total_kelly_allocation = 0.0
        
        for i, bet in enumerate(selected_bets):
            # 個別Kelly計算
            individual_kelly = self.calculate_optimal_kelly_fraction(
                bet['win_probability'], 
                bet['estimated_odds'],
                self._get_recent_volatility()
            )
            
            # 相関調整
            if i > 0:
                correlation_adjustment = self._calculate_correlation_adjustment(
                    correlation_matrix, i
                )
                individual_kelly *= correlation_adjustment
            
            # 全体の配分制限
            remaining_allocation = self.max_kelly_fraction - total_kelly_allocation
            adjusted_kelly = min(individual_kelly, remaining_allocation * 0.8)
            
            if adjusted_kelly > 0.001:  # 最小閾値
                bet['kelly_fraction'] = adjusted_kelly
                optimized_bets.append(bet)
                total_kelly_allocation += adjusted_kelly
            
            if total_kelly_allocation >= self.max_kelly_fraction * 0.9:
                break
        
        return optimized_bets
    
    def _estimate_bet_correlations(self, bets: List[Dict]) -> np.ndarray:
        """ベット間の相関を推定"""
        n = len(bets)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                # 同じレースのベットは高相関
                if bets[i].get('race_id') == bets[j].get('race_id'):
                    correlation = 0.7
                # 同じベットタイプは中相関
                elif bets[i].get('type') == bets[j].get('type'):
                    correlation = 0.3
                # その他は低相関
                else:
                    correlation = 0.1
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _calculate_correlation_adjustment(self, correlation_matrix: np.ndarray, 
                                        bet_index: int) -> float:
        """相関を考慮した調整係数"""
        if bet_index == 0:
            return 1.0
        
        # 前のベットとの相関を考慮
        max_correlation = np.max(correlation_matrix[bet_index, :bet_index])
        adjustment = 1.0 - max_correlation * 0.5
        
        return max(0.3, adjustment)
    
    def _get_recent_volatility(self) -> float:
        """最近のボラティリティを計算"""
        if len(self.volatility_history) < 5:
            return 0.0
        
        recent_returns = self.volatility_history[-20:]  # 最近20回
        return np.std(recent_returns) if recent_returns else 0.0
    
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統一特徴量エンジンを使用"""
        from ..features.unified_features import UnifiedFeatureEngine
        
        engine = UnifiedFeatureEngine()
        enhanced_df = engine.build_all_features(df)
        available_features = engine.get_feature_columns(enhanced_df)
        
        self.logger.info(f"Unified features applied: {len(available_features)} features available")
        
        return enhanced_df
    
    def train_model(self) -> lgb.Booster:
        """高度な着順予測モデル"""
        self.logger.info("Training optimized ranking model with advanced features")
        
        # 統一特徴量エンジンで特徴量を構築
        self.train_data = self.create_additional_features(self.train_data)
        
        # 利用可能な特徴量を取得
        from ..features.unified_features import UnifiedFeatureEngine
        engine = UnifiedFeatureEngine()
        engine.build_all_features(self.train_data.head(10))  # サンプルでビルド
        available_features = engine.get_feature_columns(self.train_data)
        
        if not available_features:
            raise ValueError("No features available after unified feature engineering")
        
        self.feature_cols = available_features
        features = self.train_data[available_features].values
        target = self.train_data['着順'].values
        
        # LightGBM最適化パラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 80,           # 増加
            'learning_rate': 0.02,      # 減少（精度向上）
            'feature_fraction': 0.8,    # 増加
            'bagging_fraction': 0.8,    # 増加
            'bagging_freq': 3,
            'lambda_l1': 0.05,          # 減少
            'lambda_l2': 0.3,           # 減少
            'min_gain_to_split': 0.005, # 減少
            'min_child_samples': 15,    # 増加
            'verbose': -1,
            'force_row_wise': True
        }
        
        lgb_train = lgb.Dataset(features, target)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,        # 増加
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.logger.info(f"Model trained with {len(available_features)} features")
        return model
    
    def predict_probabilities(self, model: lgb.Booster, 
                            race_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """改良された確率予測"""
        
        # 特徴量構築
        enhanced_race_data = self.create_additional_features(race_data)
        
        # 予測時に利用可能な特徴量を取得
        from ..features.unified_features import UnifiedFeatureEngine
        engine = UnifiedFeatureEngine()
        # まず特徴量を構築して feature_names を設定
        engine.build_all_features(enhanced_race_data.head(1))  # サンプルで構築
        available_features = engine.get_feature_columns(enhanced_race_data)
        
        # 訓練時と予測時で共通する特徴量のみ使用
        common_features = [col for col in self.feature_cols if col in available_features]
        
        if not common_features:
            self.logger.warning(f"No common features available for prediction")
            self.logger.warning(f"Training features count: {len(self.feature_cols)}")
            self.logger.warning(f"Prediction features count: {len(available_features)}")
            self.logger.warning(f"Training features sample: {self.feature_cols[:10]}")
            self.logger.warning(f"Prediction features sample: {available_features[:10]}")
            return {}
        
        # 足りない特徴量を0で埋める
        missing_features = [col for col in self.feature_cols if col not in enhanced_race_data.columns]
        for col in missing_features:
            enhanced_race_data[col] = 0.0
            self.logger.warning(f"Missing feature filled with 0: {col}")
        
        features = enhanced_race_data[self.feature_cols].values
        predicted_positions = model.predict(features)
        
        # 確率計算の改良
        # 1. ランキングスコアに変換
        scores = 1 / (predicted_positions + 0.1)  # 0.1を追加してゼロ除算防止
        
        # 2. 統計的調整
        scores = np.power(scores, 1.2)  # 上位への重み増加
        total_score = scores.sum()
        
        result = {}
        for i, (_, horse) in enumerate(enhanced_race_data.iterrows()):
            horse_num = int(horse['馬番'])
            base_prob = scores[i] / total_score
            rank = predicted_positions[i]
            
            # オッズ取得
            odds_value = pd.to_numeric(horse['オッズ'], errors='coerce')
            if pd.isna(odds_value):
                odds_value = 99.9
                
            # 改良された確率計算
            result[horse_num] = {
                'win_prob': self._enhanced_win_prob(base_prob, rank, odds_value),
                'place_prob': self._enhanced_place_prob(base_prob, rank, odds_value),
                'show_prob': self._enhanced_show_prob(base_prob, rank, odds_value),
                'predicted_rank': rank,
                'odds': float(odds_value),
                'popularity': int(horse['人気']) if pd.notna(horse['人気']) else 99
            }
        
        return result
    
    def _enhanced_win_prob(self, base_prob: float, rank: float, odds: float) -> float:
        """強化された単勝確率計算"""
        
        # 基本確率
        if rank <= 1.2:
            prob = base_prob * 2.5
        elif rank <= 2.5:
            prob = base_prob * 1.8
        elif rank <= 4.0:
            prob = base_prob * 1.2
        else:
            prob = base_prob * 0.6
        
        # オッズによる調整（市場効率性考慮）
        implied_prob = 1.0 / odds if odds > 0 else 0.01
        market_adjustment = 0.7  # 市場価格への信頼度
        
        # 重み付き平均
        adjusted_prob = prob * (1 - market_adjustment) + implied_prob * market_adjustment
        
        return min(adjusted_prob, 0.5)  # 上限50%
    
    def _enhanced_place_prob(self, base_prob: float, rank: float, odds: float) -> float:
        """強化された複勝確率計算"""
        if rank <= 2.5:
            prob = base_prob * 4.0
        elif rank <= 5.0:
            prob = base_prob * 3.0
        elif rank <= 8.0:
            prob = base_prob * 2.0
        else:
            prob = base_prob * 1.0
        
        return min(prob, 0.7)
    
    def _enhanced_show_prob(self, base_prob: float, rank: float, odds: float) -> float:
        """強化された連対確率計算"""
        if rank <= 1.8:
            prob = base_prob * 2.2
        elif rank <= 3.5:
            prob = base_prob * 1.6
        elif rank <= 6.0:
            prob = base_prob * 1.1
        else:
            prob = base_prob * 0.7
        
        return min(prob, 0.4)
    
    def calculate_bet_amount(self, capital: float, bet_info: Dict) -> float:
        """最適化Kelly基準によるベット額計算"""
        
        kelly_fraction = bet_info.get('kelly_fraction', 0.0)
        
        if kelly_fraction <= 0:
            return 0
        
        # 資金保護機能
        if capital <= self.initial_capital * self.bankroll_protection:
            kelly_fraction *= 0.5  # 保守的モード
        
        # 複利効果を活用
        if self.enable_compound_growth:
            base_amount = capital * kelly_fraction
        else:
            base_amount = self.initial_capital * kelly_fraction
        
        # ベット種別による微調整
        bet_type = bet_info.get('type', '')
        if '三連単' in bet_type:
            base_amount *= 0.8
        elif '馬連' in bet_type:
            base_amount *= 0.9
        
        # 最小・最大制限
        min_bet = 100
        max_bet = min(capital * 0.05, 100000)  # 資金の5%または10万円
        
        bet_amount = max(min_bet, min(base_amount, max_bet))
        
        # 100円単位に調整
        return int(bet_amount / 100) * 100
    
    def run_backtest(self, data: pd.DataFrame, train_years: List[int], 
                     test_years: List[int], feature_cols: List[str],
                     initial_capital: float = 1_000_000) -> Dict:
        """最適化Kelly戦略のバックテスト"""
        
        self.logger.info("Running Optimized Kelly Strategy Backtest")
        self.initial_capital = initial_capital
        
        # データ設定
        self.data = data
        self.train_data = data[data['year'].isin(train_years)]
        self.test_data = data[data['year'].isin(test_years)]
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト実行
        capital = initial_capital
        all_bets = []
        performance_history = []
        
        stats = {
            'trifecta': {'count': 0, 'wins': 0, 'profit': 0},
            'quinella': {'count': 0, 'wins': 0, 'profit': 0},
            'wide': {'count': 0, 'wins': 0, 'profit': 0}
        }
        
        unique_races = self.test_data['race_id'].unique()
        self.logger.info(f"Processing {len(unique_races)} races with Optimized Kelly")
        
        for i, race_id in enumerate(unique_races[:150]):  # 150レースでテスト
            
            if i % 25 == 0:
                self.logger.info(f"Processing race {i+1}/{min(len(unique_races), 150)}")
                self.logger.info(f"Current capital: ¥{capital:,.0f} ({(capital/initial_capital-1)*100:+.1f}%)")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            if len(race_data) < 8:
                continue
            
            # 確率予測
            probs = self.predict_probabilities(model, race_data)
            if not probs:
                continue
            
            # ベット機会生成
            bet_opportunities = self._generate_bet_opportunities(probs, race_data)
            
            # Kelly最適化
            optimized_bets = self.calculate_diversified_kelly(bet_opportunities)
            
            # 実際の結果を取得
            actual_result = race_data.sort_values('着順')
            
            race_profit = 0
            for bet in optimized_bets:
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                if bet_amount <= 0:
                    continue
                
                # 結果判定
                is_win, actual_odds = self._check_result(bet, actual_result)
                
                if is_win:
                    profit = bet_amount * actual_odds - bet_amount
                    stats[bet['type'].replace('三連単', 'trifecta')
                                    .replace('馬連', 'quinella')
                                    .replace('ワイド', 'wide')]['wins'] += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                race_profit += profit
                
                # 統計更新
                bet_type_key = bet['type'].replace('三連単', 'trifecta')\
                                         .replace('馬連', 'quinella')\
                                         .replace('ワイド', 'wide')
                stats[bet_type_key]['count'] += 1
                stats[bet_type_key]['profit'] += profit
                
                all_bets.append({
                    'race_id': race_id,
                    'bet': bet,
                    'amount': bet_amount,
                    'profit': profit,
                    'capital': capital,
                    'is_win': is_win,
                    'kelly_fraction': bet.get('kelly_fraction', 0)
                })
                
                if capital <= 0:
                    self.logger.warning("Bankrupt!")
                    break
            
            # パフォーマンス記録
            race_return = race_profit / capital if capital > 0 else -1.0
            performance_history.append(race_return)
            self.volatility_history.append(race_return)
            
            # ドローダウン更新
            peak_capital = max([b['capital'] for b in all_bets] + [initial_capital])
            self.current_drawdown = max(0, (peak_capital - capital) / peak_capital)
            
            if capital <= 0:
                break
        
        # 結果計算
        self.results = {
            'trades': all_bets,
            'metrics': self._calculate_advanced_metrics(initial_capital, capital, all_bets, stats, performance_history)
        }
        
        return self.results
    
    def _generate_bet_opportunities(self, probs: Dict, race_data: pd.DataFrame) -> List[Dict]:
        """ベット機会の生成"""
        opportunities = []
        
        # 予測順位でソート
        sorted_horses = sorted(probs.items(), key=lambda x: x[1]['predicted_rank'])
        
        # 三連単（上位3頭から）
        if self.enable_trifecta and len(sorted_horses) >= 6:
            for axis_idx in range(min(3, len(sorted_horses))):
                axis = sorted_horses[axis_idx][0]
                partners = [h[0] for j, h in enumerate(sorted_horses) 
                           if j != axis_idx and 1 <= j <= 7]
                
                for p2 in partners[:3]:
                    for p3 in partners:
                        if p3 != p2:
                            ev, wp, odds = self.calculate_trifecta_ev(probs, axis, p2, p3)
                            if ev >= self.min_expected_value and wp >= self.win_rate_threshold:
                                opportunities.append({
                                    'type': '三連単',
                                    'selection': (axis, p2, p3),
                                    'expected_value': ev,
                                    'win_probability': wp,
                                    'estimated_odds': odds,
                                    'race_id': race_data.iloc[0]['race_id']
                                })
        
        # 馬連とワイドも同様に生成...
        self._add_quinella_opportunities(opportunities, probs, sorted_horses, race_data)
        self._add_wide_opportunities(opportunities, probs, sorted_horses, race_data)
        
        return opportunities
    
    def _add_quinella_opportunities(self, opportunities: List, probs: Dict, 
                                   sorted_horses: List, race_data: pd.DataFrame):
        """馬連機会を追加"""
        if not self.enable_quinella or len(sorted_horses) < 4:
            return
        
        axis = sorted_horses[0][0]
        for i in range(1, min(5, len(sorted_horses))):
            partner = sorted_horses[i][0]
            ev, wp, odds = self.calculate_quinella_ev(probs, axis, partner)
            
            if ev >= self.min_expected_value and wp >= self.win_rate_threshold:
                opportunities.append({
                    'type': '馬連',
                    'selection': tuple(sorted([axis, partner])),
                    'expected_value': ev,
                    'win_probability': wp,
                    'estimated_odds': odds,
                    'race_id': race_data.iloc[0]['race_id']
                })
    
    def _add_wide_opportunities(self, opportunities: List, probs: Dict,
                               sorted_horses: List, race_data: pd.DataFrame):
        """ワイド機会を追加"""
        if not self.enable_wide or len(sorted_horses) < 5:
            return
        
        top5 = [h[0] for h in sorted_horses[:5]]
        for h1, h2 in combinations(top5, 2):
            ev, wp, odds = self.calculate_wide_ev(probs, h1, h2)
            
            if ev >= self.min_expected_value * 0.95:  # ワイドは少し緩く
                opportunities.append({
                    'type': 'ワイド',
                    'selection': tuple(sorted([h1, h2])),
                    'expected_value': ev,
                    'win_probability': wp,
                    'estimated_odds': odds,
                    'race_id': race_data.iloc[0]['race_id']
                })
    
    # 期待値計算メソッドは既存のものを使用
    def calculate_trifecta_ev(self, probs: Dict[int, Dict], h1: int, h2: int, h3: int) -> Tuple[float, float, float]:
        """三連単期待値計算（既存実装を使用）"""
        p1 = probs[h1]['win_prob']
        p2 = probs[h2]['show_prob'] * (1 - p1)
        p3 = probs[h3]['place_prob'] * (1 - p1 - p2 * 0.5)
        win_prob = p1 * p2 * p3
        
        # オッズ推定
        avg_pop = (probs[h1]['popularity'] + probs[h2]['popularity'] + probs[h3]['popularity']) / 3
        if avg_pop <= 6:
            base_odds = 30 + avg_pop * 15
        elif avg_pop <= 12:
            base_odds = 120 + avg_pop * 25
        else:
            base_odds = min(400 + avg_pop * 40, 2000)
        
        estimated_odds = base_odds * (1 - self.jra_take_rate)
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def calculate_quinella_ev(self, probs: Dict[int, Dict], h1: int, h2: int) -> Tuple[float, float, float]:
        """馬連期待値計算"""
        p12 = probs[h1]['win_prob'] * probs[h2]['show_prob']
        p21 = probs[h2]['win_prob'] * probs[h1]['show_prob']
        win_prob = p12 + p21
        
        avg_pop = (probs[h1]['popularity'] + probs[h2]['popularity']) / 2
        if avg_pop <= 4:
            base_odds = 8 + avg_pop * 4
        elif avg_pop <= 8:
            base_odds = 25 + avg_pop * 8
        else:
            base_odds = min(80 + avg_pop * 12, 300)
        
        estimated_odds = base_odds * (1 - self.jra_take_rate)
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def calculate_wide_ev(self, probs: Dict[int, Dict], h1: int, h2: int) -> Tuple[float, float, float]:
        """ワイド期待値計算"""
        win_prob = probs[h1]['place_prob'] * probs[h2]['place_prob'] * 0.65
        
        avg_pop = (probs[h1]['popularity'] + probs[h2]['popularity']) / 2
        if avg_pop <= 4:
            base_odds = 3 + avg_pop * 2
        elif avg_pop <= 8:
            base_odds = 12 + avg_pop * 4
        else:
            base_odds = min(30 + avg_pop * 6, 120)
        
        estimated_odds = base_odds * (1 - self.jra_take_rate)
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def _check_result(self, bet: Dict, result: pd.DataFrame) -> Tuple[bool, float]:
        """結果チェック（既存実装）"""
        bet_type = bet['type']
        selection = bet['selection']
        
        def to_int(x):
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return None
        
        if bet_type == '三連単':
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(3, len(result)))]
            if (len(result_horses) >= 3 and 
                result_horses[0] == selection[0] and
                result_horses[1] == selection[1] and
                result_horses[2] == selection[2]):
                return True, bet['estimated_odds']
        
        elif bet_type == '馬連':
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(2, len(result)))]
            result_horses = [h for h in result_horses if h is not None]
            if len(result_horses) >= 2:
                top2 = set(result_horses[:2])
                if set(selection) == top2:
                    return True, bet['estimated_odds']
        
        elif bet_type == 'ワイド':
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(3, len(result)))]
            result_horses = [h for h in result_horses if h is not None]
            if len(result_horses) >= 3:
                top3 = set(result_horses[:3])
                if set(selection).issubset(top3):
                    return True, bet['estimated_odds']
        
        return False, 0
    
    def _calculate_advanced_metrics(self, initial_capital: float, final_capital: float,
                                   all_bets: List[Dict], stats: Dict, 
                                   performance_history: List[float]) -> Dict:
        """高度な指標計算"""
        
        total_bets = len(all_bets)
        winning_bets = [b for b in all_bets if b['is_win']]
        
        if not all_bets:
            return {}
        
        # 基本指標
        total_return = (final_capital - initial_capital) / initial_capital
        
        # リスク調整済みリターン
        if performance_history:
            volatility = np.std(performance_history) * np.sqrt(252)  # 年率換算
            sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Kelly効率性
        kelly_fractions = [b.get('kelly_fraction', 0) for b in all_bets]
        avg_kelly = np.mean(kelly_fractions) if kelly_fractions else 0
        
        # ドローダウン
        capital_history = [initial_capital] + [b['capital'] for b in all_bets]
        peak = initial_capital
        max_drawdown = 0
        
        for capital in capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar比率 (年間リターン / 最大ドローダウン)
        calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0
        
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': total_return,  # 短期テストのため
            'total_bets': total_bets,
            'total_wins': len(winning_bets),
            'win_rate': len(winning_bets) / total_bets if total_bets > 0 else 0,
            'avg_expected_value': np.mean([b['bet']['expected_value'] for b in all_bets]),
            'avg_kelly_fraction': avg_kelly,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': abs(sum([b['profit'] for b in winning_bets]) / 
                               sum([b['profit'] for b in all_bets if not b['is_win']])) if any(not b['is_win'] for b in all_bets) else 0,
            'by_type': {}
        }
        
        # タイプ別統計
        for bet_type, stat in stats.items():
            if stat['count'] > 0:
                metrics['by_type'][bet_type] = {
                    'count': stat['count'],
                    'wins': stat['wins'],
                    'win_rate': stat['wins'] / stat['count'],
                    'profit': stat['profit'],
                    'roi': stat['profit'] / (initial_capital * 0.01 * stat['count']) if stat['count'] > 0 else 0
                }
        
        return metrics
    
    def select_bets(self, race_data: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """BaseStrategyインターフェース（使用されない）"""
        return []
    
    def _calculate_profit(self, bet: Dict, bet_amount: float) -> float:
        """BaseStrategyインターフェース（使用されない）"""
        return 0