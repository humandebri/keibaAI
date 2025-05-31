#!/usr/bin/env python3
"""
高度な馬券戦略（期待値ベース）
三連単・馬連・ワイドの流し馬券を期待値計算に基づいて購入
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from itertools import combinations, permutations
import json
from datetime import datetime
import os

from .base import BaseStrategy
from ..core.config import config
from ..core.utils import setup_logger
from ..features.efficient_features import EfficientFeatureEngineering


class AdvancedBettingStrategy(BaseStrategy):
    """高度な馬券戦略（期待値ベース）"""
    
    def __init__(self, min_expected_value: float = 1.1, 
                 enable_trifecta: bool = True,
                 enable_quinella: bool = True,
                 enable_wide: bool = True,
                 use_actual_odds: bool = True):
        super().__init__(name="AdvancedBetting")
        self.min_expected_value = min_expected_value
        self.enable_trifecta = enable_trifecta
        self.enable_quinella = enable_quinella
        self.enable_wide = enable_wide
        self.use_actual_odds = use_actual_odds
        self.betting_fraction = 0.01
        self.jra_take_rate = 0.2  # JRA控除率
        
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """追加特徴量の作成（高度な特徴量を含む）"""
        df = df.copy()
        
        # 基本的な特徴量
        df['log_odds'] = np.log1p(df['オッズ'])
        df['odds_rank'] = df.groupby('race_id')['オッズ'].rank()
        df['popularity_rank'] = df.groupby('race_id')['人気'].rank()
        df['relative_odds'] = df.groupby('race_id')['オッズ'].transform(
            lambda x: x / x.min()
        )
        
        # 体重変化率
        df['weight_change_rate'] = df['体重変化'] / df['体重']
        
        # 効率的な特徴量エンジニアリング
        feature_eng = EfficientFeatureEngineering(self.logger)
        
        # 高度な特徴量を追加
        try:
            df = feature_eng.add_all_features_fast(df)
            self.logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        except Exception as e:
            self.logger.warning(f"Failed to add some features: {e}")
            # 基本特徴量のみ追加
            try:
                df = feature_eng.add_basic_features(df)
            except:
                pass
        
        return df
    
    def train_model(self) -> lgb.Booster:
        """着順予測モデルの訓練"""
        self.logger.info("Training advanced ranking model")
        
        features = self._get_features(self.train_data)
        target = self.train_data['着順'].values
        
        # 回帰問題として訓練（着順を直接予測）
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 60,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.5,
            'min_gain_to_split': 0.01
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def predict_probabilities(self, model: lgb.Booster, 
                            race_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """各馬の着順確率を予測"""
        features = self._get_features(race_data)
        if features is None:
            return {}
        
        # 着順予測
        predicted_positions = model.predict(features)
        
        # スコアに変換（低い着順ほど高スコア）
        scores = 1 / (predicted_positions + 0.5)
        total_score = scores.sum()
        
        result = {}
        for i, (_, horse) in enumerate(race_data.iterrows()):
            horse_num = int(horse['馬番'])
            base_prob = scores[i] / total_score
            
            # 予測順位に基づく確率調整
            rank = predicted_positions[i]
            
            result[horse_num] = {
                'win_prob': self._calculate_win_prob(base_prob, rank),
                'place_prob': self._calculate_place_prob(base_prob, rank),
                'show_prob': self._calculate_show_prob(base_prob, rank),
                'predicted_rank': rank,
                'odds': float(horse['オッズ']),
                'popularity': int(horse['人気'])
            }
        
        return result
    
    def _calculate_win_prob(self, base_prob: float, rank: float) -> float:
        """単勝確率の計算"""
        # 予測1着に近いほど高確率
        if rank <= 1.5:
            return min(base_prob * 2.0, 0.4)
        elif rank <= 3:
            return min(base_prob * 1.5, 0.25)
        elif rank <= 5:
            return min(base_prob * 1.0, 0.15)
        else:
            return min(base_prob * 0.5, 0.1)
    
    def _calculate_place_prob(self, base_prob: float, rank: float) -> float:
        """複勝確率の計算（3着以内）"""
        if rank <= 3:
            return min(base_prob * 3.5, 0.6)
        elif rank <= 5:
            return min(base_prob * 2.5, 0.4)
        elif rank <= 8:
            return min(base_prob * 1.5, 0.25)
        else:
            return min(base_prob * 0.8, 0.15)
    
    def _calculate_show_prob(self, base_prob: float, rank: float) -> float:
        """2着確率の計算（馬連用）"""
        if rank <= 2:
            return min(base_prob * 1.8, 0.3)
        elif rank <= 4:
            return min(base_prob * 1.3, 0.2)
        elif rank <= 6:
            return min(base_prob * 0.9, 0.15)
        else:
            return min(base_prob * 0.5, 0.1)
    
    def calculate_trifecta_ev(self, probs: Dict[int, Dict], 
                             h1: int, h2: int, h3: int,
                             actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """三連単の期待値計算（実際のオッズ使用可能）"""
        # 的中確率
        p1 = probs[h1]['win_prob']
        p2 = probs[h2]['show_prob'] * (1 - p1)
        p3 = probs[h3]['place_prob'] * (1 - p1 - p2 * 0.5)
        win_prob = p1 * p2 * p3
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            # 実際のオッズを使用（払戻額/100円）
            estimated_odds = actual_odds / 100
        else:
            # オッズ推定（人気順位から）
            pop_sum = probs[h1]['popularity'] + probs[h2]['popularity'] + probs[h3]['popularity']
            
            if pop_sum <= 10:  # 上位人気
                base_odds = 50 + pop_sum * 10
            elif pop_sum <= 20:  # 中位人気
                base_odds = 150 + pop_sum * 20
            elif pop_sum <= 30:  # 中穴
                base_odds = 500 + pop_sum * 30
            else:  # 大穴
                base_odds = min(1000 + pop_sum * 50, 5000)
            
            # 控除率考慮
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def calculate_quinella_ev(self, probs: Dict[int, Dict], 
                             h1: int, h2: int,
                             actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """馬連の期待値計算（実際のオッズ使用可能）"""
        # どちらかが1着、もう一方が2着
        p12 = probs[h1]['win_prob'] * probs[h2]['show_prob']
        p21 = probs[h2]['win_prob'] * probs[h1]['show_prob']
        win_prob = p12 + p21
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            # 実際のオッズを使用（払戻額/100円）
            estimated_odds = actual_odds / 100
        else:
            # オッズ推定
            pop_avg = (probs[h1]['popularity'] + probs[h2]['popularity']) / 2
            
            if pop_avg <= 3:
                base_odds = 5 + pop_avg * 3
            elif pop_avg <= 6:
                base_odds = 15 + pop_avg * 5
            elif pop_avg <= 10:
                base_odds = 40 + pop_avg * 8
            else:
                base_odds = min(100 + pop_avg * 15, 500)
            
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def calculate_wide_ev(self, probs: Dict[int, Dict], 
                         h1: int, h2: int,
                         actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """ワイドの期待値計算（実際のオッズ使用可能）"""
        # 両方が3着以内
        win_prob = probs[h1]['place_prob'] * probs[h2]['place_prob'] * 0.6
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            # 実際のオッズを使用（払戻額/100円）
            estimated_odds = actual_odds / 100
        else:
            # オッズ推定（馬連の約40%）
            pop_avg = (probs[h1]['popularity'] + probs[h2]['popularity']) / 2
            
            if pop_avg <= 3:
                base_odds = 2 + pop_avg * 1
            elif pop_avg <= 6:
                base_odds = 5 + pop_avg * 2
            elif pop_avg <= 10:
                base_odds = 15 + pop_avg * 3
            else:
                base_odds = min(40 + pop_avg * 5, 200)
            
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = win_prob * estimated_odds
        
        return expected_value, win_prob, estimated_odds
    
    def select_bets(self, race_data: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """使用しない（独自実装）"""
        return []
    
    def calculate_bet_amount(self, capital: float, bet_info: Dict) -> float:
        """Kelly基準によるベット額計算"""
        ev = bet_info.get('expected_value', 1.0)
        if ev < self.min_expected_value:
            return 0
        
        # Kelly比率
        win_prob = bet_info.get('win_probability', 0.01)
        odds = bet_info.get('estimated_odds', 10)
        
        if odds <= 1:
            return 0
        
        kelly = (win_prob * odds - 1) / (odds - 1)
        kelly = max(0, min(kelly, 0.05))  # 最大5%
        
        # ベット種別による調整
        bet_type = bet_info.get('type', '')
        if '三連単' in bet_type:
            kelly *= 0.3  # リスク高いので減額
        elif '馬連' in bet_type:
            kelly *= 0.5
        elif 'ワイド' in bet_type:
            kelly *= 0.7
        
        return capital * self.betting_fraction * kelly
    
    def run_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """高度なバックテストの実行"""
        self.logger.info("Running advanced betting backtest with actual odds")
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト
        capital = initial_capital
        all_bets = []
        stats = {
            'trifecta': {'count': 0, 'wins': 0, 'profit': 0},
            'quinella': {'count': 0, 'wins': 0, 'profit': 0},
            'wide': {'count': 0, 'wins': 0, 'profit': 0}
        }
        
        unique_races = self.test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races[:2000]):  # 最初の2000レース
            if i % 200 == 0:
                self.logger.debug(f"Processing race {i}/{min(len(unique_races), 2000)}")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            if len(race_data) < 8:  # 出走頭数が少ないレースはスキップ
                continue
            
            # 払戻データを取得
            payout_data = self._get_payout_data(race_data)
            
            # 確率予測
            probs = self.predict_probabilities(model, race_data)
            if not probs:
                continue
            
            # 予測順位でソート
            sorted_horses = sorted(probs.items(), 
                                 key=lambda x: x[1]['predicted_rank'])
            
            race_bets = []
            
            # 三連単流し（上位3頭から）
            if self.enable_trifecta and len(sorted_horses) >= 6:
                for i in range(min(3, len(sorted_horses))):
                    axis = sorted_horses[i][0]
                    
                    # 相手選択（4-8位）
                    partners = [h[0] for j, h in enumerate(sorted_horses) 
                               if j != i and 1 <= j <= 7]
                    
                    for p2 in partners[:4]:
                        for p3 in partners:
                            if p3 != p2:
                                # 実際のオッズを取得
                                actual_odds = self._get_actual_odds(
                                    payout_data, 'trifecta', (axis, p2, p3)
                                )
                                
                                ev, wp, odds = self.calculate_trifecta_ev(
                                    probs, axis, p2, p3, actual_odds
                                )
                                
                                if ev >= self.min_expected_value:
                                    race_bets.append({
                                        'type': '三連単',
                                        'selection': (axis, p2, p3),
                                        'expected_value': ev,
                                        'win_probability': wp,
                                        'estimated_odds': odds
                                    })
            
            # 馬連流し（上位2頭）
            if self.enable_quinella and len(sorted_horses) >= 4:
                axis = sorted_horses[0][0]
                for i in range(1, min(6, len(sorted_horses))):
                    partner = sorted_horses[i][0]
                    # 実際のオッズを取得
                    actual_odds = self._get_actual_odds(
                        payout_data, 'quinella', tuple(sorted([axis, partner]))
                    )
                    
                    ev, wp, odds = self.calculate_quinella_ev(probs, axis, partner, actual_odds)
                    
                    if ev >= self.min_expected_value:
                        race_bets.append({
                            'type': '馬連',
                            'selection': tuple(sorted([axis, partner])),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds
                        })
            
            # ワイド（上位5頭から）
            if self.enable_wide and len(sorted_horses) >= 5:
                top5 = [h[0] for h in sorted_horses[:5]]
                for h1, h2 in combinations(top5, 2):
                    # 実際のオッズを取得
                    actual_odds = self._get_actual_odds(
                        payout_data, 'wide', tuple(sorted([h1, h2]))
                    )
                    
                    ev, wp, odds = self.calculate_wide_ev(probs, h1, h2, actual_odds)
                    
                    if ev >= self.min_expected_value * 0.9:  # ワイドは少し緩く
                        race_bets.append({
                            'type': 'ワイド',
                            'selection': tuple(sorted([h1, h2])),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds
                        })
            
            # 期待値の高い順に最大5点まで購入
            race_bets.sort(key=lambda x: x['expected_value'], reverse=True)
            
            for bet in race_bets[:5]:
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                if bet_amount <= 0 or bet_amount > capital * 0.05:
                    continue
                
                # 結果判定
                actual_result = race_data.sort_values('着順')
                is_win, actual_odds = self._check_result(bet, actual_result)
                
                if is_win:
                    profit = bet_amount * actual_odds - bet_amount
                    stats[bet['type'].replace('三連単', 'trifecta')
                                    .replace('馬連', 'quinella')
                                    .replace('ワイド', 'wide')]['wins'] += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                
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
    
    def _check_result(self, bet: Dict, result: pd.DataFrame) -> Tuple[bool, float]:
        """実際の結果をチェック"""
        bet_type = bet['type']
        selection = bet['selection']
        
        if bet_type == '三連単':
            if (result.iloc[0]['馬番'] == selection[0] and
                result.iloc[1]['馬番'] == selection[1] and
                result.iloc[2]['馬番'] == selection[2]):
                # 的中！実際のオッズがあればそれを使用
                return True, bet['estimated_odds']
        
        elif bet_type == '馬連':
            top2 = set(result.iloc[:2]['馬番'].values)
            if set(selection) == top2:
                return True, bet['estimated_odds']
        
        elif bet_type == 'ワイド':
            top3 = set(result.iloc[:3]['馬番'].values)
            if set(selection).issubset(top3):
                return True, bet['estimated_odds']
        
        return False, 0
    
    def _get_payout_data(self, race_data: pd.DataFrame) -> Dict:
        """レースの払戻データを取得"""
        # 払戻データがJSON形式で保存されている場合
        if '払戻データ' in race_data.columns:
            try:
                # 最初の行から払戻データを取得（全行同じデータ）
                payout_json = race_data.iloc[0]['払戻データ']
                if pd.notna(payout_json) and payout_json:
                    return json.loads(payout_json)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.debug(f"Failed to parse payout data: {e}")
        
        return {}
    
    def _get_actual_odds(self, payout_data: Dict, bet_type: str, 
                        selection: Tuple) -> Optional[float]:
        """実際のオッズを取得"""
        if not payout_data or bet_type not in payout_data:
            return None
        
        bet_data = payout_data[bet_type]
        
        # 三連単の場合
        if bet_type == 'trifecta':
            key = f"{selection[0]}-{selection[1]}-{selection[2]}"
            return bet_data.get(key)
        
        # 馬連・ワイドの場合
        elif bet_type in ['quinella', 'wide']:
            # 両方の順序を試す
            key1 = f"{selection[0]}-{selection[1]}"
            key2 = f"{selection[1]}-{selection[0]}"
            
            odds = bet_data.get(key1) or bet_data.get(key2)
            
            # ワイドは複数の配当がある場合がある
            if bet_type == 'wide' and isinstance(bet_data, dict):
                # 該当する組み合わせを探す
                for k, v in bet_data.items():
                    horses = set(k.split('-'))
                    if horses == set(str(h) for h in selection):
                        return v
            
            return odds
        
        return None
    
    def _calculate_metrics(self, initial_capital: float, final_capital: float,
                          all_bets: List[Dict], stats: Dict) -> Dict:
        """詳細な指標を計算"""
        total_bets = len(all_bets)
        winning_bets = [b for b in all_bets if b['is_win']]
        
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': (final_capital - initial_capital) / initial_capital,
            'total_bets': total_bets,
            'total_wins': len(winning_bets),
            'win_rate': len(winning_bets) / total_bets if total_bets > 0 else 0,
            'avg_expected_value': np.mean([b['bet']['expected_value'] for b in all_bets]) if all_bets else 0,
            'by_type': {}
        }
        
        # 馬券種別の統計
        for bet_type, stat in stats.items():
            if stat['count'] > 0:
                metrics['by_type'][bet_type] = {
                    'count': stat['count'],
                    'wins': stat['wins'],
                    'win_rate': stat['wins'] / stat['count'],
                    'profit': stat['profit'],
                    'roi': stat['profit'] / (initial_capital * 0.01 * stat['count'])
                }
        
        return metrics
    
    def _calculate_profit(self, bet: Dict, bet_amount: float) -> float:
        """BaseStrategy用のダミー実装"""
        return 0