#!/usr/bin/env python3
"""
高度な馬券戦略（期待値ベース）
三連単・馬連・ワイドの流し馬券を期待値計算に基づいて購入
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
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
                 use_actual_odds: bool = True,
                 kelly_fraction: float = 0.25):
        super().__init__(name="AdvancedBetting")
        self.min_expected_value = min_expected_value
        self.enable_trifecta = enable_trifecta
        self.enable_quinella = enable_quinella
        self.enable_wide = enable_wide
        self.use_actual_odds = use_actual_odds
        self.kelly_fraction = kelly_fraction  # ケリー基準の割合（デフォルト25%）
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
        """改良されたベット額計算（Kelly基準＋固定比率のハイブリッド）"""
        ev = bet_info.get('expected_value', 1.0)
        if ev < self.min_expected_value:
            return 0
        
        # 方式1: Kelly基準ベース
        win_prob = bet_info.get('win_probability', 0.01)
        odds = bet_info.get('estimated_odds', 10)
        
        if odds <= 1:
            return 0
        
        # フルケリー計算: f = (p * b - q) / b
        b = odds - 1
        kelly_full = (win_prob * b - (1 - win_prob)) / b
        
        # Kelly基準の適用（より積極的に）
        kelly = kelly_full * self.kelly_fraction * 2  # 2倍にして積極化
        
        # Kelly基準が負の場合は0にする
        kelly = max(0, kelly)
        
        # 方式2: 期待値ベースの固定比率
        # 期待値に応じた資本比率（EV1.2なら2%、EV1.5なら5%など）
        ev_based_ratio = min((ev - 1.0) * 0.2, 0.05)  # 最大5%
        
        # 両方式の大きい方を採用（より積極的な方）
        bet_ratio = max(kelly, ev_based_ratio, 0.005)  # 最低0.5%は賭ける
        
        # ベット種別による調整（より寛容に）
        bet_type = bet_info.get('type', '')
        if '三連単' in bet_type:
            bet_ratio *= 0.7  # 70%（以前は50%）
        elif '馬連' in bet_type:
            bet_ratio *= 0.9  # 90%（以前は80%）
        elif 'ワイド' in bet_type:
            bet_ratio *= 1.0  # 100%（変更なし）
        
        # 最小・最大制限
        bet_ratio = max(0.001, min(bet_ratio, 0.05))  # 最小0.1%、最大5%
        
        # 最終的なベット額
        bet_amount = capital * bet_ratio
        
        # 最小ベット額（100円）、最大ベット額（50,000円）
        bet_amount = max(100, min(bet_amount, 50000))
        
        # 100円単位に丸める
        return int(bet_amount / 100) * 100
    
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
                                    payout_data, '三連単', (axis, p2, p3)
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
                                        'estimated_odds': odds,
                                        'actual_odds': actual_odds if actual_odds else odds
                                    })
            
            # 馬連流し（上位2頭）
            if self.enable_quinella and len(sorted_horses) >= 4:
                axis = sorted_horses[0][0]
                for i in range(1, min(6, len(sorted_horses))):
                    partner = sorted_horses[i][0]
                    # 実際のオッズを取得
                    actual_odds = self._get_actual_odds(
                        payout_data, '馬連', tuple(sorted([axis, partner]))
                    )
                    
                    ev, wp, odds = self.calculate_quinella_ev(probs, axis, partner, actual_odds)
                    
                    if ev >= self.min_expected_value:
                        race_bets.append({
                            'type': '馬連',
                            'selection': tuple(sorted([axis, partner])),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds,
                            'actual_odds': actual_odds if actual_odds else odds
                        })
            
            # ワイド（上位5頭から）
            if self.enable_wide and len(sorted_horses) >= 5:
                top5 = [h[0] for h in sorted_horses[:5]]
                for h1, h2 in combinations(top5, 2):
                    # 実際のオッズを取得
                    actual_odds = self._get_actual_odds(
                        payout_data, 'ワイド', tuple(sorted([h1, h2]))
                    )
                    
                    ev, wp, odds = self.calculate_wide_ev(probs, h1, h2, actual_odds)
                    
                    if ev >= self.min_expected_value * 0.9:  # ワイドは少し緩く
                        race_bets.append({
                            'type': 'ワイド',
                            'selection': tuple(sorted([h1, h2])),
                            'expected_value': ev,
                            'win_probability': wp,
                            'estimated_odds': odds,
                            'actual_odds': actual_odds if actual_odds else odds
                        })
            
            # 期待値の高い順に最大5点まで購入
            race_bets.sort(key=lambda x: x['expected_value'], reverse=True)
            
            # デバッグ情報（最初の100レースのみ）
            if i < 100 and race_bets:
                self.logger.debug(f"Race {race_id}: {len(race_bets)} bets considered, "
                                f"top EV: {race_bets[0]['expected_value']:.2f}")
            
            for bet in race_bets[:5]:
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                # 最初の10レースはデバッグ情報を出力
                if i < 10:
                    self.logger.debug(f"Bet amount calculated: {bet_amount}, "
                                    f"EV: {bet['expected_value']:.2f}, "
                                    f"Current capital: {capital}")
                
                if bet_amount <= 0:
                    continue
                
                # レースごとの最大ベット額を緩和（5%→10%）
                if bet_amount > capital * 0.10:
                    bet_amount = int(capital * 0.10 / 100) * 100
                
                # 結果判定
                actual_result = race_data.sort_values('着順')
                is_win, actual_odds = self._check_result(bet, actual_result)
                
                # デバッグ情報（ワイドの的中時）
                if is_win and bet['type'] == 'ワイド' and stats['wide']['wins'] < 10:
                    top3 = [actual_result.iloc[j]['馬番'] for j in range(min(3, len(actual_result)))]
                    self.logger.debug(f"Wide win! Selection: {bet['selection']}, "
                                    f"Top 3: {top3}, Odds: {actual_odds:.1f}")
                
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
        
        # 馬番を整数に統一
        def to_int(x):
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return None
        
        if bet_type == '三連単':
            # 結果の馬番を整数に変換
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(3, len(result)))]
            
            if (len(result_horses) >= 3 and 
                result_horses[0] == selection[0] and
                result_horses[1] == selection[1] and
                result_horses[2] == selection[2]):
                # 的中！実際のオッズを返す
                # actual_oddsが払戻金額の場合は100で割って倍率に変換
                actual = bet.get('actual_odds')
                if actual and actual > 100:  # 払戻金額として保存されている場合
                    return True, actual / 100
                else:
                    return True, bet['estimated_odds']
        
        elif bet_type == '馬連':
            # 上位2頭の馬番を整数に変換
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(2, len(result)))]
            result_horses = [h for h in result_horses if h is not None]
            
            if len(result_horses) >= 2:
                top2 = set(result_horses[:2])
                if set(selection) == top2:
                    actual = bet.get('actual_odds')
                    if actual and actual > 100:
                        return True, actual / 100
                    else:
                        return True, bet['estimated_odds']
        
        elif bet_type == 'ワイド':
            # 上位3頭の馬番を整数に変換
            result_horses = [to_int(result.iloc[i]['馬番']) for i in range(min(3, len(result)))]
            result_horses = [h for h in result_horses if h is not None]
            
            if len(result_horses) >= 3:
                top3 = set(result_horses[:3])
                if set(selection).issubset(top3):
                    actual = bet.get('actual_odds')
                    if actual and actual > 100:
                        return True, actual / 100
                    else:
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
                    raw_payout = json.loads(payout_json)
                    # 不正な形式のデータを修正
                    return self._fix_malformed_payout_data(raw_payout)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.debug(f"Failed to parse payout data: {e}")
        
        return {}
    
    def _fix_malformed_payout_data(self, payout_data: Dict) -> Dict:
        """不正な形式の払戻データを修正"""
        fixed_data = payout_data.copy()
        
        # ワイドデータの修正
        if 'wide' in fixed_data and fixed_data['wide']:
            fixed_wide = {}
            for combo_str, payouts_str in fixed_data['wide'].items():
                # "5 - 85 - 66 - 8" のような形式を解析
                clean_str = combo_str.replace(' ', '')
                
                # 馬番を抽出
                horses = []
                i = 0
                while i < len(clean_str):
                    if clean_str[i] == '-':
                        i += 1
                        continue
                    
                    # 2桁の馬番チェック
                    if i + 1 < len(clean_str) and clean_str[i:i+2].isdigit():
                        horse_num = int(clean_str[i:i+2])
                        if 10 <= horse_num <= 18:
                            horses.append(horse_num)
                            i += 2
                            continue
                    
                    # 1桁の馬番
                    if clean_str[i].isdigit():
                        horses.append(int(clean_str[i]))
                    i += 1
                
                # 重複を除いて最初の3頭を取得
                seen = set()
                unique_horses = []
                for h in horses:
                    if h not in seen:
                        seen.add(h)
                        unique_horses.append(h)
                        if len(unique_horses) >= 3:
                            break
                
                # 3頭の組み合わせを生成
                if len(unique_horses) >= 3:
                    pairs = [
                        (unique_horses[0], unique_horses[1]),
                        (unique_horses[0], unique_horses[2]),
                        (unique_horses[1], unique_horses[2])
                    ]
                    
                    # 払戻金額を分割
                    payout_str = str(payouts_str)
                    payouts = self._split_concatenated_payouts(payout_str, len(pairs))
                    
                    # 各組み合わせに払戻を割り当て
                    for pair, payout in zip(pairs, payouts):
                        key = f"{min(pair)} - {max(pair)}"
                        fixed_wide[key] = payout
                
            fixed_data['wide'] = fixed_wide
        
        return fixed_data
    
    def _split_concatenated_payouts(self, payout_str: str, expected_count: int) -> List[int]:
        """連結された払戻金額を分割"""
        payouts = []
        
        if expected_count > 0 and payout_str:
            # 払戻金額は通常100円単位（末尾が0）
            # 桁数のパターンを推定
            total_len = len(payout_str)
            
            # 均等分割が可能な場合
            if total_len % expected_count == 0:
                chunk_size = total_len // expected_count
                # 各チャンクが妥当な金額か確認（2桁以上）
                if chunk_size >= 2:
                    for i in range(0, total_len, chunk_size):
                        payouts.append(int(payout_str[i:i+chunk_size]))
                    return payouts
            
            # 不均等な場合、末尾から解析
            # 払戻金額は通常3-5桁で、100円単位が多い
            remaining = payout_str
            temp_payouts = []
            
            while remaining:
                found = False
                # 5桁から2桁まで試す
                for length in [5, 4, 3, 2]:
                    if len(remaining) >= length:
                        # 末尾から取得
                        candidate = remaining[-length:]
                        # 100円単位チェック（末尾が00）または10円単位（末尾が0）
                        if length >= 3 and candidate[-2:] == '00':
                            temp_payouts.append(int(candidate))
                            remaining = remaining[:-length]
                            found = True
                            break
                        elif candidate[-1] == '0':
                            temp_payouts.append(int(candidate))
                            remaining = remaining[:-length]
                            found = True
                            break
                
                if not found:
                    # 残りを追加
                    if remaining:
                        temp_payouts.append(int(remaining))
                    break
            
            # 逆順にして正しい順序に
            payouts = temp_payouts[::-1]
            
            # 期待数と一致しない場合、再分割を試みる
            if len(payouts) != expected_count:
                # シンプルな均等分割にフォールバック
                chunk_size = total_len // expected_count
                payouts = []
                for i in range(expected_count):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < expected_count - 1 else total_len
                    payouts.append(int(payout_str[start:end]))
        
        return payouts
    
    def _get_actual_odds(self, payout_data: Dict, bet_type: str, 
                        selection: Tuple) -> Optional[float]:
        """実際のオッズを取得"""
        if not payout_data:
            return None
        
        # bet_typeをpayout_dataのキーに変換
        # バックテストで使用するbet_typeは'trifecta', 'quinella', 'wide'
        type_mapping = {
            'trifecta': 'trifecta',
            'quinella': 'quinella', 
            'wide': 'wide'
        }
        
        payout_key = type_mapping.get(bet_type)
        if not payout_key or payout_key not in payout_data:
            return None
        
        bet_data = payout_data[payout_key]
        if not isinstance(bet_data, dict):
            return None
        
        # 三連単の場合 - "→" で区切られている
        if bet_type == 'trifecta':
            # 複数のキー形式を試す
            key_formats = [
                f"{selection[0]} → {selection[1]} → {selection[2]}",
                f"{selection[0]}→{selection[1]}→{selection[2]}",
                f"{selection[0]}-{selection[1]}-{selection[2]}"
            ]
            
            for key in key_formats:
                odds = bet_data.get(key)
                if odds:
                    try:
                        return float(odds)
                    except (ValueError, TypeError):
                        continue
        
        # 馬連・ワイドの場合 - " - " で区切られている
        elif bet_type in ['quinella', 'wide']:
            # ワイドの特殊な形式に対応（複数の組み合わせが連結されている場合）
            if bet_type == 'wide':
                for k, v in bet_data.items():
                    # "2 - 53 - 52 - 3" のような形式を解析
                    parts = [h.strip() for h in k.split(' - ')]
                    
                    # 数値以外の要素を除外
                    parts = [h for h in parts if h.isdigit()]
                    
                    # 2桁馬番の分解 (53 → 5, 3)
                    horses = []
                    for p in parts:
                        if len(p) == 2 and int(p) > 18:
                            # 2桁の場合は分割
                            horses.extend([str(int(p[0])), str(int(p[1]))])
                        else:
                            horses.append(p)
                    
                    # ユニークな馬番を取得
                    unique_horses = sorted(set(horses), key=int)
                    
                    # ターゲットの組み合わせをチェック
                    target_set = set(map(str, selection))
                    
                    # すべての2頭組み合わせを生成
                    from itertools import combinations
                    pairs = list(combinations(unique_horses, 2))
                    
                    # ターゲットが含まれているかチェック
                    for i, (h1, h2) in enumerate(pairs):
                        if set([h1, h2]) == target_set:
                            # 複数のペアが含まれる場合、配当を分割
                            return self._extract_wide_payout_by_index(v, i, len(pairs))
            
            # 通常の形式を試す（順序も考慮）
            key_formats = [
                f"{selection[0]} - {selection[1]}",
                f"{selection[1]} - {selection[0]}",
                f"{selection[0]}-{selection[1]}",
                f"{selection[1]}-{selection[0]}"
            ]
            
            for key in key_formats:
                odds = bet_data.get(key)
                if odds:
                    try:
                        return float(odds)
                    except (ValueError, TypeError):
                        continue
                        
                        # 2頭ずつの組み合わせを確認
                        for i in range(0, len(parts), 2):
                            if i + 1 < len(parts):
                                pair = {parts[i], parts[i+1]}
                                if pair == selection_set:
                                    # 対応する払戻金額を抽出
                                    # 払戻金額も連結されている可能性があるので、適切に分割
                                    return self._extract_payout_amount(v, i // 2)
                        
                        # 通常の形式でチェック
                        horses_in_key = set()
                        for h in k.split(' - '):
                            h = h.strip()
                            if h:
                                horses_in_key.add(h)
                        
                        if horses_in_key == selection_set:
                            return float(v)
                            
                    except (ValueError, TypeError, AttributeError):
                        continue
        
        # 単勝の場合
        elif bet_type == '単勝':
            key = str(selection[0])
            odds = bet_data.get(key)
            if odds:
                try:
                    return float(odds)
                except (ValueError, TypeError):
                    return None
        
        return None
    
    def _extract_payout_amount(self, payout_str: Any, index: int) -> Optional[float]:
        """連結された払戻金額から特定のインデックスの金額を抽出"""
        try:
            # 数値文字列を3桁ずつに分割
            payout_str = str(payout_str)
            amounts = []
            
            # "11401620520" -> ["1140", "1620", "520"] のように分割を試みる
            # 通常、払戻金額は100円単位なので、末尾から3〜4桁ずつ取る
            remaining = payout_str
            while remaining:
                if len(remaining) >= 3:
                    # 末尾3桁を取得
                    amount = remaining[-3:]
                    remaining = remaining[:-3]
                    
                    # 先頭に残った1〜2桁があれば結合
                    if remaining and len(remaining) <= 2:
                        amount = remaining + amount
                        remaining = ""
                    
                    amounts.append(int(amount))
                else:
                    # 残りが3桁未満の場合
                    if remaining:
                        amounts.append(int(remaining))
                    break
            
            # リストを逆順にして正しい順序に
            amounts.reverse()
            
            # 指定されたインデックスの金額を返す
            if 0 <= index < len(amounts):
                return float(amounts[index])
                
        except (ValueError, TypeError, AttributeError):
            pass
        
        return None
    
    def _calculate_metrics(self, initial_capital: float, final_capital: float,
                          all_bets: List[Dict], stats: Dict) -> Dict:
        """詳細な指標を計算"""
        total_bets = len(all_bets)
        winning_bets = [b for b in all_bets if b['is_win']]
        
        # 年間リターンの計算（テスト期間の年数を考慮）
        if hasattr(self, 'test_data') and '日付' in self.test_data.columns:
            # 日付が文字列の場合は変換
            if self.test_data['日付'].dtype == 'object':
                # よくある日付フォーマットを試す
                date_formats = ['%Y年%m月%d日', '%Y/%m/%d', '%Y-%m-%d']
                dates = None
                for fmt in date_formats:
                    try:
                        dates = pd.to_datetime(self.test_data['日付'], format=fmt)
                        break
                    except:
                        continue
                if dates is None:
                    dates = pd.to_datetime(self.test_data['日付'], errors='coerce')
                test_years = len(dates.dt.year.unique()) if not dates.isna().all() else 1
            else:
                test_years = len(self.test_data['日付'].dt.year.unique())
        else:
            test_years = 1
            
        total_return = (final_capital - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (1 / max(test_years, 1)) - 1
        
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_bets': total_bets,
            'total_wins': len(winning_bets),
            'win_rate': len(winning_bets) / total_bets if total_bets > 0 else 0,
            'avg_expected_value': np.mean([b['bet']['expected_value'] for b in all_bets]) if all_bets else 0,
            'avg_bet': np.mean([b['amount'] for b in all_bets]) if all_bets else 0,
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
    
    def _extract_wide_payout(self, payout_value: int, all_horses: List[str]) -> float:
        """ワイドの複合払戻から個別の払戻を抽出"""
        # "2 - 53 - 52 - 3": 190150440 のような場合
        # 実際には：
        # 2-3: 190円, 2-5: 150円, 3-5: 440円 のように連結されている
        
        from itertools import combinations
        
        # 馬番を整数に変換してソート
        horses_int = sorted([int(h) for h in all_horses if h.isdigit()])
        
        # 馬番が2桁の場合の処理
        # 53 → 5と3 に分割
        expanded_horses = []
        for h in horses_int:
            if h > 18:  # 2桁馬番の連結
                expanded_horses.extend([h // 10, h % 10])
            else:
                expanded_horses.append(h)
        
        horses_int = sorted(set(expanded_horses))
        num_pairs = len(list(combinations(horses_int, 2)))
        
        # 払戻金額の桁数を確認
        payout_str = str(payout_value)
        
        # 各ペアの払戻を抽出
        if num_pairs > 1 and len(payout_str) >= num_pairs * 3:
            # 3桁ずつ分割
            payouts = []
            for i in range(0, len(payout_str), 3):
                if i + 3 <= len(payout_str):
                    payouts.append(int(payout_str[i:i+3]) * 10)
            
            if len(payouts) >= 1:
                return float(payouts[0])  # 最初のペアの払戻
        
        # 単一の払戻の場合
        return float(payout_value)
    
    def _extract_wide_payout_by_index(self, payout_value: int, index: int, total_pairs: int) -> float:
        """ワイドの複合払戻から特定のインデックスの払戻を抽出"""
        payout_str = str(payout_value)
        
        # 190150440 のような形式を想定
        # 190, 150, 440 の3つの払戻が連結
        if len(payout_str) == 9 and total_pairs == 3:
            # 3桁ずつ分割
            payouts = [
                int(payout_str[0:3]) * 10,  # 190 -> 1900円
                int(payout_str[3:6]) * 10,  # 150 -> 1500円
                int(payout_str[6:9]) * 10   # 440 -> 4400円
            ]
            if 0 <= index < len(payouts):
                return float(payouts[index])
        
        # その他のパターンを試す
        # 6ペアの場合など
        if total_pairs > 1 and len(payout_str) % total_pairs == 0:
            digit_per_payout = len(payout_str) // total_pairs
            if digit_per_payout >= 3:
                payouts = []
                for i in range(0, len(payout_str), digit_per_payout):
                    payouts.append(int(payout_str[i:i+digit_per_payout]))
                
                if 0 <= index < len(payouts):
                    return float(payouts[index])
        
        # 分割できない場合は全体を返す
        return float(payout_value)
    
    def _calculate_profit(self, bet: Dict, bet_amount: float) -> float:
        """BaseStrategy用のダミー実装"""
        return 0