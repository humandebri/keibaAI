#!/usr/bin/env python3
"""
改善された馬券戦略
- 馬連・ワイド重視
- より厳格な期待値フィルタ
- ドローダウン管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class ImprovedBettingStrategy(AdvancedBettingStrategy):
    """改善された馬券戦略"""
    
    def __init__(self, 
                 min_expected_value: float = 1.2,  # より厳格に
                 enable_trifecta: bool = False,    # 三連単は無効化
                 enable_quinella: bool = True,
                 enable_wide: bool = True,
                 enable_exacta: bool = True,        # 馬単を追加
                 use_actual_odds: bool = True,
                 kelly_fraction: float = 0.5,       # より積極的に
                 max_races_per_day: int = 3,        # 1日の最大レース数
                 stop_loss_ratio: float = 0.1):     # ストップロス（10%）
        
        super().__init__(
            min_expected_value=min_expected_value,
            enable_trifecta=enable_trifecta,
            enable_quinella=enable_quinella,
            enable_wide=enable_wide,
            use_actual_odds=use_actual_odds,
            kelly_fraction=kelly_fraction
        )
        
        self.enable_exacta = enable_exacta
        self.max_races_per_day = max_races_per_day
        self.stop_loss_ratio = stop_loss_ratio
        self.daily_race_count = 0
        self.current_date = None
        self.peak_capital = 0
        
    def calculate_exacta_ev(self, probs: Dict[int, Dict], 
                           h1: int, h2: int,
                           actual_odds: Optional[float] = None) -> Tuple[float, float, float]:
        """馬単の期待値計算"""
        # 馬1が1着、馬2が2着の確率
        win_prob = probs[h1]['win_prob'] * probs[h2]['show_prob']
        
        if self.use_actual_odds and actual_odds is not None and actual_odds > 0:
            estimated_odds = actual_odds / 100
        else:
            # 馬連の1.5倍程度
            pop_avg = (probs[h1]['popularity'] + probs[h2]['popularity']) / 2
            if pop_avg <= 3:
                base_odds = 8 + pop_avg * 5
            elif pop_avg <= 6:
                base_odds = 25 + pop_avg * 8
            elif pop_avg <= 10:
                base_odds = 60 + pop_avg * 12
            else:
                base_odds = min(150 + pop_avg * 20, 800)
            
            estimated_odds = base_odds * (1 - self.jra_take_rate)
        
        expected_value = win_prob * estimated_odds
        return expected_value, win_prob, estimated_odds
        
    def filter_bets_by_confidence(self, race_bets: List[Dict]) -> List[Dict]:
        """信頼度でフィルタリング"""
        filtered = []
        
        for bet in race_bets:
            # 期待値と的中確率の両方を考慮
            ev = bet['expected_value']
            wp = bet['win_probability']
            
            # 高期待値・高確率のものを優先
            confidence_score = ev * np.sqrt(wp)
            
            # 馬券種別による調整
            if bet['type'] == 'ワイド':
                confidence_score *= 1.2  # ワイドを優遇
            elif bet['type'] == '馬連':
                confidence_score *= 1.1
            elif bet['type'] == '馬単':
                confidence_score *= 1.0
            elif bet['type'] == '三連単':
                confidence_score *= 0.5  # 三連単は抑制
                
            bet['confidence_score'] = confidence_score
            
            # 閾値以上のみ採用
            if confidence_score >= 0.15:  # 調整可能
                filtered.append(bet)
                
        # 信頼度順にソート
        return sorted(filtered, key=lambda x: x['confidence_score'], reverse=True)
        
    def apply_stop_loss(self, capital: float) -> bool:
        """ストップロスの適用"""
        if capital < self.peak_capital * (1 - self.stop_loss_ratio):
            self.logger.warning(f"Stop loss triggered. Peak: {self.peak_capital}, Current: {capital}")
            return True
        return False
        
    def run_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """改善されたバックテスト"""
        self.logger.info("Running improved betting backtest")
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト
        capital = initial_capital
        self.peak_capital = initial_capital
        all_bets = []
        stats = {
            'trifecta': {'count': 0, 'wins': 0, 'profit': 0},
            'quinella': {'count': 0, 'wins': 0, 'profit': 0},
            'wide': {'count': 0, 'wins': 0, 'profit': 0},
            'exacta': {'count': 0, 'wins': 0, 'profit': 0}
        }
        
        unique_races = self.test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races[:5000]):  # 最大5000レース
            if i % 200 == 0:
                self.logger.debug(f"Processing race {i}/{min(len(unique_races), 5000)}")
            
            # 日付チェック（1日の最大レース数）
            race_date = str(race_id)[:8]
            if self.current_date != race_date:
                self.current_date = race_date
                self.daily_race_count = 0
            
            if self.daily_race_count >= self.max_races_per_day:
                continue
                
            # ストップロスチェック
            if self.apply_stop_loss(capital):
                break
                
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            if len(race_data) < 6:  # 出走頭数が少ないレースはスキップ
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
            
            # 馬連（上位4頭から）
            if self.enable_quinella and len(sorted_horses) >= 4:
                for i in range(min(4, len(sorted_horses))):
                    for j in range(i+1, min(5, len(sorted_horses))):
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
            
            # ワイド（上位4頭から）
            if self.enable_wide and len(sorted_horses) >= 4:
                for i in range(min(4, len(sorted_horses))):
                    for j in range(i+1, min(5, len(sorted_horses))):
                        h1, h2 = sorted_horses[i][0], sorted_horses[j][0]
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
            
            # 馬単（上位3頭から）
            if self.enable_exacta and len(sorted_horses) >= 3:
                for i in range(min(3, len(sorted_horses))):
                    for j in range(min(3, len(sorted_horses))):
                        if i != j:
                            h1, h2 = sorted_horses[i][0], sorted_horses[j][0]
                            actual_odds = self._get_actual_odds(
                                payout_data, '馬単', (h1, h2)
                            )
                            
                            ev, wp, odds = self.calculate_exacta_ev(probs, h1, h2, actual_odds)
                            
                            if ev >= self.min_expected_value:
                                race_bets.append({
                                    'type': '馬単',
                                    'selection': (h1, h2),
                                    'expected_value': ev,
                                    'win_probability': wp,
                                    'estimated_odds': odds,
                                    'actual_odds': actual_odds if actual_odds else odds
                                })
            
            # 信頼度でフィルタリング
            race_bets = self.filter_bets_by_confidence(race_bets)
            
            # 最大3点まで
            for bet in race_bets[:3]:
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                # 最低1000円、最大資金の5%
                bet_amount = max(1000, min(bet_amount, capital * 0.05))
                bet_amount = int(bet_amount / 100) * 100
                
                if bet_amount > capital:
                    continue
                
                # 結果判定
                actual_result = race_data.sort_values('着順')
                is_win, actual_odds = self._check_result(bet, actual_result)
                
                if is_win:
                    profit = bet_amount * actual_odds - bet_amount
                    bet_type_key = bet['type'].replace('馬連', 'quinella')\
                                             .replace('ワイド', 'wide')\
                                             .replace('馬単', 'exacta')
                    stats[bet_type_key]['wins'] += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                
                # ピーク資本を更新
                if capital > self.peak_capital:
                    self.peak_capital = capital
                
                # 統計更新
                bet_type_key = bet['type'].replace('馬連', 'quinella')\
                                         .replace('ワイド', 'wide')\
                                         .replace('馬単', 'exacta')
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
            
            if len([b for b in all_bets if b['race_id'] == race_id]) > 0:
                self.daily_race_count += 1
            
            if capital <= 0:
                break
        
        # 結果集計
        self.results = {
            'trades': all_bets,
            'metrics': self._calculate_metrics(initial_capital, capital, all_bets, stats)
        }
        
        return self.results


def main():
    """改善戦略のテスト実行"""
    logger = setup_logger('improved_strategy')
    
    # 出力ディレクトリ
    output_dir = Path('backtest_improved')
    output_dir.mkdir(exist_ok=True)
    
    logger.info("改善された戦略でバックテストを開始...")
    
    # 戦略の初期化
    strategy = ImprovedBettingStrategy(
        min_expected_value=1.2,      # より厳格な期待値フィルタ
        enable_trifecta=False,       # 三連単は無効
        enable_quinella=True,        # 馬連有効
        enable_wide=True,            # ワイド有効
        enable_exacta=True,          # 馬単有効
        use_actual_odds=True,
        kelly_fraction=0.5,          # Kelly50%
        max_races_per_day=3,         # 1日最大3レース
        stop_loss_ratio=0.1          # 10%ストップロス
    )
    
    # データ読み込み
    strategy.load_data(start_year=2022, end_year=2025, use_payout_data=True)
    
    # データ分割
    strategy.split_data(train_years=[2022, 2023, 2024], test_years=[2025])
    
    # バックテスト実行
    results = strategy.run_backtest(initial_capital=1_000_000)
    
    # 結果を保存
    with open(output_dir / 'backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'parameters': {
                'min_expected_value': 1.2,
                'enable_trifecta': False,
                'enable_quinella': True,
                'enable_wide': True,
                'enable_exacta': True,
                'kelly_fraction': 0.5,
                'max_races_per_day': 3,
                'stop_loss_ratio': 0.1
            },
            'metrics': results['metrics']
        }, f, ensure_ascii=False, indent=2)
    
    # 結果表示
    metrics = results['metrics']
    print("\n" + "="*60)
    print("改善された戦略のバックテスト結果")
    print("="*60)
    print(f"初期資金: ¥{metrics['initial_capital']:,}")
    print(f"最終資金: ¥{metrics['final_capital']:,.0f}")
    print(f"収益: ¥{metrics['final_capital'] - metrics['initial_capital']:,.0f}")
    print(f"収益率: {metrics['total_return']*100:.1f}%")
    print(f"総賭け回数: {metrics['total_bets']}回")
    print(f"勝利数: {metrics['total_wins']}回")
    print(f"勝率: {metrics['win_rate']*100:.1f}%")
    print("\n馬券種別:")
    for bet_type, stats in metrics['by_type'].items():
        print(f"  {bet_type}: {stats['count']}回, 勝率{stats['win_rate']*100:.1f}%, ROI{stats['roi']*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()