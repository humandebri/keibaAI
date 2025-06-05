#!/usr/bin/env python3
"""
改善された馬券戦略 v2
- 全レースを期待値でランク付けしてから選定
- 月次でのドローダウン管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class ImprovedBettingStrategyV2(AdvancedBettingStrategy):
    """改善された馬券戦略 v2"""
    
    def __init__(self, 
                 min_expected_value: float = 1.3,   # さらに厳格に
                 enable_trifecta: bool = False,     # 三連単は無効化
                 enable_quinella: bool = True,
                 enable_wide: bool = True,
                 use_actual_odds: bool = True,
                 kelly_fraction: float = 0.5,        # Kelly50%
                 max_dd_ratio: float = 0.15,        # 月次最大DD15%
                 top_races_percentile: float = 0.95): # 上位5%のレースのみ
        
        super().__init__(
            min_expected_value=min_expected_value,
            enable_trifecta=enable_trifecta,
            enable_quinella=enable_quinella,
            enable_wide=enable_wide,
            use_actual_odds=use_actual_odds,
            kelly_fraction=kelly_fraction
        )
        
        self.max_dd_ratio = max_dd_ratio
        self.top_races_percentile = top_races_percentile
        self.monthly_peak = defaultdict(float)
        
    def analyze_all_races(self, model) -> List[Dict]:
        """全レースを分析して期待値でランク付け"""
        self.logger.info("全レースを分析中...")
        all_race_scores = []
        
        unique_races = self.test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if i % 500 == 0:
                self.logger.debug(f"Analyzing race {i}/{len(unique_races)}")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            if len(race_data) < 6:
                continue
            
            # 払戻データを取得
            payout_data = self._get_payout_data(race_data)
            
            # 確率予測
            probs = self.predict_probabilities(model, race_data)
            if not probs:
                continue
            
            # レースの最高期待値を計算
            race_max_ev = 0
            race_best_bets = []
            
            sorted_horses = sorted(probs.items(), 
                                 key=lambda x: x[1]['predicted_rank'])
            
            # 馬連の期待値計算
            if self.enable_quinella and len(sorted_horses) >= 4:
                for i in range(min(4, len(sorted_horses))):
                    for j in range(i+1, min(5, len(sorted_horses))):
                        h1, h2 = sorted_horses[i][0], sorted_horses[j][0]
                        actual_odds = self._get_actual_odds(
                            payout_data, '馬連', tuple(sorted([h1, h2]))
                        )
                        
                        ev, wp, odds = self.calculate_quinella_ev(probs, h1, h2, actual_odds)
                        
                        if ev > race_max_ev:
                            race_max_ev = ev
                        
                        if ev >= self.min_expected_value:
                            race_best_bets.append({
                                'type': '馬連',
                                'selection': tuple(sorted([h1, h2])),
                                'expected_value': ev,
                                'win_probability': wp,
                                'estimated_odds': odds,
                                'actual_odds': actual_odds if actual_odds else odds,
                                'confidence': ev * np.sqrt(wp)  # 信頼度スコア
                            })
            
            # ワイドの期待値計算
            if self.enable_wide and len(sorted_horses) >= 4:
                for i in range(min(4, len(sorted_horses))):
                    for j in range(i+1, min(5, len(sorted_horses))):
                        h1, h2 = sorted_horses[i][0], sorted_horses[j][0]
                        actual_odds = self._get_actual_odds(
                            payout_data, 'ワイド', tuple(sorted([h1, h2]))
                        )
                        
                        ev, wp, odds = self.calculate_wide_ev(probs, h1, h2, actual_odds)
                        
                        if ev > race_max_ev:
                            race_max_ev = ev
                        
                        if ev >= self.min_expected_value * 0.9:
                            race_best_bets.append({
                                'type': 'ワイド',
                                'selection': tuple(sorted([h1, h2])),
                                'expected_value': ev,
                                'win_probability': wp,
                                'estimated_odds': odds,
                                'actual_odds': actual_odds if actual_odds else odds,
                                'confidence': ev * np.sqrt(wp) * 1.2  # ワイドは信頼度UP
                            })
            
            if race_best_bets:
                # 信頼度順にソート
                race_best_bets.sort(key=lambda x: x['confidence'], reverse=True)
                
                all_race_scores.append({
                    'race_id': race_id,
                    'race_data': race_data,
                    'max_ev': race_max_ev,
                    'best_bets': race_best_bets[:3],  # 上位3つまで
                    'date': pd.to_datetime(str(race_id)[:8], format='%Y%m%d')
                })
        
        # 最高期待値でソート
        all_race_scores.sort(key=lambda x: x['max_ev'], reverse=True)
        
        # 上位X%のみ選定
        cutoff_index = int(len(all_race_scores) * (1 - self.top_races_percentile))
        selected_races = all_race_scores[:cutoff_index]
        
        self.logger.info(f"全{len(all_race_scores)}レースから上位{len(selected_races)}レースを選定")
        
        return selected_races
        
    def check_monthly_drawdown(self, capital: float, month: str) -> bool:
        """月次ドローダウンチェック"""
        if month not in self.monthly_peak or self.monthly_peak[month] < capital:
            self.monthly_peak[month] = capital
        
        drawdown = (self.monthly_peak[month] - capital) / self.monthly_peak[month]
        
        if drawdown > self.max_dd_ratio:
            self.logger.warning(f"月次DD制限に到達: {month}, DD: {drawdown:.1%}")
            return True
        
        return False
        
    def run_backtest(self, initial_capital: float = 1_000_000) -> Dict:
        """改善されたバックテスト v2"""
        self.logger.info("Running improved betting backtest v2")
        
        # モデル訓練
        model = self.train_model()
        
        # 全レースを分析
        selected_races = self.analyze_all_races(model)
        
        # 日付順にソート（時系列を保つ）
        selected_races.sort(key=lambda x: x['date'])
        
        # バックテスト
        capital = initial_capital
        all_bets = []
        stats = {
            'quinella': {'count': 0, 'wins': 0, 'profit': 0},
            'wide': {'count': 0, 'wins': 0, 'profit': 0}
        }
        
        monthly_bets = defaultdict(int)
        
        for race_info in selected_races:
            race_id = race_info['race_id']
            race_data = race_info['race_data']
            best_bets = race_info['best_bets']
            race_month = race_info['date'].strftime('%Y%m')
            
            # 月次ドローダウンチェック
            if self.check_monthly_drawdown(capital, race_month):
                # この月はスキップ
                if race_info['date'].month != (selected_races[selected_races.index(race_info)+1]['date'].month 
                                               if selected_races.index(race_info)+1 < len(selected_races) else 0):
                    # 月が変わったらリセット
                    self.logger.info(f"{race_month}の取引を再開")
                else:
                    continue
            
            # 実際の結果を取得
            actual_result = race_data.sort_values('着順')
            
            # 最も信頼度の高い賭けを実行
            for bet in best_bets[:2]:  # 最大2点
                # Kelly基準でベット額計算
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                # 最低2000円、最大資金の5%
                bet_amount = max(2000, min(bet_amount, capital * 0.05))
                bet_amount = int(bet_amount / 100) * 100
                
                if bet_amount > capital:
                    continue
                
                # 結果判定
                is_win, actual_odds = self._check_result(bet, actual_result)
                
                if is_win:
                    profit = bet_amount * actual_odds - bet_amount
                    bet_type_key = bet['type'].replace('馬連', 'quinella')\
                                             .replace('ワイド', 'wide')
                    stats[bet_type_key]['wins'] += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                
                # 統計更新
                bet_type_key = bet['type'].replace('馬連', 'quinella')\
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
                    'date': race_info['date']
                })
                
                monthly_bets[race_month] += 1
                
                if capital <= 0:
                    self.logger.warning("Bankrupt!")
                    break
            
            if capital <= 0:
                break
        
        # 結果集計
        self.results = {
            'trades': all_bets,
            'metrics': self._calculate_metrics(initial_capital, capital, all_bets, stats),
            'monthly_bets': dict(monthly_bets)
        }
        
        return self.results


def main():
    """改善戦略v2のテスト実行"""
    logger = setup_logger('improved_strategy_v2')
    
    # 出力ディレクトリ
    output_dir = Path('backtest_improved_v2')
    output_dir.mkdir(exist_ok=True)
    
    logger.info("改善された戦略v2でバックテストを開始...")
    
    # 戦略の初期化
    strategy = ImprovedBettingStrategyV2(
        min_expected_value=1.3,      # さらに厳格な期待値フィルタ
        enable_trifecta=False,       # 三連単は無効
        enable_quinella=True,        # 馬連有効
        enable_wide=True,            # ワイド有効
        use_actual_odds=True,
        kelly_fraction=0.5,          # Kelly50%
        max_dd_ratio=0.15,          # 月次最大DD15%
        top_races_percentile=0.95    # 上位5%のレースのみ
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
                'min_expected_value': 1.3,
                'enable_trifecta': False,
                'enable_quinella': True,
                'enable_wide': True,
                'kelly_fraction': 0.5,
                'max_dd_ratio': 0.15,
                'top_races_percentile': 0.95
            },
            'metrics': results['metrics'],
            'monthly_bets': results['monthly_bets']
        }, f, ensure_ascii=False, indent=2)
    
    # 取引履歴を保存
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(output_dir / 'backtest_trades.csv', index=False, encoding='utf-8')
    
    # 結果表示
    metrics = results['metrics']
    print("\n" + "="*60)
    print("改善された戦略v2のバックテスト結果")
    print("="*60)
    print(f"初期資金: ¥{metrics['initial_capital']:,}")
    print(f"最終資金: ¥{metrics['final_capital']:,.0f}")
    print(f"収益: ¥{metrics['final_capital'] - metrics['initial_capital']:,.0f}")
    print(f"収益率: {metrics['total_return']*100:.1f}%")
    print(f"総賭け回数: {metrics['total_bets']}回")
    print(f"勝利数: {metrics['total_wins']}回")
    print(f"勝率: {metrics['win_rate']*100:.1f}%")
    print(f"平均賭け金額: ¥{metrics.get('avg_bet', 0):,.0f}")
    print("\n馬券種別:")
    for bet_type, stats in metrics['by_type'].items():
        print(f"  {bet_type}: {stats['count']}回, 勝率{stats['win_rate']*100:.1f}%, ROI{stats['roi']*100:.1f}%")
    
    print("\n月別賭け回数:")
    for month, count in sorted(results['monthly_bets'].items()):
        print(f"  {month}: {count}回")
    print("="*60)


if __name__ == "__main__":
    main()