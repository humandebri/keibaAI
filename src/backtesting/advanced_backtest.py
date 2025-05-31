#!/usr/bin/env python3
"""
高度なバックテストシステム
オッズ別の期待値分析と複数戦略のシミュレーション
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class AdvancedBacktest:
    def __init__(self):
        self.initial_capital = 1000000
        self.results = {}
        
    def load_data(self):
        """データの読み込み"""
        print("Loading data...")
        dfs = []
        for year in range(2014, 2024):
            try:
                df = pd.read_excel(f'data/{year}.xlsx')
                df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                df = df.dropna(subset=['着順'])
                df['year'] = year
                print(f"Loaded {year}.xlsx: {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {year}.xlsx - {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total rows: {len(self.data)}")
        self.prepare_features()
        
    def prepare_features(self):
        """特徴量エンジニアリング"""
        print("Preparing features...")
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        # 数値変数の欠損値処理
        numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # ターゲット変数
        self.data['is_win'] = (self.data['着順'] == 1).astype(int)
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
        
        # オッズ帯のカテゴリ作成
        self.data['odds_category'] = pd.cut(
            self.data['オッズ'],
            bins=[0, 2, 5, 10, 20, 50, 100, float('inf')],
            labels=['1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+']
        )
    
    def get_features(self, data):
        """特徴量の取得"""
        feature_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化',
                          '性', '馬場', '天気', '芝・ダート', '場名']
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                features.append(data[col].values)
        
        if len(features) == 0:
            return None
            
        return np.column_stack(features)
    
    def train_model(self, train_data, target_type='place'):
        """モデルの訓練"""
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
            
        target = train_data[f'is_{target_type}']
        
        # クラス重み調整
        pos_weight = len(target[target == 0]) / len(target[target == 1])
        
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': pos_weight
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def calculate_place_odds(self, win_odds):
        """複勝オッズの推定（改善版）"""
        # より現実的な複勝オッズモデル
        if win_odds <= 1.5:
            return 1.1
        elif win_odds <= 2.0:
            return win_odds * 0.45
        elif win_odds <= 3.0:
            return win_odds * 0.40
        elif win_odds <= 5.0:
            return win_odds * 0.35
        elif win_odds <= 10.0:
            return win_odds * 0.30
        elif win_odds <= 20.0:
            return win_odds * 0.25
        elif win_odds <= 50.0:
            return win_odds * 0.20
        else:
            return win_odds * 0.15
    
    def analyze_odds_performance(self):
        """オッズ帯別のパフォーマンス分析"""
        print("\nAnalyzing performance by odds range...")
        
        odds_analysis = []
        
        for category in self.data['odds_category'].cat.categories:
            category_data = self.data[self.data['odds_category'] == category]
            
            if len(category_data) > 0:
                win_rate = category_data['is_win'].mean()
                place_rate = category_data['is_place'].mean()
                avg_odds = category_data['オッズ'].mean()
                
                # 理論的期待値
                win_ev = win_rate * avg_odds
                place_ev = place_rate * self.calculate_place_odds(avg_odds)
                
                odds_analysis.append({
                    'オッズ帯': category,
                    '出走数': len(category_data),
                    '単勝率': win_rate,
                    '複勝率': place_rate,
                    '平均オッズ': avg_odds,
                    '単勝期待値': win_ev,
                    '複勝期待値': place_ev
                })
        
        self.odds_performance = pd.DataFrame(odds_analysis)
        return self.odds_performance
    
    def optimize_ev_threshold(self, model, test_data, odds_range=None):
        """オッズ帯別の最適な期待値閾値を探索"""
        if odds_range:
            test_data = test_data[test_data['odds_category'] == odds_range]
        
        thresholds = np.arange(0.8, 2.0, 0.05)
        best_threshold = 1.0
        best_profit = -float('inf')
        
        for threshold in thresholds:
            profit = self.simulate_betting(
                model, test_data, 
                betting_fraction=0.005,
                ev_threshold=threshold,
                verbose=False
            )['total_profit']
            
            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold
        
        return best_threshold, best_profit
    
    def simulate_betting(self, model, test_data, betting_fraction=0.005, 
                        ev_threshold=1.2, verbose=True):
        """ベッティングシミュレーション"""
        capital = self.initial_capital
        history = []
        bets_by_odds = {}
        
        unique_races = test_data['race_id'].unique()
        
        for race_id in unique_races:
            race_data = test_data[test_data['race_id'] == race_id]
            
            features = self.get_features(race_data)
            if features is None:
                continue
                
            predictions = model.predict(features)
            
            # 最良の馬を選択
            best_horse_idx = None
            best_ev = 0
            
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                win_odds = horse['オッズ']
                place_odds = self.calculate_place_odds(win_odds)
                place_prob = predictions[idx]
                
                ev = place_prob * place_odds
                
                if ev > ev_threshold and ev > best_ev:
                    best_ev = ev
                    best_horse_idx = idx
            
            if best_horse_idx is not None:
                bet_amount = capital * betting_fraction
                horse = race_data.iloc[best_horse_idx]
                odds_cat = str(horse['odds_category'])
                
                # オッズ帯別の統計
                if odds_cat not in bets_by_odds:
                    bets_by_odds[odds_cat] = {
                        'bets': 0, 'wins': 0, 'profit': 0
                    }
                
                bets_by_odds[odds_cat]['bets'] += 1
                
                # 結果判定
                if horse['着順'] <= 3:
                    place_odds = self.calculate_place_odds(horse['オッズ'])
                    payout = bet_amount * place_odds
                    profit = payout - bet_amount
                    bets_by_odds[odds_cat]['wins'] += 1
                else:
                    profit = -bet_amount
                
                bets_by_odds[odds_cat]['profit'] += profit
                capital += profit
                
                history.append({
                    'race_id': race_id,
                    'capital': capital,
                    'bet_amount': bet_amount,
                    'profit': profit,
                    'odds': horse['オッズ'],
                    'odds_category': odds_cat,
                    'result': horse['着順'] <= 3
                })
        
        return {
            'final_capital': capital,
            'total_profit': capital - self.initial_capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'history': history,
            'bets_by_odds': bets_by_odds
        }
    
    def run_comprehensive_backtest(self):
        """包括的なバックテスト実行"""
        # データ分割
        train_data = self.data[self.data['year'].isin(range(2014, 2021))]
        test_data = self.data[self.data['year'].isin(range(2021, 2024))]
        
        print(f"\nTrain data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # オッズ帯別パフォーマンス分析
        odds_perf = self.analyze_odds_performance()
        print("\nOdds Performance Analysis:")
        print(odds_perf.to_string(index=False))
        
        # モデル訓練
        print("\nTraining model...")
        model = self.train_model(train_data, target_type='place')
        
        # 複数の戦略でシミュレーション
        strategies = [
            {'name': 'Conservative', 'fraction': 0.003, 'threshold': 1.5},
            {'name': 'Standard', 'fraction': 0.005, 'threshold': 1.2},
            {'name': 'Aggressive', 'fraction': 0.01, 'threshold': 1.0},
            {'name': 'Adaptive', 'fraction': 0.005, 'threshold': 'adaptive'}
        ]
        
        results = {}
        
        for strategy in strategies:
            print(f"\n--- {strategy['name']} Strategy ---")
            
            if strategy['threshold'] == 'adaptive':
                # オッズ帯別の最適閾値を使用
                total_profit = 0
                adaptive_results = {}
                
                for odds_cat in self.data['odds_category'].cat.categories:
                    cat_data = test_data[test_data['odds_category'] == odds_cat]
                    if len(cat_data) > 100:  # 十分なデータがある場合のみ
                        threshold, _ = self.optimize_ev_threshold(model, cat_data)
                        result = self.simulate_betting(
                            model, cat_data,
                            betting_fraction=strategy['fraction'],
                            ev_threshold=threshold,
                            verbose=False
                        )
                        adaptive_results[odds_cat] = {
                            'threshold': threshold,
                            'profit': result['total_profit']
                        }
                        total_profit += result['total_profit']
                
                results[strategy['name']] = {
                    'total_profit': total_profit,
                    'total_return': total_profit / self.initial_capital,
                    'adaptive_thresholds': adaptive_results
                }
            else:
                result = self.simulate_betting(
                    model, test_data,
                    betting_fraction=strategy['fraction'],
                    ev_threshold=strategy['threshold']
                )
                results[strategy['name']] = result
            
            print(f"Total Return: {results[strategy['name']]['total_return']:.2%}")
        
        self.results = results
        return results
    
    def visualize_results(self):
        """結果の可視化"""
        # 戦略別リターンの比較
        plt.figure(figsize=(12, 8))
        
        # 1. 戦略別の総リターン
        plt.subplot(2, 2, 1)
        strategies = []
        returns = []
        for name, result in self.results.items():
            if name != 'Adaptive':  # Adaptiveは別途表示
                strategies.append(name)
                returns.append(result['total_return'] * 100)
        
        bars = plt.bar(strategies, returns)
        for bar, ret in zip(bars, returns):
            color = 'green' if ret > 0 else 'red'
            bar.set_color(color)
        plt.title('Strategy Returns (%)')
        plt.ylabel('Return %')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. オッズ帯別の期待値
        plt.subplot(2, 2, 2)
        odds_perf = self.odds_performance
        x = range(len(odds_perf))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], odds_perf['単勝期待値'], 
                width, label='Win EV', alpha=0.8)
        plt.bar([i + width/2 for i in x], odds_perf['複勝期待値'], 
                width, label='Place EV', alpha=0.8)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        plt.xticks(x, odds_perf['オッズ帯'], rotation=45)
        plt.xlabel('Odds Range')
        plt.ylabel('Expected Value')
        plt.title('Expected Value by Odds Range')
        plt.legend()
        
        # 3. オッズ帯別の勝率
        plt.subplot(2, 2, 3)
        plt.bar([i - width/2 for i in x], odds_perf['単勝率'] * 100, 
                width, label='Win Rate', alpha=0.8)
        plt.bar([i + width/2 for i in x], odds_perf['複勝率'] * 100, 
                width, label='Place Rate', alpha=0.8)
        plt.xticks(x, odds_perf['オッズ帯'], rotation=45)
        plt.xlabel('Odds Range')
        plt.ylabel('Win Rate %')
        plt.title('Win Rate by Odds Range')
        plt.legend()
        
        # 4. 資金推移（Standard戦略）
        plt.subplot(2, 2, 4)
        if 'Standard' in self.results and 'history' in self.results['Standard']:
            history = self.results['Standard']['history']
            capitals = [h['capital'] for h in history]
            plt.plot(capitals)
            plt.axhline(y=self.initial_capital, color='red', 
                       linestyle='--', alpha=0.5, label='Initial')
            plt.xlabel('Number of Bets')
            plt.ylabel('Capital (¥)')
            plt.title('Capital Evolution (Standard Strategy)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # オッズ帯別の詳細分析
        if 'Standard' in self.results and 'bets_by_odds' in self.results['Standard']:
            plt.figure(figsize=(10, 6))
            bets_by_odds = self.results['Standard']['bets_by_odds']
            
            odds_cats = []
            profits = []
            win_rates = []
            
            for cat, stats in bets_by_odds.items():
                odds_cats.append(cat)
                profits.append(stats['profit'])
                win_rates.append(stats['wins'] / stats['bets'] * 100 if stats['bets'] > 0 else 0)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Odds Category')
            ax1.set_ylabel('Profit (¥)', color=color)
            bars = ax1.bar(odds_cats, profits, alpha=0.7, color=color)
            
            # 利益がプラスのバーを緑、マイナスを赤に
            for bar, profit in zip(bars, profits):
                if profit > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xticks(rotation=45)
            
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Win Rate %', color=color)
            ax2.plot(odds_cats, win_rates, color=color, marker='o', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Profit and Win Rate by Odds Category')
            fig.tight_layout()
            plt.savefig('odds_category_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self):
        """結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 結果をJSONで保存
        save_data = {
            'timestamp': timestamp,
            'odds_performance': self.odds_performance.to_dict(),
            'strategy_results': {}
        }
        
        for name, result in self.results.items():
            save_data['strategy_results'][name] = {
                'total_return': result['total_return'],
                'total_profit': result.get('total_profit', 0)
            }
            
            if 'adaptive_thresholds' in result:
                save_data['strategy_results'][name]['adaptive_thresholds'] = result['adaptive_thresholds']
        
        os.makedirs('backtest_results', exist_ok=True)
        with open(f'backtest_results/advanced_backtest_{timestamp}.json', 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to backtest_results/advanced_backtest_{timestamp}.json")

def main():
    """メイン実行関数"""
    print("=== Advanced Backtest System ===")
    
    backtest = AdvancedBacktest()
    backtest.load_data()
    results = backtest.run_comprehensive_backtest()
    
    print("\n=== Summary ===")
    for strategy, result in results.items():
        print(f"\n{strategy} Strategy:")
        print(f"  Total Return: {result['total_return']:.2%}")
        if 'adaptive_thresholds' in result:
            print("  Adaptive Thresholds:")
            for odds_cat, info in result['adaptive_thresholds'].items():
                print(f"    {odds_cat}: threshold={info['threshold']:.2f}")
    
    backtest.visualize_results()
    backtest.save_results()
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()