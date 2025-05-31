#!/usr/bin/env python3
"""
バックテスト結果の詳細な可視化と分析
レースごとの推移、期待値分布、モデル検証
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class VisualBacktestAnalysis:
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
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {year}.xlsx - {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.prepare_features()
        
    def prepare_features(self):
        """特徴量エンジニアリング"""
        categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
    
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
    
    def train_simple_model(self, train_data):
        """シンプルなモデルの訓練（過学習を避ける）"""
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
            
        target = train_data['is_place']
        
        # より控えめなパラメータ
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # 少なめ
            'learning_rate': 0.1,  # 高め（イテレーション少なくする）
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'lambda_l1': 1.0,  # 強い正則化
            'lambda_l2': 1.0,
            'min_data_in_leaf': 50  # 葉の最小サンプル数を増やす
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=50,  # 少なめ
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def calculate_realistic_place_odds(self, win_odds):
        """より現実的な複勝オッズ推定"""
        # 実際の複勝オッズに近い値
        if win_odds <= 1.5:
            return 1.05  # 105円
        elif win_odds <= 2.0:
            return 1.1   # 110円
        elif win_odds <= 3.0:
            return 1.2   # 120円
        elif win_odds <= 5.0:
            return 1.3   # 130円
        elif win_odds <= 10.0:
            return 1.5   # 150円
        elif win_odds <= 20.0:
            return 1.8   # 180円
        elif win_odds <= 50.0:
            return 2.2   # 220円
        else:
            return 2.8   # 280円
    
    def run_detailed_backtest(self):
        """詳細なバックテスト実行"""
        # データ分割
        train_data = self.data[self.data['year'].isin(range(2014, 2021))]
        test_data = self.data[self.data['year'].isin(range(2021, 2024))]
        
        print(f"Train data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # モデル訓練
        print("\nTraining model with regularization...")
        model = self.train_simple_model(train_data)
        
        # バックテスト実行
        print("\nRunning detailed backtest...")
        capital_history = [self.initial_capital]
        ev_history = []
        bet_details = []
        race_count = 0
        
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races[:1000]):  # 最初の1000レースのみ
            if i % 100 == 0:
                print(f"Processing race {i}/1000...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            features = self.get_features(race_data)
            if features is None:
                continue
            
            predictions = model.predict(features)
            
            # 各馬の期待値を計算
            race_evs = []
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                win_odds = horse['オッズ']
                place_odds = self.calculate_realistic_place_odds(win_odds)
                place_prob = predictions[idx]
                ev = place_prob * place_odds
                
                race_evs.append({
                    'horse_no': horse['馬番'],
                    'odds': win_odds,
                    'place_odds': place_odds,
                    'probability': place_prob,
                    'expected_value': ev,
                    'actual_place': horse['着順'] <= 3
                })
            
            # 最高期待値の馬を選択
            best_horse = max(race_evs, key=lambda x: x['expected_value'])
            
            # ベッティング判断（期待値が1.05以上）
            if best_horse['expected_value'] > 1.05:
                bet_amount = capital_history[-1] * 0.01  # 資金の1%
                
                if best_horse['actual_place']:
                    # 的中
                    payout = bet_amount * best_horse['place_odds']
                    profit = payout - bet_amount
                else:
                    # 外れ
                    profit = -bet_amount
                
                new_capital = capital_history[-1] + profit
                capital_history.append(new_capital)
                
                bet_details.append({
                    'race_no': race_count,
                    'bet': True,
                    'expected_value': best_horse['expected_value'],
                    'actual_odds': best_horse['place_odds'],
                    'win': best_horse['actual_place'],
                    'profit': profit,
                    'capital': new_capital
                })
            else:
                # ベットしない
                capital_history.append(capital_history[-1])
                bet_details.append({
                    'race_no': race_count,
                    'bet': False,
                    'expected_value': best_horse['expected_value'],
                    'actual_odds': 0,
                    'win': False,
                    'profit': 0,
                    'capital': capital_history[-1]
                })
            
            ev_history.extend([h['expected_value'] for h in race_evs])
            race_count += 1
        
        self.capital_history = capital_history
        self.ev_history = ev_history
        self.bet_details = pd.DataFrame(bet_details)
        
        return {
            'final_capital': capital_history[-1],
            'total_return': (capital_history[-1] - self.initial_capital) / self.initial_capital,
            'total_races': race_count,
            'total_bets': len(self.bet_details[self.bet_details['bet'] == True]),
            'win_rate': self.bet_details[self.bet_details['bet'] == True]['win'].mean() if len(self.bet_details[self.bet_details['bet'] == True]) > 0 else 0,
            'avg_expected_value': np.mean(ev_history)
        }
    
    def visualize_results(self):
        """結果の詳細な可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 資金推移
        ax1 = axes[0, 0]
        races = range(len(self.capital_history))
        ax1.plot(races, self.capital_history, linewidth=2)
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial')
        ax1.fill_between(races, self.initial_capital, self.capital_history, 
                        where=(np.array(self.capital_history) > self.initial_capital), 
                        alpha=0.3, color='green', label='Profit')
        ax1.fill_between(races, self.initial_capital, self.capital_history, 
                        where=(np.array(self.capital_history) <= self.initial_capital), 
                        alpha=0.3, color='red', label='Loss')
        ax1.set_xlabel('Race Number')
        ax1.set_ylabel('Capital (¥)')
        ax1.set_title('Capital Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 期待値の分布
        ax2 = axes[0, 1]
        ax2.hist(self.ev_history, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Break-even (EV=1.0)')
        ax2.axvline(x=np.mean(self.ev_history), color='green', linestyle='-', linewidth=2, label=f'Mean EV={np.mean(self.ev_history):.3f}')
        ax2.set_xlabel('Expected Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Expected Value Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ベット判断の分析
        ax3 = axes[0, 2]
        bet_data = self.bet_details[self.bet_details['bet'] == True]
        if len(bet_data) > 0:
            colors = ['green' if w else 'red' for w in bet_data['win']]
            ax3.scatter(bet_data.index, bet_data['expected_value'], c=colors, alpha=0.6)
            ax3.axhline(y=1.05, color='blue', linestyle='--', label='Betting Threshold (1.05)')
            ax3.set_xlabel('Bet Number')
            ax3.set_ylabel('Expected Value')
            ax3.set_title('Betting Decisions (Green=Win, Red=Loss)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 累積利益
        ax4 = axes[1, 0]
        cumulative_profit = np.array(self.capital_history) - self.initial_capital
        ax4.plot(cumulative_profit, linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.fill_between(range(len(cumulative_profit)), 0, cumulative_profit, alpha=0.3)
        ax4.set_xlabel('Race Number')
        ax4.set_ylabel('Cumulative Profit (¥)')
        ax4.set_title('Cumulative Profit/Loss')
        ax4.grid(True, alpha=0.3)
        
        # 5. 期待値vs実際の結果
        ax5 = axes[1, 1]
        if len(bet_data) > 0:
            ev_bins = np.linspace(1.0, bet_data['expected_value'].max(), 10)
            win_rates_by_ev = []
            ev_centers = []
            
            for i in range(len(ev_bins)-1):
                mask = (bet_data['expected_value'] >= ev_bins[i]) & (bet_data['expected_value'] < ev_bins[i+1])
                if mask.sum() > 0:
                    win_rate = bet_data[mask]['win'].mean()
                    win_rates_by_ev.append(win_rate)
                    ev_centers.append((ev_bins[i] + ev_bins[i+1]) / 2)
            
            if win_rates_by_ev:
                ax5.bar(ev_centers, win_rates_by_ev, width=0.05, alpha=0.7)
                ax5.set_xlabel('Expected Value Range')
                ax5.set_ylabel('Actual Win Rate')
                ax5.set_title('Expected Value vs Actual Win Rate')
                ax5.grid(True, alpha=0.3)
        
        # 6. モデルの予測確率分布
        ax6 = axes[1, 2]
        # 的中馬と外れ馬の予測確率分布
        if len(bet_data) > 0:
            win_probs = self.bet_details[self.bet_details['win'] == True]['expected_value'] / self.bet_details[self.bet_details['win'] == True]['actual_odds']
            lose_probs = self.bet_details[(self.bet_details['bet'] == True) & (self.bet_details['win'] == False)]['expected_value'] / self.bet_details[(self.bet_details['bet'] == True) & (self.bet_details['win'] == False)]['actual_odds'].replace(0, 1)
            
            if len(win_probs) > 0:
                ax6.hist(win_probs, bins=20, alpha=0.5, label='Winners', color='green', density=True)
            if len(lose_probs) > 0:
                ax6.hist(lose_probs, bins=20, alpha=0.5, label='Losers', color='red', density=True)
            
            ax6.set_xlabel('Predicted Probability')
            ax6.set_ylabel('Density')
            ax6.set_title('Model Prediction Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 追加の分析図
        self.create_additional_analysis()
    
    def create_additional_analysis(self):
        """追加の詳細分析"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 連続的中・連続外れの分析
        ax1 = axes[0, 0]
        bet_data = self.bet_details[self.bet_details['bet'] == True]
        if len(bet_data) > 0:
            wins = bet_data['win'].values
            streaks = []
            current_streak = 0
            
            for i in range(1, len(wins)):
                if wins[i] == wins[i-1]:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 0
            
            if streaks:
                ax1.hist(streaks, bins=20, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Streak Length')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Win/Loss Streak Distribution')
                ax1.grid(True, alpha=0.3)
        
        # 2. 時系列での期待値の変化
        ax2 = axes[0, 1]
        window = 50
        if len(self.bet_details) > window:
            rolling_ev = self.bet_details['expected_value'].rolling(window).mean()
            ax2.plot(rolling_ev, linewidth=2)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Race Number')
            ax2.set_ylabel('Rolling Average EV (50 races)')
            ax2.set_title('Expected Value Trend Over Time')
            ax2.grid(True, alpha=0.3)
        
        # 3. ベット額とリターンの関係
        ax3 = axes[1, 0]
        bet_data = self.bet_details[self.bet_details['bet'] == True]
        if len(bet_data) > 0:
            returns = bet_data['profit'] / (bet_data['capital'].shift(1).fillna(self.initial_capital) * 0.01)
            ax3.scatter(bet_data['expected_value'], returns, alpha=0.5)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Expected Value')
            ax3.set_ylabel('Return Rate')
            ax3.set_title('Expected Value vs Actual Returns')
            ax3.grid(True, alpha=0.3)
        
        # 4. モデルの較正曲線
        ax4 = axes[1, 1]
        if len(bet_data) > 0:
            # 予測確率をビンに分割
            n_bins = 10
            fraction_of_positives, mean_predicted_value = [], []
            
            bet_data_copy = bet_data.copy()
            bet_data_copy['pred_prob'] = bet_data_copy['expected_value'] / bet_data_copy['actual_odds'].replace(0, 1)
            
            for i in range(n_bins):
                bin_lower = i / n_bins
                bin_upper = (i + 1) / n_bins
                mask = (bet_data_copy['pred_prob'] >= bin_lower) & (bet_data_copy['pred_prob'] < bin_upper)
                
                if mask.sum() > 0:
                    fraction_of_positives.append(bet_data_copy[mask]['win'].mean())
                    mean_predicted_value.append(bet_data_copy[mask]['pred_prob'].mean())
            
            if fraction_of_positives:
                ax4.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                ax4.plot(mean_predicted_value, fraction_of_positives, 'ro-', label='Model Calibration')
                ax4.set_xlabel('Mean Predicted Probability')
                ax4.set_ylabel('Fraction of Positives')
                ax4.set_title('Calibration Plot')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('additional_backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary_statistics(self, results):
        """詳細な統計サマリー"""
        print("\n=== 詳細統計 ===")
        print(f"総レース数: {results['total_races']}")
        print(f"総ベット数: {results['total_bets']}")
        print(f"ベット率: {results['total_bets']/results['total_races']:.1%}")
        print(f"勝率: {results['win_rate']:.1%}")
        print(f"平均期待値: {results['avg_expected_value']:.3f}")
        
        bet_data = self.bet_details[self.bet_details['bet'] == True]
        if len(bet_data) > 0:
            print(f"\n期待値の統計:")
            print(f"  最小: {bet_data['expected_value'].min():.3f}")
            print(f"  25%: {bet_data['expected_value'].quantile(0.25):.3f}")
            print(f"  中央値: {bet_data['expected_value'].median():.3f}")
            print(f"  75%: {bet_data['expected_value'].quantile(0.75):.3f}")
            print(f"  最大: {bet_data['expected_value'].max():.3f}")
            
            # Sharpe Ratio的な指標
            returns = bet_data['profit'] / (bet_data['capital'].shift(1).fillna(self.initial_capital) * 0.01)
            if len(returns) > 1 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # 年率換算
                print(f"\nリスク調整後リターン (Sharpe Ratio): {sharpe:.2f}")

def main():
    """メイン実行関数"""
    print("=== 詳細なバックテスト可視化分析 ===")
    
    analyzer = VisualBacktestAnalysis()
    analyzer.load_data()
    results = analyzer.run_detailed_backtest()
    
    print("\n=== 基本結果 ===")
    print(f"初期資産: ¥{analyzer.initial_capital:,}")
    print(f"最終資産: ¥{results['final_capital']:,.0f}")
    print(f"総リターン: {results['total_return']:.2%}")
    
    analyzer.print_summary_statistics(results)
    
    print("\n可視化を作成中...")
    analyzer.visualize_results()
    
    print("\n分析完了！")
    print("- detailed_backtest_analysis.png: メイン分析グラフ")
    print("- additional_backtest_analysis.png: 追加分析グラフ")
    
    # 警告
    if results['avg_expected_value'] > 1.5:
        print("\n⚠️ 警告: 平均期待値が異常に高いです。")
        print("モデルが過学習している可能性があります。")
        print("より厳しい正則化や、シンプルなモデルの使用を検討してください。")

if __name__ == "__main__":
    main()