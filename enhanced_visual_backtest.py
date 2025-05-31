#!/usr/bin/env python3
"""
Enhanced Visual Backtest with Actual Odds Data
This script performs backtesting using actual payout data and provides
comprehensive visualization of results.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Japanese font settings
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# LightGBM error mitigation
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class EnhancedVisualBacktest:
    """Enhanced backtesting system with actual odds data and visualization"""
    
    def __init__(self, betting_fraction: float = 0.01, min_expected_value: float = 1.05):
        self.betting_fraction = betting_fraction
        self.min_expected_value = min_expected_value
        self.initial_capital = 1_000_000
        self.results = {}
        
    def load_data_with_payout(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Load data from data_with_payout directory"""
        print(f"Loading data with payout information from {start_year} to {end_year}...")
        
        dfs = []
        data_dir = Path("data_with_payout")
        
        for year in range(start_year, end_year + 1):
            # Try different file patterns
            file_patterns = [
                f"{year}_with_payout.xlsx",
                f"{year}.xlsx",
                f"{year}_*.xlsx"
            ]
            
            file_loaded = False
            for pattern in file_patterns:
                files = list(data_dir.glob(pattern))
                if files:
                    # Use the first matching file
                    file_path = files[0]
                    try:
                        df = pd.read_excel(file_path)
                        df['year'] = year
                        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                        df = df.dropna(subset=['着順'])
                        print(f"Loaded {file_path.name}: {len(df)} rows")
                        dfs.append(df)
                        file_loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            
            if not file_loaded:
                # Fallback to regular data directory
                try:
                    file_path = Path(f"data/{year}.xlsx")
                    if file_path.exists():
                        df = pd.read_excel(file_path)
                        df['year'] = year
                        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                        df = df.dropna(subset=['着順'])
                        print(f"Loaded {file_path.name} from data directory: {len(df)} rows")
                        dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load data for {year} - {e}")
        
        if not dfs:
            raise ValueError("No data files were loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total rows loaded: {len(self.data)}")
        
        # Check for payout columns
        payout_columns = ['単勝払戻', '複勝払戻', '馬連払戻', '馬単払戻', 
                         '三連複払戻', '三連単払戻', 'ワイド払戻']
        available_payouts = [col for col in payout_columns if col in self.data.columns]
        print(f"Available payout columns: {available_payouts}")
        
        self.prepare_features()
        return self.data
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Categorical encoding
        categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        # Numeric columns
        numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Target variables
        self.data['is_win'] = (self.data['着順'] == 1).astype(int)
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
        self.data['is_exacta'] = (self.data['着順'] <= 2).astype(int)
        
        # Calculate actual rates
        print(f"Win rate: {self.data['is_win'].mean():.1%}")
        print(f"Place rate: {self.data['is_place'].mean():.1%}")
        
    def get_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get feature matrix"""
        feature_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化',
                          '性', '馬場', '天気', '芝・ダート', '場名']
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                features.append(data[col].values)
        
        if len(features) == 0:
            return None
            
        return np.column_stack(features)
    
    def train_model(self, train_data: pd.DataFrame, target_col: str = 'is_place'):
        """Train LightGBM model with conservative parameters"""
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
            
        target = train_data[target_col]
        
        # Calculate class weights
        pos_weight = len(target[target == 0]) / len(target[target == 1])
        
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': pos_weight,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 50
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def get_actual_place_odds(self, race_data: pd.DataFrame, horse_no: int) -> float:
        """Get actual place odds from payout data if available"""
        if '複勝払戻' in race_data.columns:
            # Try to parse actual payout data
            place_payout = race_data[race_data['馬番'] == horse_no]['複勝払戻'].iloc[0]
            if pd.notna(place_payout) and place_payout > 0:
                return place_payout / 100  # Convert to odds format
        
        # Fallback to estimation based on win odds
        win_odds = race_data[race_data['馬番'] == horse_no]['オッズ'].iloc[0]
        return self.estimate_place_odds(win_odds)
    
    def estimate_place_odds(self, win_odds: float) -> float:
        """Estimate place odds from win odds"""
        if win_odds <= 2.0:
            return 1.1
        elif win_odds <= 5.0:
            return 1.2
        elif win_odds <= 10.0:
            return 1.4
        elif win_odds <= 20.0:
            return 1.7
        elif win_odds <= 50.0:
            return 2.2
        else:
            return 3.0
    
    def run_backtest(self, train_years: List[int], test_years: List[int]) -> Dict:
        """Run comprehensive backtest with detailed tracking"""
        print(f"\nRunning backtest...")
        print(f"Train years: {train_years}")
        print(f"Test years: {test_years}")
        
        # Split data
        train_data = self.data[self.data['year'].isin(train_years)]
        test_data = self.data[self.data['year'].isin(test_years)]
        
        print(f"Train data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # Train model
        print("\nTraining model...")
        model = self.train_model(train_data)
        
        # Initialize tracking variables
        capital = self.initial_capital
        capital_history = [capital]
        trade_history = []
        monthly_results = {}
        
        # Process each race
        unique_races = test_data['race_id'].unique()
        print(f"\nProcessing {len(unique_races)} races...")
        
        for i, race_id in enumerate(unique_races):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(unique_races)} races...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            # Skip races with too few horses
            if len(race_data) < 5:
                continue
            
            # Get predictions
            features = self.get_features(race_data)
            if features is None:
                continue
            
            predictions = model.predict(features)
            
            # Calculate expected values
            race_evs = []
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                place_odds = self.get_actual_place_odds(race_data, horse['馬番'])
                place_prob = predictions[idx]
                ev = place_prob * place_odds
                
                race_evs.append({
                    'idx': idx,
                    'horse_no': horse['馬番'],
                    'win_odds': horse['オッズ'],
                    'place_odds': place_odds,
                    'probability': place_prob,
                    'expected_value': ev,
                    'actual_place': horse['着順'] <= 3,
                    'popularity': int(horse['人気'])
                })
            
            # Select best expected value
            best_horse = max(race_evs, key=lambda x: x['expected_value'])
            
            # Make betting decision
            if best_horse['expected_value'] >= self.min_expected_value:
                bet_amount = capital * self.betting_fraction
                
                if best_horse['actual_place']:
                    # Win
                    payout = bet_amount * best_horse['place_odds']
                    profit = payout - bet_amount
                else:
                    # Loss
                    profit = -bet_amount
                
                capital += profit
                capital_history.append(capital)
                
                # Record trade
                trade_history.append({
                    'race_id': race_id,
                    'race_no': i,
                    'horse_no': best_horse['horse_no'],
                    'popularity': best_horse['popularity'],
                    'win_odds': best_horse['win_odds'],
                    'place_odds': best_horse['place_odds'],
                    'probability': best_horse['probability'],
                    'expected_value': best_horse['expected_value'],
                    'bet_amount': bet_amount,
                    'win': best_horse['actual_place'],
                    'profit': profit,
                    'capital': capital,
                    'return_rate': profit / bet_amount
                })
                
                # Track monthly results
                if 'date' in race_data.columns:
                    month_key = pd.to_datetime(race_data.iloc[0]['date']).strftime('%Y-%m')
                    if month_key not in monthly_results:
                        monthly_results[month_key] = {'bets': 0, 'wins': 0, 'profit': 0}
                    monthly_results[month_key]['bets'] += 1
                    if best_horse['actual_place']:
                        monthly_results[month_key]['wins'] += 1
                    monthly_results[month_key]['profit'] += profit
            
            # Check for bankruptcy
            if capital <= 0:
                print(f"Bankrupt at race {i}")
                break
        
        # Store results
        self.capital_history = capital_history
        self.trade_history = pd.DataFrame(trade_history)
        self.monthly_results = monthly_results
        
        # Calculate final metrics
        total_trades = len(self.trade_history)
        winning_trades = len(self.trade_history[self.trade_history['win'] == True])
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_bet_amount': self.trade_history['bet_amount'].mean() if total_trades > 0 else 0,
            'avg_profit': self.trade_history['profit'].mean() if total_trades > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(capital_history),
            'sharpe_ratio': self.calculate_sharpe_ratio()
        }
    
    def calculate_max_drawdown(self, capital_history: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(capital_history) < 2:
            return 0
        
        peak = capital_history[0]
        max_dd = 0
        
        for value in capital_history[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trade_history) < 2:
            return 0
        
        returns = self.trade_history['return_rate']
        if returns.std() == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def create_visualizations(self, results: Dict, output_dir: str = "backtest_results"):
        """Create comprehensive visualization of results"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Capital evolution
        ax1 = plt.subplot(3, 3, 1)
        self.plot_capital_evolution(ax1)
        
        # 2. Monthly performance
        ax2 = plt.subplot(3, 3, 2)
        self.plot_monthly_performance(ax2)
        
        # 3. Expected value distribution
        ax3 = plt.subplot(3, 3, 3)
        self.plot_ev_distribution(ax3)
        
        # 4. Win rate by popularity
        ax4 = plt.subplot(3, 3, 4)
        self.plot_winrate_by_popularity(ax4)
        
        # 5. Profit distribution
        ax5 = plt.subplot(3, 3, 5)
        self.plot_profit_distribution(ax5)
        
        # 6. Rolling performance
        ax6 = plt.subplot(3, 3, 6)
        self.plot_rolling_performance(ax6)
        
        # 7. Drawdown analysis
        ax7 = plt.subplot(3, 3, 7)
        self.plot_drawdown(ax7)
        
        # 8. Bet amount distribution
        ax8 = plt.subplot(3, 3, 8)
        self.plot_bet_distribution(ax8)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        self.plot_summary_stats(ax9, results)
        
        plt.tight_layout()
        plt.savefig(output_path / 'backtest_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional detailed plots
        self.create_detailed_analysis_plots(output_path)
        
        print(f"\nVisualizations saved to {output_path}/")
    
    def plot_capital_evolution(self, ax):
        """Plot capital evolution over time"""
        ax.plot(self.capital_history, linewidth=2, color='darkblue')
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Fill profit/loss areas
        x = range(len(self.capital_history))
        ax.fill_between(x, self.initial_capital, self.capital_history,
                       where=(np.array(self.capital_history) > self.initial_capital),
                       alpha=0.3, color='green', label='Profit')
        ax.fill_between(x, self.initial_capital, self.capital_history,
                       where=(np.array(self.capital_history) <= self.initial_capital),
                       alpha=0.3, color='red', label='Loss')
        
        ax.set_xlabel('Number of Bets')
        ax.set_ylabel('Capital (¥)')
        ax.set_title('Capital Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_monthly_performance(self, ax):
        """Plot monthly performance"""
        if not self.monthly_results:
            ax.text(0.5, 0.5, 'No monthly data available', ha='center', va='center')
            return
        
        months = sorted(self.monthly_results.keys())
        profits = [self.monthly_results[m]['profit'] for m in months]
        
        colors = ['green' if p > 0 else 'red' for p in profits]
        bars = ax.bar(range(len(months)), profits, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m[-2:] for m in months], rotation=45)
        ax.set_xlabel('Month')
        ax.set_ylabel('Profit (¥)')
        ax.set_title('Monthly Profit/Loss')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def plot_ev_distribution(self, ax):
        """Plot expected value distribution"""
        if len(self.trade_history) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return
        
        ev_values = self.trade_history['expected_value']
        
        ax.hist(ev_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Break-even (EV=1.0)')
        ax.axvline(x=self.min_expected_value, color='green', linestyle='--', linewidth=2, 
                  label=f'Min EV ({self.min_expected_value})')
        ax.axvline(x=ev_values.mean(), color='orange', linestyle='-', linewidth=2,
                  label=f'Mean EV ({ev_values.mean():.3f})')
        
        ax.set_xlabel('Expected Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Expected Value Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_winrate_by_popularity(self, ax):
        """Plot win rate by horse popularity"""
        if len(self.trade_history) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return
        
        # Group by popularity
        popularity_stats = self.trade_history.groupby('popularity').agg({
            'win': ['count', 'sum', 'mean']
        }).reset_index()
        
        popularity_stats.columns = ['popularity', 'count', 'wins', 'win_rate']
        
        # Only show popularities with at least 10 bets
        popularity_stats = popularity_stats[popularity_stats['count'] >= 10]
        
        if len(popularity_stats) > 0:
            ax.bar(popularity_stats['popularity'], popularity_stats['win_rate'], alpha=0.7)
            ax.set_xlabel('Horse Popularity')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate by Horse Popularity')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for idx, row in popularity_stats.iterrows():
                ax.text(row['popularity'], row['win_rate'] + 0.01, 
                       f"n={row['count']}", ha='center', fontsize=8)
    
    def plot_profit_distribution(self, ax):
        """Plot profit distribution"""
        if len(self.trade_history) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return
        
        profits = self.trade_history['profit']
        
        # Separate wins and losses
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        
        bins = np.linspace(profits.min(), profits.max(), 50)
        
        ax.hist(losses, bins=bins, alpha=0.5, label=f'Losses (n={len(losses)})', color='red')
        ax.hist(wins, bins=bins, alpha=0.5, label=f'Wins (n={len(wins)})', color='green')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=profits.mean(), color='blue', linestyle='--', linewidth=2,
                  label=f'Mean ({profits.mean():.0f})')
        
        ax.set_xlabel('Profit (¥)')
        ax.set_ylabel('Frequency')
        ax.set_title('Profit Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_rolling_performance(self, ax):
        """Plot rolling performance metrics"""
        if len(self.trade_history) < 50:
            ax.text(0.5, 0.5, 'Insufficient data for rolling analysis', ha='center', va='center')
            return
        
        window = 50
        
        # Calculate rolling metrics
        rolling_winrate = self.trade_history['win'].rolling(window).mean()
        rolling_avg_profit = self.trade_history['profit'].rolling(window).mean()
        
        ax2 = ax.twinx()
        
        # Plot win rate
        line1 = ax.plot(rolling_winrate.index, rolling_winrate, 'b-', 
                       label=f'{window}-bet Rolling Win Rate')
        ax.set_ylabel('Win Rate', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot average profit
        line2 = ax2.plot(rolling_avg_profit.index, rolling_avg_profit, 'r-',
                        label=f'{window}-bet Rolling Avg Profit')
        ax2.set_ylabel('Average Profit (¥)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        ax.set_xlabel('Bet Number')
        ax.set_title('Rolling Performance Metrics')
        ax.grid(True, alpha=0.3)
    
    def plot_drawdown(self, ax):
        """Plot drawdown analysis"""
        if len(self.capital_history) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for drawdown analysis', ha='center', va='center')
            return
        
        # Calculate drawdown series
        capital_series = pd.Series(self.capital_history)
        rolling_max = capital_series.expanding().max()
        drawdown = (capital_series - rolling_max) / rolling_max
        
        ax.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red')
        ax.plot(drawdown, color='darkred', linewidth=2)
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()
        ax.plot(max_dd_idx, max_dd, 'ro', markersize=10)
        ax.annotate(f'Max DD: {max_dd:.1%}', xy=(max_dd_idx, max_dd),
                   xytext=(max_dd_idx + len(drawdown) * 0.05, max_dd - 0.05),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Bet Number')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0.05)
    
    def plot_bet_distribution(self, ax):
        """Plot bet amount distribution"""
        if len(self.trade_history) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return
        
        bet_amounts = self.trade_history['bet_amount']
        
        ax.hist(bet_amounts, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=bet_amounts.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: ¥{bet_amounts.mean():,.0f}')
        ax.axvline(x=bet_amounts.median(), color='green', linestyle='--', linewidth=2,
                  label=f'Median: ¥{bet_amounts.median():,.0f}')
        
        ax.set_xlabel('Bet Amount (¥)')
        ax.set_ylabel('Frequency')
        ax.set_title('Bet Amount Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_summary_stats(self, ax, results):
        """Plot summary statistics table"""
        ax.axis('off')
        
        # Create summary data
        summary_data = [
            ['Initial Capital', f'¥{results["initial_capital"]:,.0f}'],
            ['Final Capital', f'¥{results["final_capital"]:,.0f}'],
            ['Total Return', f'{results["total_return"]:.2%}'],
            ['Total Trades', f'{results["total_trades"]:,}'],
            ['Win Rate', f'{results["win_rate"]:.1%}'],
            ['Max Drawdown', f'{results["max_drawdown"]:.1%}'],
            ['Sharpe Ratio', f'{results["sharpe_ratio"]:.2f}'],
            ['Avg Bet Amount', f'¥{results["avg_bet_amount"]:,.0f}'],
            ['Avg Profit/Trade', f'¥{results["avg_profit"]:,.0f}']
        ]
        
        # Create table
        table = ax.table(cellText=summary_data, cellLoc='left',
                        loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(summary_data)):
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
            
            # Color code performance metrics
            if i == 2:  # Total Return
                color = 'lightgreen' if results['total_return'] > 0 else 'lightcoral'
                table[(i, 1)].set_facecolor(color)
        
        ax.set_title('Summary Statistics', fontsize=12, weight='bold', pad=20)
    
    def create_detailed_analysis_plots(self, output_path: Path):
        """Create additional detailed analysis plots"""
        # Expected Value vs Actual Returns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. EV vs Actual Returns scatter
        ax1 = axes[0, 0]
        if len(self.trade_history) > 0:
            colors = ['green' if w else 'red' for w in self.trade_history['win']]
            ax1.scatter(self.trade_history['expected_value'], 
                       self.trade_history['return_rate'],
                       c=colors, alpha=0.5)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.axvline(x=1.0, color='black', linestyle='-', alpha=0.5)
            ax1.set_xlabel('Expected Value')
            ax1.set_ylabel('Actual Return Rate')
            ax1.set_title('Expected Value vs Actual Returns')
            ax1.grid(True, alpha=0.3)
        
        # 2. Calibration plot
        ax2 = axes[0, 1]
        self.plot_calibration_curve(ax2)
        
        # 3. Performance by odds range
        ax3 = axes[1, 0]
        self.plot_performance_by_odds_range(ax3)
        
        # 4. Time of day analysis (if available)
        ax4 = axes[1, 1]
        self.plot_time_analysis(ax4)
        
        plt.tight_layout()
        plt.savefig(output_path / 'backtest_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curve(self, ax):
        """Plot model calibration curve"""
        if len(self.trade_history) < 50:
            ax.text(0.5, 0.5, 'Insufficient data for calibration plot', ha='center', va='center')
            return
        
        # Bin predictions
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        actual_prob = []
        predicted_prob = []
        
        for i in range(n_bins):
            mask = (self.trade_history['probability'] >= bin_edges[i]) & \
                   (self.trade_history['probability'] < bin_edges[i + 1])
            
            if mask.sum() > 10:  # Only use bins with enough samples
                actual_prob.append(self.trade_history[mask]['win'].mean())
                predicted_prob.append(self.trade_history[mask]['probability'].mean())
        
        if len(actual_prob) > 0:
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax.plot(predicted_prob, actual_prob, 'ro-', linewidth=2, markersize=8,
                   label='Model Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Actual Win Rate')
            ax.set_title('Model Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_performance_by_odds_range(self, ax):
        """Plot performance by odds range"""
        if len(self.trade_history) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return
        
        # Define odds ranges
        odds_ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
        range_labels = ['1-5', '5-10', '10-20', '20-50', '50-100', '100+']
        
        win_rates = []
        counts = []
        avg_profits = []
        
        for low, high in odds_ranges:
            mask = (self.trade_history['win_odds'] > low) & (self.trade_history['win_odds'] <= high)
            if mask.sum() > 0:
                win_rates.append(self.trade_history[mask]['win'].mean())
                counts.append(mask.sum())
                avg_profits.append(self.trade_history[mask]['profit'].mean())
            else:
                win_rates.append(0)
                counts.append(0)
                avg_profits.append(0)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        ax2 = ax.twinx()
        
        # Bar chart for win rate
        bars1 = ax.bar(x - width/2, win_rates, width, label='Win Rate', alpha=0.7)
        
        # Bar chart for average profit
        bars2 = ax2.bar(x + width/2, avg_profits, width, label='Avg Profit', 
                        color='orange', alpha=0.7)
        
        ax.set_xlabel('Win Odds Range')
        ax.set_ylabel('Win Rate')
        ax2.set_ylabel('Average Profit (¥)')
        ax.set_title('Performance by Odds Range')
        ax.set_xticks(x)
        ax.set_xticklabels(range_labels)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={count}', ha='center', fontsize=8)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax.grid(True, alpha=0.3)
    
    def plot_time_analysis(self, ax):
        """Plot performance by time of day or day of week if available"""
        ax.text(0.5, 0.5, 'Time analysis not available\n(requires timestamp data)', 
               ha='center', va='center', fontsize=12)
        ax.set_title('Time-based Analysis')
        ax.axis('off')
    
    def save_results_to_json(self, results: Dict, output_dir: str = "backtest_results"):
        """Save detailed results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Prepare data for JSON serialization
        json_data = {
            'summary': results,
            'configuration': {
                'betting_fraction': self.betting_fraction,
                'min_expected_value': self.min_expected_value,
                'initial_capital': self.initial_capital
            },
            'trade_history': self.trade_history.to_dict('records') if len(self.trade_history) > 0 else [],
            'monthly_results': self.monthly_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        with open(output_path / 'backtest_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save trade history to CSV for easier analysis
        if len(self.trade_history) > 0:
            self.trade_history.to_csv(output_path / 'trade_history.csv', index=False, encoding='utf-8-sig')
        
        print(f"Results saved to {output_path}/backtest_results.json")
        print(f"Trade history saved to {output_path}/trade_history.csv")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced Visual Backtest with Actual Odds')
    
    parser.add_argument('--start-year', type=int, default=2022,
                       help='Start year for backtesting (default: 2022)')
    parser.add_argument('--end-year', type=int, default=2025,
                       help='End year for backtesting (default: 2025)')
    parser.add_argument('--train-years', type=int, nargs='+', default=[2022],
                       help='Years to use for training (default: [2022])')
    parser.add_argument('--test-years', type=int, nargs='+', default=[2023, 2024, 2025],
                       help='Years to use for testing (default: [2023, 2024, 2025])')
    parser.add_argument('--bet-fraction', type=float, default=0.01,
                       help='Fraction of capital to bet (default: 0.01)')
    parser.add_argument('--min-ev', type=float, default=1.05,
                       help='Minimum expected value for betting (default: 1.05)')
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                       help='Output directory for results (default: backtest_results)')
    
    args = parser.parse_args()
    
    print("=== Enhanced Visual Backtest with Actual Odds ===")
    print(f"Configuration:")
    print(f"  Data period: {args.start_year} - {args.end_year}")
    print(f"  Train years: {args.train_years}")
    print(f"  Test years: {args.test_years}")
    print(f"  Bet fraction: {args.bet_fraction:.1%}")
    print(f"  Minimum EV: {args.min_ev}")
    
    # Initialize backtest system
    backtest = EnhancedVisualBacktest(
        betting_fraction=args.bet_fraction,
        min_expected_value=args.min_ev
    )
    
    # Load data
    try:
        backtest.load_data_with_payout(args.start_year, args.end_year)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run backtest
    try:
        results = backtest.run_backtest(args.train_years, args.test_years)
    except Exception as e:
        print(f"Error running backtest: {e}")
        return
    
    # Display results
    print("\n=== Backtest Results ===")
    print(f"Initial Capital: ¥{results['initial_capital']:,}")
    print(f"Final Capital: ¥{results['final_capital']:,.0f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']:,}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Average Bet: ¥{results['avg_bet_amount']:,.0f}")
    print(f"Average Profit per Trade: ¥{results['avg_profit']:,.0f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    backtest.create_visualizations(results, args.output_dir)
    
    # Save results
    backtest.save_results_to_json(results, args.output_dir)
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to {args.output_dir}/")
    
    # Risk warnings
    if results['total_return'] < 0:
        print("\n⚠️ Warning: Negative returns detected")
        print("Consider adjusting strategy parameters or reviewing model performance")
    
    if results['max_drawdown'] > 0.2:
        print("\n⚠️ Warning: High maximum drawdown (>20%)")
        print("The strategy may be too risky for practical use")
    
    if results['sharpe_ratio'] < 0.5:
        print("\n⚠️ Warning: Low Sharpe ratio (<0.5)")
        print("Risk-adjusted returns are poor")


if __name__ == "__main__":
    main()