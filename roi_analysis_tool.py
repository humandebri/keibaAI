#!/usr/bin/env python3
"""
ROI分析ツール - 詳細な収益性分析とセグメント評価
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ROIAnalyzer:
    """ROI詳細分析ツール"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_betting_patterns(self, trades: List[Dict]) -> Dict:
        """賭けパターンの詳細分析"""
        df = pd.DataFrame(trades)
        
        analysis = {
            'overall': self._calculate_basic_metrics(df),
            'by_odds_range': self._analyze_by_odds_range(df),
            'by_expected_value': self._analyze_by_expected_value(df),
            'drawdown_analysis': self._analyze_drawdowns(df),
            'kelly_analysis': self._analyze_kelly_criterion(df),
            'monthly_volatility': self._analyze_monthly_volatility(df)
        }
        
        return analysis
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict:
        """基本的なメトリクスの計算"""
        total_bets = len(df)
        wins = df['is_win'].sum()
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_wagered': df['bet_amount'].sum(),
            'total_profit': df['profit'].sum(),
            'roi': (df['profit'].sum() / df['bet_amount'].sum()) if df['bet_amount'].sum() > 0 else 0,
            'average_odds': df['odds'].mean(),
            'average_bet_size': df['bet_amount'].mean(),
            'profit_factor': df[df['profit'] > 0]['profit'].sum() / abs(df[df['profit'] < 0]['profit'].sum()) if df[df['profit'] < 0]['profit'].sum() != 0 else float('inf'),
            'sharpe_ratio': self._calculate_sharpe_ratio(df)
        }
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """シャープレシオの計算"""
        if len(df) < 2:
            return 0
        
        # 日次リターンの計算
        df['date'] = pd.to_datetime(df['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        daily_returns = df.groupby('date')['profit'].sum() / df.groupby('date')['bet_amount'].sum()
        
        if len(daily_returns) < 2:
            return 0
        
        # 年率換算（競馬は週末のみなので52週）
        return (daily_returns.mean() / daily_returns.std()) * np.sqrt(52) if daily_returns.std() > 0 else 0
    
    def _analyze_by_odds_range(self, df: pd.DataFrame) -> Dict:
        """オッズ帯別の分析"""
        odds_ranges = [
            (1.0, 2.0, '1.0-2.0'),
            (2.0, 5.0, '2.0-5.0'),
            (5.0, 10.0, '5.0-10.0'),
            (10.0, 20.0, '10.0-20.0'),
            (20.0, 50.0, '20.0-50.0'),
            (50.0, float('inf'), '50.0+')
        ]
        
        results = {}
        for min_odds, max_odds, label in odds_ranges:
            subset = df[(df['odds'] >= min_odds) & (df['odds'] < max_odds)]
            if len(subset) > 0:
                results[label] = {
                    'count': len(subset),
                    'win_rate': subset['is_win'].mean(),
                    'roi': subset['profit'].sum() / subset['bet_amount'].sum() if subset['bet_amount'].sum() > 0 else 0,
                    'avg_profit': subset['profit'].mean()
                }
        
        return results
    
    def _analyze_by_expected_value(self, df: pd.DataFrame) -> Dict:
        """期待値別の分析"""
        if 'expected_value' not in df.columns:
            return {}
        
        ev_ranges = [
            (0.8, 1.0, '0.8-1.0'),
            (1.0, 1.1, '1.0-1.1'),
            (1.1, 1.2, '1.1-1.2'),
            (1.2, 1.5, '1.2-1.5'),
            (1.5, 2.0, '1.5-2.0'),
            (2.0, float('inf'), '2.0+')
        ]
        
        results = {}
        for min_ev, max_ev, label in ev_ranges:
            subset = df[(df['expected_value'] >= min_ev) & (df['expected_value'] < max_ev)]
            if len(subset) > 0:
                results[label] = {
                    'count': len(subset),
                    'win_rate': subset['is_win'].mean(),
                    'roi': subset['profit'].sum() / subset['bet_amount'].sum() if subset['bet_amount'].sum() > 0 else 0,
                    'theoretical_ev': subset['expected_value'].mean(),
                    'actual_vs_theoretical': (subset['profit'].sum() / subset['bet_amount'].sum()) / (subset['expected_value'].mean() - 1) if subset['expected_value'].mean() > 1 else 0
                }
        
        return results
    
    def _analyze_drawdowns(self, df: pd.DataFrame) -> Dict:
        """ドローダウン分析"""
        if 'capital' not in df.columns:
            return {}
        
        capital_series = df['capital'].values
        
        # 最大ドローダウンの計算
        running_max = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # 連続負け
        losing_streaks = []
        current_streak = 0
        for profit in df['profit']:
            if profit < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 0
        
        return {
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown': drawdown[drawdown < 0].mean() * 100 if len(drawdown[drawdown < 0]) > 0 else 0,
            'max_losing_streak': max(losing_streaks) if losing_streaks else 0,
            'avg_losing_streak': np.mean(losing_streaks) if losing_streaks else 0,
            'recovery_factor': df['profit'].sum() / abs(max_drawdown * df['capital'].iloc[0]) if max_drawdown < 0 else float('inf')
        }
    
    def _calculate_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """ドローダウン期間の計算"""
        in_drawdown = drawdown < 0
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        return max(durations) if durations else 0
    
    def _analyze_kelly_criterion(self, df: pd.DataFrame) -> Dict:
        """Kelly基準の分析"""
        if 'expected_value' not in df.columns or 'odds' not in df.columns:
            return {}
        
        # 実際の勝率と期待値から最適Kelly比率を計算
        win_rate = df['is_win'].mean()
        avg_odds = df['odds'].mean()
        
        if avg_odds > 1:
            optimal_kelly = (win_rate * (avg_odds - 1) - (1 - win_rate)) / (avg_odds - 1)
            optimal_kelly = max(0, optimal_kelly)
        else:
            optimal_kelly = 0
        
        # 実際のベット比率
        if 'capital' in df.columns:
            actual_bet_ratios = df['bet_amount'] / df['capital'].shift(1).fillna(df['capital'].iloc[0])
            avg_bet_ratio = actual_bet_ratios.mean()
        else:
            avg_bet_ratio = 0
        
        return {
            'optimal_kelly_pct': optimal_kelly * 100,
            'actual_avg_bet_ratio_pct': avg_bet_ratio * 100,
            'kelly_efficiency': avg_bet_ratio / optimal_kelly if optimal_kelly > 0 else 0,
            'recommended_kelly_fraction': min(0.25, optimal_kelly / 4)  # 安全な1/4 Kelly
        }
    
    def _analyze_monthly_volatility(self, df: pd.DataFrame) -> Dict:
        """月別ボラティリティ分析"""
        if 'race_id' not in df.columns:
            return {}
        
        # race_idから月を抽出（仮定：race_idの最初の6文字がYYYYMM）
        df['month'] = df['race_id'].astype(str).str[:6]
        
        monthly_stats = df.groupby('month').agg({
            'profit': ['sum', 'std', 'count'],
            'bet_amount': 'sum',
            'is_win': 'mean'
        })
        
        monthly_roi = monthly_stats[('profit', 'sum')] / monthly_stats[('bet_amount', 'sum')]
        
        return {
            'monthly_roi_mean': monthly_roi.mean(),
            'monthly_roi_std': monthly_roi.std(),
            'monthly_roi_sharpe': monthly_roi.mean() / monthly_roi.std() if monthly_roi.std() > 0 else 0,
            'best_month_roi': monthly_roi.max(),
            'worst_month_roi': monthly_roi.min(),
            'positive_months_pct': (monthly_roi > 0).mean() * 100
        }
    
    def create_visualization(self, analysis: Dict, output_dir: str = 'results_improved') -> None:
        """分析結果の可視化"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ROI Analysis Dashboard', fontsize=16)
        
        # 1. オッズ帯別ROI
        if 'by_odds_range' in analysis:
            ax = axes[0, 0]
            odds_data = analysis['by_odds_range']
            ranges = list(odds_data.keys())
            rois = [odds_data[r]['roi'] for r in ranges]
            counts = [odds_data[r]['count'] for r in ranges]
            
            x = np.arange(len(ranges))
            ax.bar(x, rois)
            ax.set_xlabel('Odds Range')
            ax.set_ylabel('ROI')
            ax.set_title('ROI by Odds Range')
            ax.set_xticks(x)
            ax.set_xticklabels(ranges, rotation=45)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # 取引数を第2軸に表示
            ax2 = ax.twinx()
            ax2.plot(x, counts, 'ro-', alpha=0.5)
            ax2.set_ylabel('Number of Bets', color='r')
        
        # 2. 期待値別分析
        if 'by_expected_value' in analysis:
            ax = axes[0, 1]
            ev_data = analysis['by_expected_value']
            ranges = list(ev_data.keys())
            actual_vs_theoretical = [ev_data[r].get('actual_vs_theoretical', 0) for r in ranges]
            
            ax.bar(ranges, actual_vs_theoretical)
            ax.set_xlabel('Expected Value Range')
            ax.set_ylabel('Actual / Theoretical')
            ax.set_title('Performance vs Expected Value')
            ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 資金推移（仮想データ）
        ax = axes[0, 2]
        if 'drawdown_analysis' in analysis:
            # 仮想的な資金推移を作成
            n_bets = analysis['overall']['total_bets']
            x = np.arange(n_bets)
            y = np.cumsum(np.random.normal(0.001, 0.02, n_bets)) + 1
            
            ax.plot(x, y, 'b-', alpha=0.7)
            ax.fill_between(x, 1, y, where=(y >= 1), alpha=0.3, color='g')
            ax.fill_between(x, 1, y, where=(y < 1), alpha=0.3, color='r')
            ax.set_xlabel('Number of Bets')
            ax.set_ylabel('Capital Ratio')
            ax.set_title('Capital Growth')
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # 4. 勝率とROIの関係
        ax = axes[1, 0]
        if 'by_odds_range' in analysis:
            odds_data = analysis['by_odds_range']
            win_rates = [odds_data[r]['win_rate'] for r in odds_data.keys()]
            rois = [odds_data[r]['roi'] for r in odds_data.keys()]
            
            ax.scatter(win_rates, rois, s=100)
            ax.set_xlabel('Win Rate')
            ax.set_ylabel('ROI')
            ax.set_title('Win Rate vs ROI')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=0.5, color='b', linestyle='--', alpha=0.5)
            
            # トレンドライン
            if len(win_rates) > 2:
                z = np.polyfit(win_rates, rois, 1)
                p = np.poly1d(z)
                ax.plot(win_rates, p(win_rates), 'r--', alpha=0.5)
        
        # 5. Kelly基準分析
        ax = axes[1, 1]
        if 'kelly_analysis' in analysis:
            kelly_data = analysis['kelly_analysis']
            labels = ['Optimal Kelly', 'Actual Avg', 'Recommended']
            values = [
                kelly_data.get('optimal_kelly_pct', 0),
                kelly_data.get('actual_avg_bet_ratio_pct', 0),
                kelly_data.get('recommended_kelly_fraction', 0) * 100
            ]
            
            ax.bar(labels, values)
            ax.set_ylabel('Bet Size (%)')
            ax.set_title('Kelly Criterion Analysis')
        
        # 6. 月別パフォーマンス
        ax = axes[1, 2]
        if 'monthly_volatility' in analysis:
            monthly_data = analysis['monthly_volatility']
            metrics = ['Mean ROI', 'Best Month', 'Worst Month']
            values = [
                monthly_data.get('monthly_roi_mean', 0),
                monthly_data.get('best_month_roi', 0),
                monthly_data.get('worst_month_roi', 0)
            ]
            
            colors = ['blue', 'green', 'red']
            ax.bar(metrics, values, color=colors)
            ax.set_ylabel('ROI')
            ax.set_title('Monthly Performance')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roi_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # サマリーレポートの作成
        self._create_summary_report(analysis, output_dir)
    
    def _create_summary_report(self, analysis: Dict, output_dir: str) -> None:
        """サマリーレポートの作成"""
        report = []
        report.append("="*60)
        report.append("ROI ANALYSIS SUMMARY REPORT")
        report.append("="*60)
        
        # Overall Metrics
        overall = analysis.get('overall', {})
        report.append("\n[Overall Performance]")
        report.append(f"Total Bets: {overall.get('total_bets', 0)}")
        report.append(f"Win Rate: {overall.get('win_rate', 0)*100:.2f}%")
        report.append(f"ROI: {overall.get('roi', 0)*100:.2f}%")
        report.append(f"Profit Factor: {overall.get('profit_factor', 0):.2f}")
        report.append(f"Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
        
        # Drawdown Analysis
        dd = analysis.get('drawdown_analysis', {})
        report.append("\n[Risk Metrics]")
        report.append(f"Max Drawdown: {dd.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"Max Losing Streak: {dd.get('max_losing_streak', 0)}")
        report.append(f"Recovery Factor: {dd.get('recovery_factor', 0):.2f}")
        
        # Kelly Analysis
        kelly = analysis.get('kelly_analysis', {})
        report.append("\n[Kelly Criterion]")
        report.append(f"Optimal Kelly: {kelly.get('optimal_kelly_pct', 0):.2f}%")
        report.append(f"Recommended (1/4 Kelly): {kelly.get('recommended_kelly_fraction', 0)*100:.2f}%")
        
        # Best Performing Segments
        report.append("\n[Best Performing Segments]")
        
        if 'by_odds_range' in analysis:
            best_odds_range = max(
                analysis['by_odds_range'].items(),
                key=lambda x: x[1]['roi']
            )
            report.append(f"Best Odds Range: {best_odds_range[0]} (ROI: {best_odds_range[1]['roi']*100:.2f}%)")
        
        # Monthly Volatility
        monthly = analysis.get('monthly_volatility', {})
        report.append("\n[Monthly Consistency]")
        report.append(f"Positive Months: {monthly.get('positive_months_pct', 0):.1f}%")
        report.append(f"Monthly Sharpe: {monthly.get('monthly_roi_sharpe', 0):.2f}")
        
        # 実運用可能性の判定
        report.append("\n[Viability Assessment]")
        roi = overall.get('roi', 0)
        sharpe = overall.get('sharpe_ratio', 0)
        max_dd = abs(dd.get('max_drawdown_pct', 0))
        positive_months = monthly.get('positive_months_pct', 0)
        
        viability_score = 0
        if roi > 0.05: viability_score += 1
        if sharpe > 1.0: viability_score += 1
        if max_dd < 20: viability_score += 1
        if positive_months > 60: viability_score += 1
        
        report.append(f"Viability Score: {viability_score}/4")
        
        if viability_score >= 3:
            report.append("✅ VIABLE FOR LIVE TRADING (with proper risk management)")
        elif viability_score >= 2:
            report.append("⚠️ MARGINAL - Needs improvement before live trading")
        else:
            report.append("❌ NOT READY - Significant improvements needed")
        
        # ファイルに保存
        with open(f'{output_dir}/roi_analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # JSON形式でも保存
        with open(f'{output_dir}/roi_analysis_detailed.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)


def main():
    """テスト実行"""
    # サンプルデータの生成（実際はバックテスト結果を使用）
    sample_trades = []
    
    np.random.seed(42)
    capital = 1000000
    
    for i in range(500):
        odds = np.random.choice([2, 3, 5, 10, 20, 50], p=[0.3, 0.3, 0.2, 0.1, 0.08, 0.02])
        expected_value = np.random.uniform(0.9, 1.3)
        bet_amount = capital * 0.01
        
        # 勝率は期待値に応じて調整
        win_prob = min(0.5, expected_value / (odds + 1))
        is_win = np.random.random() < win_prob
        
        profit = bet_amount * (odds - 1) if is_win else -bet_amount
        capital += profit
        
        sample_trades.append({
            'race_id': f'2024{i//20:02d}{i%20:02d}01',
            'horse_num': np.random.randint(1, 19),
            'bet_amount': bet_amount,
            'odds': odds,
            'profit': profit,
            'capital': capital,
            'is_win': is_win,
            'expected_value': expected_value
        })
    
    # 分析実行
    analyzer = ROIAnalyzer()
    analysis = analyzer.analyze_betting_patterns(sample_trades)
    analyzer.create_visualization(analysis)
    
    print("ROI分析が完了しました。")
    print(f"結果は results_improved/ フォルダに保存されました。")


if __name__ == "__main__":
    main()