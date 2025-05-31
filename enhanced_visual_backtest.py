#!/usr/bin/env python3
"""
視覚的なバックテスト結果を生成するスクリプト
実際のオッズデータを使用し、複数のグラフを出力
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
from datetime import datetime
import argparse

from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.core.utils import setup_logger


class VisualBacktest:
    """視覚的なバックテストクラス"""
    
    def __init__(self, output_dir: str = "backtest_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = setup_logger("VisualBacktest")
        
        # プロット設定
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
        
    def run_backtest_with_visualization(self, 
                                      start_year: int = 2022,
                                      end_year: int = 2025,
                                      train_years: list = None,
                                      test_years: list = None,
                                      min_ev: float = 1.1,
                                      initial_capital: float = 1_000_000,
                                      bet_fraction: float = 0.01):
        """バックテストを実行し、結果を視覚化"""
        
        # デフォルトの訓練・テスト年設定
        if train_years is None:
            train_years = [2022]
        if test_years is None:
            test_years = [2024, 2025]
            
        self.logger.info(f"バックテスト期間: {start_year}-{end_year}")
        self.logger.info(f"訓練年: {train_years}")
        self.logger.info(f"テスト年: {test_years}")
        
        # 戦略の初期化
        strategy = AdvancedBettingStrategy(
            min_expected_value=min_ev,
            enable_trifecta=True,
            enable_quinella=True,
            enable_wide=True,
            use_actual_odds=True
        )
        
        # データ読み込み（data_with_payoutディレクトリから）
        try:
            strategy.load_data(start_year=start_year, end_year=end_year, use_payout_data=True)
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            return None
            
        # データ分割
        strategy.split_data(train_years=train_years, test_years=test_years)
        
        # バックテスト実行
        results = strategy.run_backtest(initial_capital=initial_capital)
        
        # 結果の視覚化
        self._create_comprehensive_plots(results, initial_capital)
        
        # 結果の保存
        self._save_results(results, {
            'start_year': start_year,
            'end_year': end_year,
            'train_years': train_years,
            'test_years': test_years,
            'min_ev': min_ev,
            'initial_capital': initial_capital,
            'bet_fraction': bet_fraction
        })
        
        return results
        
    def _create_comprehensive_plots(self, results: dict, initial_capital: float):
        """包括的なプロットを作成"""
        if not results or 'trades' not in results:
            self.logger.warning("取引データがありません")
            return
            
        trades = results['trades']
        if not trades:
            self.logger.warning("取引が0件です")
            return
            
        # データフレーム作成
        df = pd.DataFrame(trades)
        
        # 日付情報の追加（race_idから推定）
        df['date'] = pd.to_datetime(df['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        
        # メイン図の作成
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. 資金推移
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_capital_evolution(ax1, df, initial_capital)
        
        # 2. 月別パフォーマンス
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_monthly_performance(ax2, df)
        
        # 3. 期待値分布
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_ev_distribution(ax3, df)
        
        # 4. 馬券種別勝率
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_win_rate_by_type(ax4, df)
        
        # 5. 利益分布
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_profit_distribution(ax5, df)
        
        # 6. ローリング勝率
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_rolling_metrics(ax6, df)
        
        # 7. ドローダウン
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_drawdown(ax7, df, initial_capital)
        
        # 8. ベット額の推移
        ax8 = fig.add_subplot(gs[4, 0])
        self._plot_bet_amounts(ax8, df)
        
        # 9. サマリー統計
        ax9 = fig.add_subplot(gs[4, 1])
        self._plot_summary_stats(ax9, results)
        
        plt.suptitle('競馬AIバックテスト結果 - 総合分析', fontsize=16, fontweight='bold')
        
        # 保存
        output_path = self.output_dir / 'backtest_comprehensive_analysis.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        self.logger.info(f"総合分析グラフを保存しました: {output_path}")
        
        # 追加の詳細分析
        self._create_detailed_analysis(df, results)
        
    def _plot_capital_evolution(self, ax, df, initial_capital):
        """資金推移のプロット"""
        ax.plot(df['date'], df['capital'], linewidth=2, color='darkblue', label='資金残高')
        ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='初期資金')
        
        # 移動平均線
        if len(df) > 20:
            ma20 = df['capital'].rolling(20).mean()
            ax.plot(df['date'], ma20, color='orange', alpha=0.7, label='20取引移動平均')
        
        ax.set_title('資金推移', fontsize=14, fontweight='bold')
        ax.set_xlabel('日付')
        ax.set_ylabel('資金 (円)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Y軸のフォーマット
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
    def _plot_monthly_performance(self, ax, df):
        """月別パフォーマンス"""
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month')['profit'].sum()
        
        colors = ['green' if x > 0 else 'red' for x in monthly.values]
        bars = ax.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        
        ax.set_title('月別損益', fontsize=12, fontweight='bold')
        ax.set_xlabel('月')
        ax.set_ylabel('損益 (円)')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(m) for m in monthly.index], rotation=45)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
    def _plot_ev_distribution(self, ax, df):
        """期待値分布"""
        ev_data = [b['expected_value'] for b in df['bet']]
        
        ax.hist(ev_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', label='EV=1.0')
        ax.axvline(x=np.mean(ev_data), color='green', linestyle='--', label=f'平均EV={np.mean(ev_data):.2f}')
        
        ax.set_title('期待値分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('期待値')
        ax.set_ylabel('頻度')
        ax.legend()
        
    def _plot_win_rate_by_type(self, ax, df):
        """馬券種別勝率"""
        bet_types = [b['type'] for b in df['bet']]
        df['bet_type'] = bet_types
        
        win_rates = df.groupby('bet_type').agg({
            'is_win': ['sum', 'count', 'mean']
        })
        
        types = win_rates.index
        rates = win_rates[('is_win', 'mean')] * 100
        counts = win_rates[('is_win', 'count')]
        
        bars = ax.bar(types, rates, alpha=0.7, color=['blue', 'green', 'orange'])
        
        # データラベル
        for bar, count, rate in zip(bars, counts, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%\n(n={count})', ha='center', va='bottom')
        
        ax.set_title('馬券種別勝率', fontsize=12, fontweight='bold')
        ax.set_ylabel('勝率 (%)')
        ax.set_ylim(0, max(rates) * 1.2)
        
    def _plot_profit_distribution(self, ax, df):
        """利益分布"""
        profits = df['profit'].values
        
        # ヒストグラム
        n, bins, patches = ax.hist(profits, bins=50, alpha=0.7, edgecolor='black')
        
        # 色分け
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('red')
            else:
                patch.set_facecolor('green')
                
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=np.mean(profits), color='blue', linestyle='--', 
                  label=f'平均: {np.mean(profits):,.0f}円')
        
        ax.set_title('利益分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('利益 (円)')
        ax.set_ylabel('頻度')
        ax.legend()
        
    def _plot_rolling_metrics(self, ax, df):
        """ローリング勝率"""
        window = min(100, len(df) // 10)
        if window < 10:
            window = 10
            
        rolling_wr = df['is_win'].rolling(window).mean() * 100
        
        ax.plot(df.index, rolling_wr, linewidth=2, color='purple')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{window}取引ローリング勝率', fontsize=12, fontweight='bold')
        ax.set_xlabel('取引番号')
        ax.set_ylabel('勝率 (%)')
        ax.set_ylim(0, 100)
        
    def _plot_drawdown(self, ax, df, initial_capital):
        """ドローダウン"""
        # 累積最大値
        cummax = df['capital'].cummax()
        drawdown = (df['capital'] - cummax) / cummax * 100
        
        ax.fill_between(df.index, 0, drawdown, color='red', alpha=0.3)
        ax.plot(df.index, drawdown, color='darkred', linewidth=1)
        
        # 最大ドローダウン
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        ax.plot(max_dd_idx, max_dd, 'ro', markersize=8)
        ax.text(max_dd_idx, max_dd - 1, f'最大DD: {max_dd:.1f}%', 
               ha='center', va='top')
        
        ax.set_title('ドローダウン', fontsize=12, fontweight='bold')
        ax.set_xlabel('取引番号')
        ax.set_ylabel('ドローダウン (%)')
        ax.set_ylim(min(drawdown.min() * 1.1, -5), 1)
        
    def _plot_bet_amounts(self, ax, df):
        """ベット額の推移"""
        ax.scatter(df.index, df['amount'], alpha=0.5, s=20, c=df['is_win'], 
                  cmap='RdYlGn', label='ベット額')
        
        # 移動平均
        if len(df) > 20:
            ma = df['amount'].rolling(20).mean()
            ax.plot(df.index, ma, color='blue', linewidth=2, label='20取引MA')
        
        ax.set_title('ベット額推移', fontsize=12, fontweight='bold')
        ax.set_xlabel('取引番号')
        ax.set_ylabel('ベット額 (円)')
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
    def _plot_summary_stats(self, ax, results):
        """サマリー統計"""
        ax.axis('off')
        
        metrics = results.get('metrics', {})
        
        # 基本統計
        stats_text = f"""
【基本統計】
初期資金: ¥{metrics.get('initial_capital', 0):,.0f}
最終資金: ¥{metrics.get('final_capital', 0):,.0f}
総リターン: {metrics.get('total_return', 0)*100:.1f}%
総ベット数: {metrics.get('total_bets', 0):,}
勝利数: {metrics.get('total_wins', 0):,}
勝率: {metrics.get('win_rate', 0)*100:.1f}%
平均期待値: {metrics.get('avg_expected_value', 0):.2f}

【馬券種別統計】
"""
        
        # 馬券種別統計
        for bet_type, stats in metrics.get('by_type', {}).items():
            stats_text += f"\n{bet_type}:"
            stats_text += f"\n  勝率: {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['count']})"
            stats_text += f"\n  損益: ¥{stats['profit']:,.0f}"
            stats_text += f"\n  ROI: {stats['roi']*100:.1f}%"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def _create_detailed_analysis(self, df, results):
        """詳細分析グラフの作成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('競馬AIバックテスト結果 - 詳細分析', fontsize=16, fontweight='bold')
        
        # 1. 期待値 vs 実際のリターン
        ax = axes[0, 0]
        self._plot_ev_vs_returns(ax, df)
        
        # 2. 時間帯別パフォーマンス
        ax = axes[0, 1]
        self._plot_performance_by_time(ax, df)
        
        # 3. オッズ帯別勝率
        ax = axes[1, 0]
        self._plot_win_rate_by_odds_range(ax, df)
        
        # 4. 累積ROI
        ax = axes[1, 1]
        self._plot_cumulative_roi(ax, df)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'backtest_detailed_analysis.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        self.logger.info(f"詳細分析グラフを保存しました: {output_path}")
        
    def _plot_ev_vs_returns(self, ax, df):
        """期待値vs実際のリターン"""
        ev_bins = np.arange(1.0, 2.5, 0.1)
        ev_groups = pd.cut([b['expected_value'] for b in df['bet']], bins=ev_bins)
        
        grouped = df.groupby(ev_groups).agg({
            'profit': 'mean',
            'amount': 'mean',
            'is_win': ['mean', 'count']
        })
        
        grouped['roi'] = grouped['profit'] / grouped['amount']
        
        # ROIプロット
        valid_idx = grouped[('is_win', 'count')] >= 10
        x = [interval.mid for interval in grouped[valid_idx].index]
        y = grouped[valid_idx]['roi'] * 100
        
        ax.scatter(x, y, s=grouped[valid_idx][('is_win', 'count')] * 5, alpha=0.6)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title('期待値別ROI', fontsize=12, fontweight='bold')
        ax.set_xlabel('期待値')
        ax.set_ylabel('ROI (%)')
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_by_time(self, ax, df):
        """時間帯別パフォーマンス"""
        df['hour'] = pd.to_datetime(df['race_id'].astype(str).str[8:12], 
                                   format='%H%M', errors='coerce').dt.hour
        
        hourly = df.groupby('hour').agg({
            'profit': 'sum',
            'is_win': ['mean', 'count']
        })
        
        ax2 = ax.twinx()
        
        # 利益棒グラフ
        bars = ax.bar(hourly.index, hourly['profit'], alpha=0.7, label='利益')
        
        # 勝率線グラフ
        ax2.plot(hourly.index, hourly[('is_win', 'mean')] * 100, 
                'r-o', linewidth=2, label='勝率')
        
        ax.set_title('時間帯別パフォーマンス', fontsize=12, fontweight='bold')
        ax.set_xlabel('時間')
        ax.set_ylabel('利益 (円)')
        ax2.set_ylabel('勝率 (%)')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
    def _plot_win_rate_by_odds_range(self, ax, df):
        """オッズ帯別勝率"""
        odds_data = []
        for i, bet in enumerate(df['bet']):
            odds_data.append({
                'odds': bet.get('estimated_odds', 0),
                'is_win': df.iloc[i]['is_win']
            })
        
        odds_df = pd.DataFrame(odds_data)
        odds_bins = [0, 10, 30, 50, 100, 200, 500, 1000, 10000]
        odds_groups = pd.cut(odds_df['odds'], bins=odds_bins)
        
        grouped = odds_df.groupby(odds_groups)['is_win'].agg(['mean', 'count'])
        
        # 勝率バー
        bars = ax.bar(range(len(grouped)), grouped['mean'] * 100, alpha=0.7)
        
        # データラベル
        for i, (rate, count) in enumerate(zip(grouped['mean'] * 100, grouped['count'])):
            ax.text(i, rate + 1, f'{rate:.1f}%\n(n={count})', 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_title('オッズ帯別勝率', fontsize=12, fontweight='bold')
        ax.set_ylabel('勝率 (%)')
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels([f'{int(iv.left)}-{int(iv.right)}' 
                           for iv in grouped.index], rotation=45)
        
    def _plot_cumulative_roi(self, ax, df):
        """累積ROI"""
        df['cumulative_investment'] = df['amount'].cumsum()
        df['cumulative_profit'] = df['profit'].cumsum()
        df['cumulative_roi'] = (df['cumulative_profit'] / 
                                df['cumulative_investment'] * 100)
        
        ax.plot(df.index, df['cumulative_roi'], linewidth=2, color='darkgreen')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 最終ROI表示
        final_roi = df['cumulative_roi'].iloc[-1]
        ax.text(0.98, 0.02, f'最終ROI: {final_roi:.1f}%', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_title('累積ROI推移', fontsize=12, fontweight='bold')
        ax.set_xlabel('取引番号')
        ax.set_ylabel('累積ROI (%)')
        ax.grid(True, alpha=0.3)
        
    def _save_results(self, results, params):
        """結果をファイルに保存"""
        # JSON形式で保存
        output_data = {
            'parameters': params,
            'metrics': results.get('metrics', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / 'backtest_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        # 取引履歴をCSVで保存
        if results.get('trades'):
            trades_df = pd.DataFrame(results['trades'])
            csv_path = self.output_dir / 'backtest_trades.csv'
            trades_df.to_csv(csv_path, index=False, encoding='utf-8')
            
        self.logger.info(f"結果を保存しました: {self.output_dir}")
        
        # リスク指標の計算と表示
        self._calculate_risk_metrics(results)
        
    def _calculate_risk_metrics(self, results):
        """リスク指標の計算"""
        if not results.get('trades'):
            return
            
        df = pd.DataFrame(results['trades'])
        
        # シャープレシオ（簡易版）
        returns = df['profit'] / df['amount']
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大ドローダウン
        cummax = df['capital'].cummax()
        drawdown = (df['capital'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        risk_text = f"""
【リスク指標】
シャープレシオ: {sharpe:.2f}
最大ドローダウン: {max_drawdown*100:.1f}%
"""
        
        self.logger.info(risk_text)
        
        # リスク警告
        if sharpe < 0.5:
            self.logger.warning("警告: シャープレシオが低いです")
        if max_drawdown < -0.20:
            self.logger.warning("警告: 最大ドローダウンが20%を超えています")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='視覚的バックテスト実行')
    
    parser.add_argument('--start-year', type=int, default=2022,
                       help='データ開始年')
    parser.add_argument('--end-year', type=int, default=2025,
                       help='データ終了年')
    parser.add_argument('--train-years', nargs='+', type=int,
                       help='訓練年（例: --train-years 2022 2023）')
    parser.add_argument('--test-years', nargs='+', type=int,
                       help='テスト年（例: --test-years 2024 2025）')
    parser.add_argument('--min-ev', type=float, default=1.1,
                       help='最低期待値')
    parser.add_argument('--initial-capital', type=float, default=1_000_000,
                       help='初期資金')
    parser.add_argument('--bet-fraction', type=float, default=0.01,
                       help='ベット割合')
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # ビジュアルバックテストの実行
    vb = VisualBacktest(output_dir=args.output_dir)
    
    results = vb.run_backtest_with_visualization(
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        test_years=args.test_years,
        min_ev=args.min_ev,
        initial_capital=args.initial_capital,
        bet_fraction=args.bet_fraction
    )
    
    if results:
        print(f"\nバックテスト完了！")
        print(f"結果は {args.output_dir} ディレクトリに保存されました。")
        print(f"- 総合分析: backtest_comprehensive_analysis.png")
        print(f"- 詳細分析: backtest_detailed_analysis.png")
        print(f"- 取引履歴: backtest_trades.csv")
        print(f"- 結果JSON: backtest_results.json")


if __name__ == "__main__":
    main()