#!/usr/bin/env python3
"""
Keiba AI Simple Quick Start
シンプルなバックテスト実行
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.backtesting.backtest import ImprovedBacktest

def main():
    print("=== Keiba AI バックテスト（シンプル版） ===")
    print("複勝ベッティング戦略で実行します\n")
    
    # バックテストシステムの初期化
    backtest = ImprovedBacktest()
    
    # データの読み込みと準備
    backtest.load_and_prepare_data()
    
    print(f"\nLoaded {len(backtest.data)} race entries")
    print(f"Date range: {backtest.data['日付'].min()} to {backtest.data['日付'].max()}")
    
    # パラメータ表示
    print(f"\nParameters:")
    print(f"- Betting fraction: {backtest.betting_fraction:.1%}")
    print(f"- EV threshold: {backtest.ev_threshold}")
    print(f"- Monthly stop loss: {backtest.monthly_stop_loss:.1%}")
    
    # バックテスト実行（パラメータ最適化なし）
    results = backtest.run_backtest()
    
    # 結果の表示
    if results:
        final_capital = results[-1]['end_capital']
        total_return = (final_capital - backtest.initial_capital) / backtest.initial_capital
        total_bets = sum(r['num_bets'] for r in results)
        total_wins = sum(r['num_wins'] for r in results)
        overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
        
        print(f"\n=== 最終結果 ===")
        print(f"初期資産: ¥{backtest.initial_capital:,}")
        print(f"最終資産: ¥{final_capital:,.0f}")
        print(f"総リターン: {total_return:.2%}")
        print(f"年率リターン: {(1 + total_return) ** (1/10) - 1:.2%}")
        print(f"勝率: {overall_win_rate:.1%}")
        print(f"総ベット数: {total_bets}")
        
        # 改善効果の表示
        print("\n=== 改善効果 ===")
        print("改善前（単勝）: -100%の損失")
        print(f"改善後（複勝）: {total_return:+.1%}のリターン")
        
        if total_return > 0:
            print("\n✓ プラスのリターンを達成しました！")

if __name__ == "__main__":
    main()