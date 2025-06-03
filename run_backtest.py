#!/usr/bin/env python3
"""
バックテスト実行スクリプト
使い方: python run_backtest.py [オプション]
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
from datetime import datetime

from src.strategies.optimized_kelly_strategy import OptimizedKellyStrategy

def main():
    parser = argparse.ArgumentParser(description='競馬AI バックテスト実行')
    
    # データ関連
    parser.add_argument('--data_file', type=str, 
                       default='encoded/2020_2025encoded_data_v2.csv',
                       help='エンコード済みデータファイル')
    parser.add_argument('--data_limit', type=int, default=None,
                       help='使用データ行数制限（テスト用）')
    
    # 期間設定
    parser.add_argument('--train_years', nargs='+', type=int,
                       default=[2020, 2021, 2022],
                       help='訓練期間の年（例: 2020 2021 2022）')
    parser.add_argument('--test_years', nargs='+', type=int,
                       default=[2024],
                       help='テスト期間の年（例: 2024）')
    
    # 戦略パラメータ
    parser.add_argument('--strategy', type=str, 
                       choices=['conservative', 'standard', 'aggressive'],
                       default='standard',
                       help='戦略タイプ')
    parser.add_argument('--min_ev', type=float, default=None,
                       help='最低期待値（カスタム設定時）')
    parser.add_argument('--kelly_fraction', type=float, default=None,
                       help='Kelly基準最大比率（カスタム設定時）')
    parser.add_argument('--risk_adjustment', type=float, default=None,
                       help='リスク調整係数（カスタム設定時）')
    
    # 資金設定
    parser.add_argument('--initial_capital', type=int, default=1_000_000,
                       help='初期資金（円）')
    
    # 出力設定
    parser.add_argument('--output_dir', type=str, default='backtest_results',
                       help='結果出力ディレクトリ')
    parser.add_argument('--save_details', action='store_true',
                       help='詳細な取引履歴も保存')
    
    args = parser.parse_args()
    
    print("🔬 競馬AI バックテスト実行")
    print("=" * 50)
    
    # データ読み込み
    print(f"📊 データ読み込み: {args.data_file}")
    if not os.path.exists(args.data_file):
        print(f"❌ データファイルが見つかりません: {args.data_file}")
        return
    
    data = pd.read_csv(args.data_file)
    if args.data_limit:
        data = data.head(args.data_limit)
        print(f"   データ制限: {args.data_limit}行")
    
    print(f"   データサイズ: {len(data)}行, {len(data.columns)}列")
    
    # 年カラムが無い場合は追加
    if 'year' not in data.columns:
        print("   年カラム追加中...")
        if '日付' in data.columns:
            try:
                # 数値データを年に変換
                base_year = 2020
                data['year'] = base_year + (data['日付'] // 365)
            except:
                data['year'] = 2024
        else:
            data['year'] = 2024
    
    # 戦略設定
    strategy_configs = {
        'conservative': {
            'min_expected_value': 1.15,
            'max_kelly_fraction': 0.08,
            'risk_adjustment': 0.5,
            'diversification_limit': 3
        },
        'standard': {
            'min_expected_value': 1.05,
            'max_kelly_fraction': 0.15,
            'risk_adjustment': 0.7,
            'diversification_limit': 8
        },
        'aggressive': {
            'min_expected_value': 1.02,
            'max_kelly_fraction': 0.20,
            'risk_adjustment': 0.8,
            'diversification_limit': 12
        }
    }
    
    config = strategy_configs[args.strategy].copy()
    
    # カスタムパラメータで上書き
    if args.min_ev:
        config['min_expected_value'] = args.min_ev
    if args.kelly_fraction:
        config['max_kelly_fraction'] = args.kelly_fraction
    if args.risk_adjustment:
        config['risk_adjustment'] = args.risk_adjustment
    
    print(f"⚙️ 戦略設定: {args.strategy}")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 戦略初期化
    strategy = OptimizedKellyStrategy(**config)
    
    print(f"📅 訓練期間: {args.train_years}")
    print(f"📅 テスト期間: {args.test_years}")
    print(f"💰 初期資金: ¥{args.initial_capital:,}")
    
    # バックテスト実行
    print("\n🚀 バックテスト実行中...")
    
    try:
        results = strategy.run_backtest(
            data=data,
            train_years=args.train_years,
            test_years=args.test_years,
            feature_cols=[],
            initial_capital=args.initial_capital
        )
        
        # 結果表示
        print("\n📊 バックテスト結果")
        print("=" * 30)
        
        metrics = results.get('metrics', {})
        
        print(f"期待値平均: {metrics.get('avg_expected_value', 0):.3f}")
        print(f"年間リターン: {metrics.get('annual_return', 0)*100:+.1f}%")
        print(f"総リターン: {metrics.get('total_return', 0)*100:+.1f}%")
        print(f"最終資金: ¥{metrics.get('final_capital', 0):,.0f}")
        print(f"勝率: {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"総ベット数: {metrics.get('total_bets', 0)}")
        print(f"シャープ比: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"最大ドローダウン: {metrics.get('max_drawdown', 0)*100:.1f}%")
        
        if 'calmar_ratio' in metrics:
            print(f"Calmar比: {metrics['calmar_ratio']:.2f}")
        
        # ベットタイプ別結果
        if 'by_type' in metrics:
            print("\nベットタイプ別結果:")
            for bet_type, stats in metrics['by_type'].items():
                print(f"  {bet_type}:")
                print(f"    回数: {stats['count']}, 勝率: {stats['win_rate']*100:.1f}%")
                print(f"    利益: ¥{stats['profit']:,.0f}, ROI: {stats['roi']*100:+.1f}%")
        
        # 結果保存
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # メトリクス保存
        metrics_file = os.path.join(args.output_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n💾 メトリクス保存: {metrics_file}")
        
        # 詳細履歴保存（オプション）
        if args.save_details:
            trades_file = os.path.join(args.output_dir, f'trades_{timestamp}.json')
            with open(trades_file, 'w', encoding='utf-8') as f:
                json.dump(results['trades'], f, ensure_ascii=False, indent=2)
            print(f"💾 取引履歴保存: {trades_file}")
        
        # 成功評価
        if metrics.get('avg_expected_value', 0) >= 1.0:
            print("\n✅ 期待値1.0以上達成！")
        if metrics.get('annual_return', 0) >= 0.15:
            print("✅ 年間リターン15%以上達成！")
        if metrics.get('sharpe_ratio', 0) >= 1.5:
            print("✅ シャープ比1.5以上達成！")
        
    except Exception as e:
        print(f"❌ バックテストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()