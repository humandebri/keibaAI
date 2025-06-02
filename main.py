#!/usr/bin/env python3
"""
競馬AI統一エントリーポイント

使用例:
    # バックテスト実行
    python main.py backtest --train-years 2020 2021 2022 --test-years 2023 2024
    
    # データ収集
    python main.py collect --start-year 2024 --end-year 2024
    
    # モデル訓練
    python main.py train --years 2020 2021 2022 2023
    
    # リアルタイム実行
    python main.py realtime --mode paper
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.unified_system import UnifiedKeibaAISystem, SystemMode
from src.core.config import Config
from src.strategies.advanced_betting import AdvancedBettingStrategy
from src.data_processing.data_scraping_with_checkpoint import main as scrape_data
from src.data_processing.data_encoding_v2 import main as encode_data


def setup_args() -> argparse.ArgumentParser:
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(
        description="競馬AI統一システム",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # バックテストコマンド
    backtest_parser = subparsers.add_parser('backtest', help='バックテスト実行')
    backtest_parser.add_argument(
        '--train-years', 
        type=int, 
        nargs='+', 
        default=[2020, 2021, 2022],
        help='訓練期間の年（複数指定可能）'
    )
    backtest_parser.add_argument(
        '--test-years', 
        type=int, 
        nargs='+', 
        default=[2023, 2024],
        help='テスト期間の年（複数指定可能）'
    )
    backtest_parser.add_argument(
        '--strategy', 
        choices=['advanced', 'conservative', 'aggressive'],
        default='advanced',
        help='使用する戦略'
    )
    backtest_parser.add_argument(
        '--config', 
        type=str,
        help='設定ファイルパス'
    )
    backtest_parser.add_argument(
        '--output', 
        type=str,
        default='results/backtest_results.json',
        help='結果出力先'
    )
    
    # データ収集コマンド
    collect_parser = subparsers.add_parser('collect', help='データ収集')
    collect_parser.add_argument(
        '--start-year', 
        type=int, 
        required=True,
        help='収集開始年'
    )
    collect_parser.add_argument(
        '--end-year', 
        type=int,
        help='収集終了年（省略時は開始年と同じ）'
    )
    collect_parser.add_argument(
        '--workers', 
        type=int, 
        default=3,
        help='並行処理数'
    )
    
    # データエンコーディングコマンド
    encode_parser = subparsers.add_parser('encode', help='データエンコーディング')
    encode_parser.add_argument(
        '--start-year', 
        type=int, 
        required=True,
        help='エンコーディング開始年'
    )
    encode_parser.add_argument(
        '--end-year', 
        type=int,
        help='エンコーディング終了年'
    )
    
    # モデル訓練コマンド
    train_parser = subparsers.add_parser('train', help='モデル訓練')
    train_parser.add_argument(
        '--years', 
        type=int, 
        nargs='+', 
        default=[2020, 2021, 2022, 2023],
        help='訓練データの年（複数指定可能）'
    )
    train_parser.add_argument(
        '--config', 
        type=str,
        help='設定ファイルパス'
    )
    train_parser.add_argument(
        '--output-dir', 
        type=str,
        default='models',
        help='モデル出力ディレクトリ'
    )
    
    # リアルタイムコマンド
    realtime_parser = subparsers.add_parser('realtime', help='リアルタイム実行')
    realtime_parser.add_argument(
        '--mode', 
        choices=['paper', 'live'],
        default='paper',
        help='実行モード（paper: ペーパートレード, live: 実取引）'
    )
    realtime_parser.add_argument(
        '--model-path', 
        type=str,
        help='使用するモデルのパス'
    )
    realtime_parser.add_argument(
        '--config', 
        type=str,
        help='設定ファイルパス'
    )
    
    # 結果表示コマンド
    show_parser = subparsers.add_parser('show', help='結果表示')
    show_parser.add_argument(
        'result_file', 
        type=str,
        help='結果ファイルパス'
    )
    
    return parser


async def run_backtest(args) -> None:
    """バックテスト実行"""
    print(f"🏇 バックテスト開始: train={args.train_years}, test={args.test_years}")
    
    # 設定の読み込み
    config = Config.load_from_file(args.config) if args.config else Config.load_default()
    
    # システムの初期化
    system = UnifiedKeibaAISystem(config=config, mode=SystemMode.BACKTEST)
    
    # 戦略の設定
    if args.strategy == 'advanced':
        strategy = AdvancedBettingStrategy(config.backtest)
    else:
        # 他の戦略は後で実装
        strategy = AdvancedBettingStrategy(config.backtest)
    
    system.set_strategy(strategy)
    
    # バックテストの実行
    results = system.run_backtest(
        train_years=args.train_years,
        test_years=args.test_years
    )
    
    # 結果の保存
    output_path = Path(args.output)
    system.export_results(output_path)
    
    # 結果の概要表示
    print("\n📊 バックテスト結果:")
    print(f"総収益: {results.get('total_return', 'N/A'):.2%}")
    print(f"年間収益: {results.get('annual_return', 'N/A'):.2%}")
    print(f"勝率: {results.get('win_rate', 'N/A'):.2%}")
    print(f"取引数: {results.get('total_trades', 'N/A')}")
    print(f"結果ファイル: {output_path}")


async def run_collect(args) -> None:
    """データ収集実行"""
    end_year = args.end_year or args.start_year
    print(f"📊 データ収集開始: {args.start_year}-{end_year}")
    
    # データ収集の実行
    await scrape_data(
        start_year=args.start_year,
        end_year=end_year,
        workers=args.workers
    )
    
    print("✅ データ収集完了")


async def run_encode(args) -> None:
    """データエンコーディング実行"""
    end_year = args.end_year or args.start_year
    print(f"🔄 データエンコーディング開始: {args.start_year}-{end_year}")
    
    # エンコーディングの実行
    await encode_data(
        start_year=args.start_year,
        end_year=end_year
    )
    
    print("✅ データエンコーディング完了")


async def run_train(args) -> None:
    """モデル訓練実行"""
    print(f"🤖 モデル訓練開始: years={args.years}")
    
    # 設定の読み込み
    config = Config.load_from_file(args.config) if args.config else Config.load_default()
    
    # システムの初期化
    system = UnifiedKeibaAISystem(config=config, mode=SystemMode.RESEARCH)
    
    # データの読み込みと前処理
    system.load_data(years=args.years)
    system.prepare_features()
    
    # モデルの訓練
    models = system.train_models()
    
    print(f"✅ モデル訓練完了: {len(models)}個のモデルを訓練")


async def run_realtime(args) -> None:
    """リアルタイム実行"""
    mode = SystemMode.PAPER_TRADING if args.mode == 'paper' else SystemMode.LIVE_TRADING
    print(f"⚡ リアルタイム実行開始: mode={mode.value}")
    
    # 設定の読み込み
    config = Config.load_from_file(args.config) if args.config else Config.load_default()
    
    # システムの初期化
    system = UnifiedKeibaAISystem(config=config, mode=mode)
    
    # モデルの読み込み
    if args.model_path:
        system.load_models(args.model_path)
    else:
        print("⚠️ モデルパスが指定されていません。最新のモデルを使用します。")
    
    # 戦略の設定
    strategy = AdvancedBettingStrategy(config.backtest)
    system.set_strategy(strategy)
    
    try:
        # リアルタイム実行
        await system.run_realtime()
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる停止")
        system.stop_realtime()


def show_results(args) -> None:
    """結果表示"""
    result_file = Path(args.result_file)
    
    if not result_file.exists():
        print(f"❌ 結果ファイルが見つかりません: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"📈 結果表示: {result_file}")
    print("=" * 50)
    
    # 主要指標の表示
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if 'return' in key.lower() or 'rate' in key.lower():
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")


async def main():
    """メイン関数"""
    parser = setup_args()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # コマンドの実行
        if args.command == 'backtest':
            await run_backtest(args)
        elif args.command == 'collect':
            await run_collect(args)
        elif args.command == 'encode':
            await run_encode(args)
        elif args.command == 'train':
            await run_train(args)
        elif args.command == 'realtime':
            await run_realtime(args)
        elif args.command == 'show':
            show_results(args)
        else:
            print(f"❌ 不明なコマンド: {args.command}")
            parser.print_help()
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())