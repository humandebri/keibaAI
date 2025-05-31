#!/usr/bin/env python3
"""
Keiba AI メインスクリプト
リファクタリング後の統合エントリーポイント
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import config
from src.core.utils import setup_logger
from src.strategies.advanced_betting import AdvancedBettingStrategy


def run_backtest(strategy_name: str = 'advanced', **kwargs):
    """バックテストの実行"""
    logger = setup_logger('main')
    
    # 戦略の選択
    if strategy_name == 'advanced':
        strategy = AdvancedBettingStrategy(
            min_expected_value=kwargs.get('min_ev', 1.1),
            enable_trifecta=kwargs.get('trifecta', True),
            enable_quinella=kwargs.get('quinella', True),
            enable_wide=kwargs.get('wide', True),
            use_actual_odds=kwargs.get('use_actual_odds', True)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    logger.info(f"Running {strategy_name} strategy")
    
    # データ読み込み
    start_year = kwargs.get('start_year', 2019)
    end_year = kwargs.get('end_year', 2023)
    use_payout_data = kwargs.get('use_payout_data', False)
    
    strategy.load_data(start_year=start_year, end_year=end_year, use_payout_data=use_payout_data)
    
    # データ分割
    train_years = kwargs.get('train_years', [2019, 2020])
    test_years = kwargs.get('test_years', [2021, 2022, 2023])
    
    strategy.split_data(
        train_years=train_years,
        test_years=test_years
    )
    
    # バックテスト実行
    results = strategy.run_backtest(initial_capital=1_000_000)
    
    # 結果表示
    strategy.print_results()
    
    return results


def run_data_collection(start_year: int, end_year: int):
    """データ収集の実行"""
    from src.data_processing.enhanced_scraping import EnhancedRaceScraper
    
    logger = setup_logger('main')
    logger.info(f"Collecting data from {start_year} to {end_year}")
    
    scraper = EnhancedRaceScraper()
    scraper.scrape_years(start_year, end_year)


def run_encoding(start_year: int, end_year: int):
    """データエンコーディングの実行"""
    from src.data_processing.data_encoding import RaceDataEncoder
    
    logger = setup_logger('main')
    logger.info(f"Encoding data from {start_year} to {end_year}")
    
    encoder = RaceDataEncoder()
    output_path = encoder.encode_data(start_year, end_year)
    logger.info(f"Encoded data saved to: {output_path}")


def run_training():
    """モデル訓練の実行"""
    from src.modeling.model_training import train_and_evaluate_model
    
    logger = setup_logger('main')
    logger.info("Training model")
    
    model, results = train_and_evaluate_model()
    
    logger.info(f"Model AUC: {results['auc']:.4f}")
    logger.info(f"Model Accuracy: {results['accuracy']:.4f}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Keiba AI - 競馬予測システム')
    
    parser.add_argument(
        'command',
        choices=['backtest', 'collect', 'encode', 'train'],
        help='実行するコマンド'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['advanced'],
        default='advanced',
        help='バックテスト戦略 (backtest時のみ)'
    )
    
    parser.add_argument(
        '--min-ev',
        type=float,
        default=1.1,
        help='最低期待値'
    )
    
    parser.add_argument(
        '--no-trifecta',
        action='store_true',
        help='三連単を無効化'
    )
    
    parser.add_argument(
        '--no-quinella',
        action='store_true',
        help='馬連を無効化'
    )
    
    parser.add_argument(
        '--no-wide',
        action='store_true',
        help='ワイドを無効化'
    )
    
    parser.add_argument(
        '--no-actual-odds',
        action='store_true',
        help='実際のオッズを使用しない（推定オッズのみ）'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2019,
        help='開始年'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2023,
        help='終了年'
    )
    
    parser.add_argument(
        '--train-years',
        type=int,
        nargs='+',
        help='訓練データの年（スペース区切りで複数指定可）'
    )
    
    parser.add_argument(
        '--test-years',
        type=int,
        nargs='+',
        help='テストデータの年（スペース区切りで複数指定可）'
    )
    
    parser.add_argument(
        '--use-payout-data',
        action='store_true',
        help='data_with_payoutディレクトリのデータを使用'
    )
    
    args = parser.parse_args()
    
    # コマンド実行
    if args.command == 'backtest':
        # デフォルトの訓練・テスト年を設定
        train_years = args.train_years if args.train_years else [args.start_year, args.start_year + 1]
        test_years = args.test_years if args.test_years else list(range(args.start_year + 2, args.end_year + 1))
        
        run_backtest(
            args.strategy,
            min_ev=args.min_ev,
            trifecta=not args.no_trifecta,
            quinella=not args.no_quinella,
            wide=not args.no_wide,
            use_actual_odds=not args.no_actual_odds,
            start_year=args.start_year,
            end_year=args.end_year,
            train_years=train_years,
            test_years=test_years,
            use_payout_data=args.use_payout_data
        )
    elif args.command == 'collect':
        run_data_collection(args.start_year, args.end_year)
    elif args.command == 'encode':
        run_encoding(args.start_year, args.end_year)
    elif args.command == 'train':
        run_training()


if __name__ == '__main__':
    main()