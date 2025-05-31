#!/usr/bin/env python3
"""
Keiba AI Quick Start Script
競馬AI予測システムの簡単実行スクリプト
"""

import sys
import os
import argparse
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Keiba AI Quick Start')
    parser.add_argument('--mode', choices=['encode', 'train', 'backtest'], 
                       default='backtest',
                       help='実行モード: encode (データ前処理), train (モデル訓練), backtest (バックテスト)')
    parser.add_argument('--year', type=int, help='処理する年（encodeモードで使用）')
    parser.add_argument('--start', type=int, help='開始年（encodeモードで使用）')
    parser.add_argument('--end', type=int, help='終了年（encodeモードで使用）')
    
    args = parser.parse_args()
    
    print(f"=== Keiba AI Quick Start ===")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'encode':
        # データエンコーディングのみ実行（スクレイピングは含まない）
        from src.data_processing.data_encoding import main as encode_main
        print("\n既存のデータをエンコードします...")
        print("注意: スクレイピングは実行しません。データファイルが必要です。")
        
        if args.year:
            encode_main(args.year, args.year)
        elif args.start and args.end:
            encode_main(args.start, args.end)
        else:
            print("エンコードする年を指定してください: --year 2024 または --start 2022 --end 2023")
            
    elif args.mode == 'train':
        # モデル訓練
        from src.modeling.model_training import main as train_main
        print("\nモデルを訓練します...")
        train_main()
        
    elif args.mode == 'backtest':
        # バックテスト実行
        from src.backtesting.backtest import main as backtest_main
        print("\nバックテストを実行します...")
        backtest_main()
    
    print("\n完了しました！")

if __name__ == "__main__":
    main()