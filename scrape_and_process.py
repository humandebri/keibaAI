#!/usr/bin/env python3
"""
データスクレイピング＆処理統合スクリプト
使い方: python scrape_and_process.py [年] [オプション]
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """コマンドを実行"""
    print(f"🚀 {description}")
    print(f"   実行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} 完了")
        if result.stdout:
            print(f"   出力: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗")
        print(f"   エラー: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='競馬データスクレイピング＆処理')
    
    # 基本設定
    parser.add_argument('year', type=int, nargs='?', 
                       default=datetime.now().year,
                       help='スクレイピング対象年（デフォルト: 今年）')
    parser.add_argument('--end_year', type=int,
                       help='終了年（範囲指定時）')
    
    # スクレイピング設定
    parser.add_argument('--scraping_method', type=str,
                       choices=['basic', 'enhanced', 'checkpoint'],
                       default='enhanced',
                       help='スクレイピング方法')
    parser.add_argument('--workers', type=int, default=4,
                       help='並列処理数')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='リクエスト間隔（秒）')
    
    # 処理設定
    parser.add_argument('--skip_scraping', action='store_true',
                       help='スクレイピングをスキップ')
    parser.add_argument('--skip_encoding', action='store_true',
                       help='エンコーディングをスキップ')
    parser.add_argument('--skip_training', action='store_true',
                       help='モデル訓練をスキップ')
    
    # 出力設定
    parser.add_argument('--data_dir', type=str, default='data_with_payout',
                       help='生データ保存ディレクトリ')
    parser.add_argument('--encoded_dir', type=str, default='encoded',
                       help='エンコード済みデータ保存ディレクトリ')
    
    args = parser.parse_args()
    
    start_year = args.year
    end_year = args.end_year or args.year
    
    print("🕷️ 競馬データスクレイピング＆処理パイプライン")
    print("=" * 50)
    print(f"対象年: {start_year} - {end_year}")
    print(f"スクレイピング方法: {args.scraping_method}")
    print(f"並列処理数: {args.workers}")
    
    # ディレクトリ作成
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.encoded_dir, exist_ok=True)
    
    # Step 1: データスクレイピング
    if not args.skip_scraping:
        print(f"\n📥 Step 1: データスクレイピング ({start_year}-{end_year})")
        
        if args.scraping_method == 'enhanced':
            for year in range(start_year, end_year + 1):
                command = (f"python src/data_processing/enhanced_scraping.py "
                          f"--year {year} --workers {args.workers} "
                          f"--delay {args.delay} --output_dir {args.data_dir}")
                
                if not run_command(command, f"{year}年データ取得"):
                    print(f"⚠️ {year}年のスクレイピング失敗、継続します")
        
        elif args.scraping_method == 'checkpoint':
            command = (f"python src/data_processing/data_scraping_with_checkpoint.py "
                      f"--start {start_year} --end {end_year} "
                      f"--workers {args.workers} --output_dir {args.data_dir}")
            
            run_command(command, f"チェックポイント方式スクレイピング")
        
        elif args.scraping_method == 'basic':
            # 基本的なスクレイピング（notebook実行）
            print("   基本スクレイピングはJupyter Notebookで実行してください:")
            print("   notebooks/00.data_scraping.ipynb")
    
    else:
        print("\n⏭️ Step 1: データスクレイピング スキップ")
    
    # Step 2: データエンコーディング
    if not args.skip_encoding:
        print(f"\n🔄 Step 2: データエンコーディング")
        
        command = (f"python src/data_processing/data_encoding_v2.py "
                  f"--start {start_year} --end {end_year} "
                  f"--data_dir {args.data_dir} "
                  f"--encoded_dir {args.encoded_dir}")
        
        if not run_command(command, "データエンコーディング"):
            print("❌ エンコーディング失敗、処理を中断します")
            return
    
    else:
        print("\n⏭️ Step 2: データエンコーディング スキップ")
    
    # Step 3: モデル訓練（オプション）
    if not args.skip_training:
        print(f"\n🤖 Step 3: モデル訓練")
        
        # 利用可能なエンコード済みファイルを確認
        encoded_file = f"{args.encoded_dir}/{start_year}_{end_year}encoded_data_v2.csv"
        
        if os.path.exists(encoded_file):
            command = f"python train_model_2020_2025.py --data_file {encoded_file}"
            run_command(command, "モデル訓練")
        else:
            print(f"⚠️ エンコード済みファイルが見つかりません: {encoded_file}")
    
    else:
        print("\n⏭️ Step 3: モデル訓練 スキップ")
    
    # 完了メッセージ
    print(f"\n🎉 データ処理パイプライン完了！")
    print(f"📁 生データ: {args.data_dir}")
    print(f"📁 エンコード済み: {args.encoded_dir}")
    
    # 次のステップの案内
    print(f"\n🔜 次のステップ:")
    print(f"1. バックテスト実行:")
    print(f"   python run_backtest.py --data_file {args.encoded_dir}/{start_year}_{end_year}encoded_data_v2.csv")
    print(f"2. レース予測:")
    print(f"   python predict_races.py your_race_data.csv")

def check_dependencies():
    """依存関係チェック"""
    required_files = [
        'src/data_processing/enhanced_scraping.py',
        'src/data_processing/data_encoding_v2.py',
        'src/data_processing/data_scraping_with_checkpoint.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 必要なファイルが見つかりません:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n以下のファイルを確認してください:")
        print("- src/data_processing/ ディレクトリ内のスクリプト")
        return False
    
    return True

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)
    
    main()