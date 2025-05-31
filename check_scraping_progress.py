#!/usr/bin/env python3
"""
スクレイピングの進捗確認
"""

import pandas as pd
import os

def check_progress():
    """スクレイピング結果を確認"""
    
    # 出力ディレクトリを確認
    data_dir = "data_with_payout"
    
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
        print(f"=== {data_dir} の内容 ===")
        print(f"ファイル数: {len(files)}")
        
        for file in sorted(files):
            filepath = os.path.join(data_dir, file)
            try:
                df = pd.read_excel(filepath)
                unique_races = df['race_id'].nunique() if 'race_id' in df.columns else 0
                print(f"\n{file}:")
                print(f"  総行数: {len(df):,}")
                print(f"  レース数: {unique_races:,}")
                print(f"  平均頭数: {len(df)/unique_races:.1f}頭/レース" if unique_races > 0 else "")
                
                # 月別の分布
                if '日付' in df.columns and len(df) > 0:
                    # 日付から月を抽出
                    months = df['日付'].str.extract(r'(\d+)月')[0].value_counts().sort_index()
                    print(f"  月別レース分布: {dict(months)}")
                    
            except Exception as e:
                print(f"  読み込みエラー: {e}")
    else:
        print(f"{data_dir} が見つかりません")
    
    # 通常のdataディレクトリも確認
    if os.path.exists("data"):
        xlsx_files = [f for f in os.listdir("data") if f.endswith('.xlsx')]
        if xlsx_files:
            print(f"\n=== data/ にある既存ファイル ===")
            print(f"ファイル数: {len(xlsx_files)}")
            print(f"ファイル: {sorted(xlsx_files)[:5]}...")

if __name__ == "__main__":
    check_progress()