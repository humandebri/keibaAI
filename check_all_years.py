#!/usr/bin/env python3
"""
全年度のデータ型を確認
"""

import pandas as pd

print("=== 全年度の着順データ型確認 ===")

for year in range(2014, 2024):
    try:
        df = pd.read_excel(f'data/{year}.xlsx', nrows=100)
        print(f"\n{year}年:")
        print(f"  着順の型: {df['着順'].dtype}")
        
        # 文字列が含まれているかチェック
        non_numeric = df['着順'].apply(lambda x: not str(x).isdigit() if pd.notna(x) else False)
        if non_numeric.any():
            print(f"  文字列値あり: {df[non_numeric]['着順'].unique()}")
        
        # サンプル表示
        unique_values = df['着順'].unique()[:10]
        print(f"  サンプル: {unique_values}")
        
    except Exception as e:
        print(f"\n{year}年: エラー - {e}")