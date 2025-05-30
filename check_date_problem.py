#!/usr/bin/env python3
"""
日付パースエラーの原因を特定
"""

import pandas as pd
import numpy as np

# 各年のデータを確認
print("Checking for problematic date values...")

all_dates = []
for year in range(2014, 2025):
    file_path = f'data/{year}.xlsx'
    try:
        df = pd.read_excel(file_path)
        print(f"\n{year}.xlsx: {len(df)} rows")
        
        # 日付列の値を確認
        date_col = df['日付']
        
        # 問題のある値を探す
        for idx, val in enumerate(date_col):
            try:
                # 文字列に変換して確認
                str_val = str(val)
                if '??' in str_val or '?' in str_val:
                    print(f"  Row {idx}: Problematic date: {repr(val)}")
                    all_dates.append((year, idx, val))
            except:
                pass
                
        # 日付がNaTかnullの行も確認
        null_dates = date_col.isna().sum()
        if null_dates > 0:
            print(f"  Null dates: {null_dates} rows")
            
    except Exception as e:
        print(f"Error reading {year}.xlsx: {e}")

# 問題のある日付の詳細を表示
if all_dates:
    print("\n=== Summary of problematic dates ===")
    for year, idx, val in all_dates[:10]:  # 最初の10個を表示
        print(f"{year}.xlsx row {idx}: {repr(val)}")
else:
    print("\nNo problematic dates found with ?? pattern")
    
# concat時の問題を確認
print("\n=== Testing concatenation ===")
dfs = []
for year in [2019, 2020, 2021]:  # 問題が起きそうな年付近を確認
    try:
        df = pd.read_excel(f'data/{year}.xlsx')
        dfs.append(df)
        print(f"Loaded {year}: {len(df)} rows, date type: {df['日付'].dtype}")
    except Exception as e:
        print(f"Error: {e}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")
    
    # 日付変換を試す
    try:
        combined['日付'] = pd.to_datetime(combined['日付'])
        print("Date conversion successful!")
    except Exception as e:
        print(f"Date conversion error: {e}")
        
        # エラーの詳細を確認
        print("\nChecking individual values...")
        for i, val in enumerate(combined['日付']):
            try:
                pd.to_datetime(val)
            except Exception as e2:
                print(f"Row {i}: {repr(val)} -> {e2}")
                if i > 10:  # 最初の10個のエラーのみ表示
                    break