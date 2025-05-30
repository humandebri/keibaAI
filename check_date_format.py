#!/usr/bin/env python3
"""
データの日付形式を確認するスクリプト
"""

import pandas as pd
import os

# pipenvの仮想環境で実行していることを確認
print("Checking date formats in Excel files...")

# 各年のデータを確認
for year in range(2014, 2025):
    file_path = f'data/{year}.xlsx'
    if os.path.exists(file_path):
        try:
            # 最初の5行だけ読み込んで確認
            df = pd.read_excel(file_path, nrows=5)
            print(f"\n{year}.xlsx:")
            print(f"Date column sample: {df['日付'].tolist()}")
            print(f"Date column type: {df['日付'].dtype}")
        except Exception as e:
            print(f"Error reading {year}.xlsx: {e}")