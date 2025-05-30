#!/usr/bin/env python3
"""
データのカラム名を確認
"""

import pandas as pd

# 2023年のデータでカラム名を確認
df = pd.read_excel('data/2023.xlsx', nrows=5)
print("Available columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())