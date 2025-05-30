#!/usr/bin/env python3
"""
データ型を確認
"""

import pandas as pd

# データを読み込んで型を確認
df = pd.read_excel('data/2023.xlsx', nrows=10)

print("=== データ型の確認 ===")
print("\n着順カラムの情報:")
print(f"データ型: {df['着順'].dtype}")
print(f"サンプル値: {df['着順'].tolist()}")
print(f"ユニーク値: {df['着順'].unique()}")

# 数値に変換できるか確認
print("\n数値変換テスト:")
try:
    df['着順_numeric'] = pd.to_numeric(df['着順'], errors='coerce')
    print("✓ 数値変換成功")
    print(f"変換後: {df['着順_numeric'].tolist()}")
    print(f"NaN数: {df['着順_numeric'].isna().sum()}")
except Exception as e:
    print(f"✗ エラー: {e}")