#!/usr/bin/env python3
"""
作成されたExcelファイルから払戻データを確認
"""

import pandas as pd
import json

def verify_payout_data():
    """払戻データの確認"""
    print("=== 払戻データの確認 ===")
    
    # Excelファイルを読み込み
    df = pd.read_excel('test_data_payout/test_with_payout.xlsx')
    
    print(f"データ行数: {len(df)}")
    print(f"カラム数: {len(df.columns)}")
    print(f"\nカラム一覧:")
    for col in df.columns:
        print(f"  - {col}")
    
    # 最初の行の払戻データを確認
    if '払戻データ' in df.columns:
        print("\n=== 1行目の払戻データ ===")
        payout_json = df.iloc[0]['払戻データ']
        print(f"JSONデータ: {payout_json}")
        
        try:
            payout_data = json.loads(payout_json)
            print("\n解析結果:")
            for bet_type, payouts in payout_data.items():
                if payouts:
                    print(f"\n{bet_type}:")
                    for combo, amount in payouts.items():
                        print(f"  {combo}: {amount}円")
        except Exception as e:
            print(f"JSON解析エラー: {e}")
    
    # 基本情報の確認
    print("\n=== 基本情報の確認 ===")
    first_row = df.iloc[0]
    print(f"馬名: {first_row['馬']}")
    print(f"着順: {first_row['着順']}")
    print(f"馬番: {first_row['馬番']}")
    print(f"枠番: {first_row['枠番']}")

if __name__ == "__main__":
    verify_payout_data()