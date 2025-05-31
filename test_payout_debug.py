#!/usr/bin/env python3
"""
払戻データ取得の詳細デバッグ
"""

import requests
from bs4 import BeautifulSoup

def debug_payout_detailed():
    race_id = "202405010101"
    url = f"https://db.netkeiba.com/race/{race_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-jp")
    
    print("=== 払戻テーブルの詳細確認 ===")
    
    # pay_table_01クラスのテーブルを探す
    pay_tables = soup.find_all("table", {"class": "pay_table_01"})
    print(f"\npay_table_01テーブル数: {len(pay_tables)}")
    
    if pay_tables:
        for i, table in enumerate(pay_tables):
            print(f"\n--- テーブル{i+1} ---")
            rows = table.find_all("tr")
            print(f"行数: {len(rows)}")
            
            # 最初の5行を表示
            for j, row in enumerate(rows[:5]):
                cells = row.find_all(["td", "th"])
                cell_contents = [cell.text.strip() for cell in cells]
                print(f"行{j}: {' | '.join(cell_contents)}")
    
    # すべてのテーブルから払戻情報を探す
    print("\n\n=== 全テーブルから払戻情報を探索 ===")
    all_tables = soup.find_all("table")
    
    for i, table in enumerate(all_tables):
        table_text = table.text
        if any(word in table_text for word in ["単勝", "複勝", "馬連", "払戻", "円"]):
            print(f"\n--- 払戻関連テーブル{i} ---")
            print(f"class: {table.get('class')}")
            
            # 最初の数行を確認
            rows = table.find_all("tr")[:3]
            for row in rows:
                cells = row.find_all(["td", "th"])
                print(f"  {' | '.join([c.text.strip() for c in cells])}")

if __name__ == "__main__":
    debug_payout_detailed()