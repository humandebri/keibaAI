#!/usr/bin/env python3
"""
払戻データ取得の最終デバッグ
"""

import sys
sys.path.append('src')

from data_processing.data_scraping_with_payout import RaceScraperWithPayout
from bs4 import BeautifulSoup
import json

def debug_payout():
    scraper = RaceScraperWithPayout()
    
    race_id = "202405010101"
    url = f"https://db.netkeiba.com/race/{race_id}"
    
    print(f"URL: {url}")
    
    # HTMLを取得
    content = scraper.fetch_race_data(url)
    if not content:
        print("✗ コンテンツ取得失敗")
        return
    
    print("✓ コンテンツ取得成功")
    
    # スープを作成
    soup = BeautifulSoup(content, "html.parser", from_encoding="euc-jp")
    
    # 払戻テーブルを探す
    print("\n=== 払戻テーブルの探索 ===")
    pay_tables = soup.find_all("table", {"class": "pay_table_01"})
    print(f"pay_table_01テーブル数: {len(pay_tables)}")
    
    # 払戻データを抽出
    payout_data = scraper.extract_payout_data(soup)
    
    print("\n=== 抽出された払戻データ ===")
    print(json.dumps(payout_data, ensure_ascii=False, indent=2))
    
    # デバッグ: テーブルの中身を詳しく見る
    if pay_tables:
        print("\n=== 最初のテーブルの詳細 ===")
        table = pay_tables[0]
        rows = table.find_all("tr")
        
        for i, row in enumerate(rows[:5]):
            print(f"\n行{i}:")
            cells = row.find_all("td")
            for j, cell in enumerate(cells):
                print(f"  セル{j}: '{cell.text.strip()}'")
                # 改行が含まれているか確認
                if '\n' in cell.text:
                    print(f"    → 改行あり: {repr(cell.text)}")

if __name__ == "__main__":
    debug_payout()