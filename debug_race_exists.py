#!/usr/bin/env python3
"""
実際に存在するレースIDを確認
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def find_recent_races():
    """最近のレースを探す"""
    print("=== 最近の実際のレースIDを探索 ===")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 2024年の東京競馬場のレースを試す
    # 形式: YYYYppkkndrr (年+場所+回+日+レース番号)
    test_ids = [
        "202405010101",  # 2024年東京1回1日1R
        "202305010101",  # 2023年東京1回1日1R
        "202205010101",  # 2022年東京1回1日1R
    ]
    
    for race_id in test_ids:
        url = f"https://db.netkeiba.com/race/{race_id}"
        print(f"\n試行: {race_id}")
        print(f"URL: {url}")
        
        try:
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-jp")
            
            # レーステーブルの存在確認
            race_table = soup.find("table", {"class": "race_table_01"})
            if race_table:
                print(f"✓ レースデータ発見！")
                
                # レース名を取得
                race_name = soup.find("h1")
                if race_name:
                    print(f"  レース名: {race_name.text.strip()}")
                
                # 日付を取得
                date_elem = soup.find("p", {"class": "smalltxt"})
                if date_elem:
                    print(f"  日付情報: {date_elem.text.strip()[:20]}...")
                
                # 払戻テーブルの確認
                pay_table = soup.find("table", {"class": "pay_table_01"})
                if pay_table:
                    print(f"  ✓ 払戻データあり")
                    # 最初の払戻データを表示
                    first_row = pay_table.find("tr")
                    if first_row:
                        cells = first_row.find_all("td")
                        if cells:
                            print(f"    例: {' | '.join([c.text.strip() for c in cells[:3]])}")
                else:
                    print(f"  ✗ 払戻データなし")
                
                return race_id
            else:
                print(f"✗ レースデータなし")
                
        except Exception as e:
            print(f"✗ エラー: {e}")
    
    # カレンダーページから最新レースを探す
    print("\n\n=== カレンダーから最新レースを探索 ===")
    calendar_url = "https://race.netkeiba.com/top/calendar.html"
    
    try:
        r = requests.get(calendar_url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        
        # レースへのリンクを探す
        race_links = soup.find_all("a", href=lambda x: x and "race_id=" in x)[:5]
        
        print(f"最近のレース候補:")
        for link in race_links:
            href = link.get("href")
            if "race_id=" in href:
                race_id = href.split("race_id=")[1].split("&")[0]
                print(f"  - {race_id}: {link.text.strip()}")
        
    except Exception as e:
        print(f"カレンダー取得エラー: {e}")

if __name__ == "__main__":
    find_recent_races()