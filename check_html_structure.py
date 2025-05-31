#!/usr/bin/env python3
"""
HTMLの構造を確認
"""

import requests
from bs4 import BeautifulSoup

def check_structure():
    race_id = "202405010101"
    url = f"https://db.netkeiba.com/race/{race_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-jp")
    
    print("=== HTMLの構造確認 ===")
    
    # 1. タイトル
    h1s = soup.find_all("h1")
    print(f"\n<h1>タグ数: {len(h1s)}")
    for i, h1 in enumerate(h1s[:3]):
        print(f"  h1[{i}]: {h1.text.strip()[:50]}")
    
    # 2. レーステーブルのクラス名を確認
    print("\n\nレーステーブルのクラス名:")
    tables = soup.find_all("table")
    for i, table in enumerate(tables):
        class_names = table.get("class", [])
        # レース結果っぽいテーブルを探す
        if any(word in str(table) for word in ["着順", "馬名", "騎手"]):
            print(f"  テーブル{i}: class={class_names}")
            # ヘッダーを表示
            headers = table.find_all("th")[:5]
            if headers:
                print(f"    ヘッダー: {[h.text.strip() for h in headers]}")
    
    # 3. spanタグの確認
    print("\n\nspanタグの確認:")
    spans = soup.find_all("span")[:20]
    print(f"spanタグ数: {len(soup.find_all('span'))}")
    
    # レース情報が含まれそうなspanを探す
    for i, span in enumerate(spans):
        text = span.text.strip()
        if any(word in text for word in ["芝", "ダート", "m", "良", "稍重"]):
            print(f"  span[{i}]: {text}")

if __name__ == "__main__":
    check_structure()