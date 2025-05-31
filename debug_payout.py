#!/usr/bin/env python3
"""
払戻データ取得のデバッグ
"""

import requests
from bs4 import BeautifulSoup
import json

def debug_payout_fetch():
    """払戻ページの構造を調査"""
    race_id = "202405010101"
    
    # まず通常のレースページを確認
    print("=== レース結果ページの確認 ===")
    race_url = f"https://db.netkeiba.com/race/{race_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    r = requests.get(race_url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-jp")
    
    # 払戻関連の要素を探す
    print("\n【払戻関連の要素を探索】")
    
    # 1. classにpayが含まれる要素
    pay_elements = soup.find_all(class_=lambda x: x and 'pay' in x.lower() if x else False)
    print(f"'pay'を含むclass: {len(pay_elements)}個")
    for elem in pay_elements[:3]:
        print(f"  - {elem.get('class')}: {elem.name}")
    
    # 2. 払戻テーブルを探す
    tables = soup.find_all("table")
    print(f"\nテーブル総数: {len(tables)}")
    
    for i, table in enumerate(tables):
        # テーブルの中身を確認
        if any(keyword in str(table) for keyword in ['単勝', '複勝', '馬連', '払戻', '的中']):
            print(f"\nテーブル{i}: 払戻関連の可能性あり")
            print(f"  class: {table.get('class')}")
            # 最初の数行を表示
            rows = table.find_all("tr")[:3]
            for row in rows:
                cells = row.find_all(['td', 'th'])
                print(f"    {' | '.join([cell.text.strip() for cell in cells[:5]])}")
    
    # 3. 払戻金額を直接探す
    print("\n【金額らしき数値を探索】")
    import re
    money_pattern = re.compile(r'[\d,]+円')
    money_matches = money_pattern.findall(str(soup))
    print(f"金額表記: {len(money_matches)}個")
    print(f"例: {money_matches[:5]}")
    
    # 4. レース後の結果ページを試す
    print("\n=== 別のURL形式を試す ===")
    result_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    print(f"URL: {result_url}")
    
    try:
        r2 = requests.get(result_url, headers=headers)
        if r2.status_code == 200:
            print("✓ アクセス成功")
            soup2 = BeautifulSoup(r2.content, "html.parser")
            
            # 払戻テーブルを探す
            pay_tables = soup2.find_all("table", class_=lambda x: x and 'pay' in str(x).lower() if x else False)
            print(f"払戻テーブル候補: {len(pay_tables)}個")
            
            # 払戻金額を含むdivを探す
            pay_blocks = soup2.find_all("div", class_=lambda x: x and 'pay' in str(x).lower() if x else False)
            print(f"払戻ブロック候補: {len(pay_blocks)}個")
        else:
            print(f"✗ アクセス失敗: {r2.status_code}")
    except Exception as e:
        print(f"✗ エラー: {e}")

if __name__ == "__main__":
    debug_payout_fetch()