#!/usr/bin/env python3
"""
オッズテーブルの内容を詳細に調査するデバッグツール
"""

import requests
from bs4 import BeautifulSoup
import time
import random

def debug_odds_table(race_id: str):
    """オッズテーブルの実際の内容を詳細に調査"""
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.5,en;q=0.3',
    })
    
    print(f"🔍 レース {race_id} のオッズテーブル内容を詳細調査")
    
    url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"
    
    try:
        time.sleep(random.uniform(1.0, 2.0))
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # すべてのテーブルを詳細チェック
        tables = soup.find_all('table')
        print(f"総テーブル数: {len(tables)}")
        
        for i, table in enumerate(tables):
            print(f"\n📊 テーブル {i+1}:")
            
            # ヘッダー行
            header_row = table.find('tr')
            if header_row:
                header_cells = header_row.find_all(['th', 'td'])
                header_texts = [cell.get_text(strip=True) for cell in header_cells]
                print(f"ヘッダー: {header_texts}")
                
                # 単勝オッズテーブルかチェック
                if ('人気' in header_texts and ('単勝オッズ' in header_texts or 'オッズ' in ' '.join(header_texts))):
                    print("🎯 これが単勝オッズテーブルです！")
                    
                    # 全データ行を詳細表示
                    rows = table.find_all('tr')
                    print(f"総行数: {len(rows)} (ヘッダー含む)")
                    
                    for j, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        if j == 0:
                            print(f"行{j} (ヘッダー): {[cell.get_text(strip=True) for cell in cells]}")
                        else:
                            cell_data = []
                            for k, cell in enumerate(cells):
                                text = cell.get_text(strip=True)
                                classes = cell.get('class', [])
                                cell_data.append(f"'{text}'({classes})")
                            print(f"行{j}: {cell_data}")
                            
                            if j >= 20:  # 最初の20行のみ表示
                                print("... (以下省略)")
                                break
                    
                    print("\n" + "="*50)
                    
        # ページのJavaScript部分も確認
        print("\n🔍 JavaScript内容確認:")
        scripts = soup.find_all('script')
        
        for i, script in enumerate(scripts):
            if script.string and ('odds' in script.string.lower() or '倍' in script.string):
                content = script.string
                print(f"\nスクリプト{i} (オッズ関連):")
                # 関連部分のみ抽出
                lines = content.split('\n')
                for line in lines:
                    if 'odds' in line.lower() or '倍' in line or 'popular' in line.lower():
                        print(f"  {line.strip()}")
                
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用法: python debug_odds_table.py <race_id>")
        sys.exit(1)
    
    race_id = sys.argv[1]
    debug_odds_table(race_id)