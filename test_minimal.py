#!/usr/bin/env python3
"""
最小限のテスト
"""

import sys
sys.path.append('src')

from data_processing.enhanced_scraping import EnhancedRaceScraper

def test_minimal():
    """最小限のテスト"""
    print("=== 最小限のテスト ===")
    
    scraper = EnhancedRaceScraper(output_dir="test_data", max_workers=1)
    
    race_id = "202405010101"
    url = f"https://db.netkeiba.com/race/{race_id}"
    
    print(f"URL: {url}")
    
    # HTMLを取得
    content = scraper.fetch_data(url)
    if content:
        print("✓ HTML取得成功")
        
        # パースを試みる
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser", from_encoding="euc-jp")
            
            # テーブルの存在確認
            table = soup.find("table", {"class": "race_table_01 nk_tb_common"})
            if table:
                print("✓ レーステーブル発見")
            else:
                print("✗ レーステーブルなし")
                # 他のテーブルを探す
                all_tables = soup.find_all("table")
                print(f"  テーブル総数: {len(all_tables)}")
                for i, t in enumerate(all_tables[:3]):
                    print(f"  テーブル{i}: class={t.get('class')}")
            
            # エラーの詳細を確認
            print("\nパース実行...")
            result = scraper.parse_race_data_enhanced(race_id, content, "05", "東京")
            print(f"結果: {len(result)}件のデータ")
            
        except Exception as e:
            print(f"✗ エラー: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ HTML取得失敗")

if __name__ == "__main__":
    test_minimal()