#!/usr/bin/env python3
"""
元のスクレイピングコードでテスト
"""

import sys
sys.path.append('src')

from data_processing.data_scraping import RaceScraper
import pandas as pd

def test_original():
    """元のスクレイピングコードでテスト"""
    print("=== 元のスクレイピングコードでテスト ===")
    
    scraper = RaceScraper(output_dir="test_data_original", max_workers=1)
    
    # 単一レースのテスト
    race_id = "202405010101"
    url = f"https://db.netkeiba.com/race/{race_id}"
    place_code = "05"
    place = "東京"
    
    print(f"レースID: {race_id}")
    print(f"URL: {url}")
    
    # レースデータを取得
    content = scraper.fetch_race_data(url)
    if content:
        print("✓ HTMLコンテンツ取得成功")
        
        # パース
        race_data = scraper.parse_race_data(race_id, content, place_code, place)
        
        if race_data:
            print(f"✓ {len(race_data)}頭のデータを取得")
            
            # DataFrameに変換して保存
            df = pd.DataFrame(race_data, columns=[
                'race_id', '馬', '騎手', '馬番', '調教師', '走破時間', 'オッズ', '通過順', 
                '着順', '体重', '体重変化', '性', '齢', '斤量', '賞金', '上がり', '人気', 
                'レース名', '日付', '開催', 'クラス', '芝・ダート', '距離', '回り', '馬場', 
                '天気', '場id', '場名'
            ])
            
            import os
            os.makedirs('test_data_original', exist_ok=True)
            df.to_excel('test_data_original/original_test.xlsx', index=False)
            print("✓ test_data_original/original_test.xlsx に保存")
            
            # 最初のデータを表示
            print("\n--- 1頭目のデータ ---")
            print(f"馬名: {race_data[0][1]}")
            print(f"着順: {race_data[0][8]}")
            print(f"騎手: {race_data[0][2]}")
            print(f"日付: {race_data[0][18]}")
            
        else:
            print("✗ レースデータの取得に失敗")
    else:
        print("✗ HTMLコンテンツの取得に失敗")

if __name__ == "__main__":
    test_original()