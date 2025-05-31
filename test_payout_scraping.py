#!/usr/bin/env python3
"""
払戻データ付きスクレイピングのテスト
"""

import sys
sys.path.append('src')

from data_processing.data_scraping_with_payout import RaceScraperWithPayout
import pandas as pd
import json

def test_payout_scraping():
    """払戻データ付きスクレイピングのテスト"""
    print("=== 払戻データ付きスクレイピングのテスト ===")
    
    scraper = RaceScraperWithPayout(output_dir="test_data_payout", max_workers=1)
    
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
            
            # DataFrameに変換
            df = pd.DataFrame(race_data, columns=[
                'race_id', '馬', '騎手', '馬番', '調教師', '走破時間', 'オッズ', '通過順', 
                '着順', '体重', '体重変化', '性', '齢', '斤量', '賞金', '上がり', '人気', 
                'レース名', '日付', '開催', 'クラス', '芝・ダート', '距離', '回り', '馬場', 
                '天気', '場id', '場名', '払戻データ', '枠番'
            ])
            
            # 最初の馬のデータを表示
            print("\n--- 1頭目のデータ ---")
            first_horse = df.iloc[0]
            print(f"馬名: {first_horse['馬']}")
            print(f"着順: {first_horse['着順']}")
            print(f"騎手: {first_horse['騎手']}")
            print(f"馬番: {first_horse['馬番']}")
            print(f"枠番: {first_horse['枠番']}")
            print(f"日付: {first_horse['日付']}")
            
            # 払戻データを解析
            print("\n--- 払戻データ ---")
            try:
                payout_json = first_horse['払戻データ']
                payout_data = json.loads(payout_json)
                
                for bet_type, payouts in payout_data.items():
                    if payouts:
                        print(f"\n{bet_type}:")
                        for combo, amount in payouts.items():
                            print(f"  {combo}: {amount}円")
            except Exception as e:
                print(f"払戻データの解析エラー: {e}")
            
            # 保存
            import os
            os.makedirs('test_data_payout', exist_ok=True)
            df.to_excel('test_data_payout/test_with_payout.xlsx', index=False)
            print("\n✓ test_data_payout/test_with_payout.xlsx に保存")
            
        else:
            print("✗ レースデータの取得に失敗")
    else:
        print("✗ HTMLコンテンツの取得に失敗")

def test_multiple_races():
    """複数レースのテスト"""
    print("\n\n=== 複数レースのテスト ===")
    
    scraper = RaceScraperWithPayout(output_dir="test_data_payout", max_workers=2)
    
    # 2024年1月27日東京の最初の3レースをテスト
    race_data_all = []
    
    for race_num in range(1, 4):  # 1R〜3Rまで
        race_id = f"20240501010{race_num:02d}"
        url = f"https://db.netkeiba.com/race/{race_id}"
        
        print(f"\nレース {race_num}/3: {race_id}")
        result = scraper.process_race((url, race_id, "05", "東京"))
        
        if result:
            race_data_all.extend(result)
            print(f"  ✓ {len(result)}頭のデータを取得")
        else:
            print(f"  ✗ データなし")
    
    if race_data_all:
        # DataFrameに変換
        df = pd.DataFrame(race_data_all, columns=[
            'race_id', '馬', '騎手', '馬番', '調教師', '走破時間', 'オッズ', '通過順', 
            '着順', '体重', '体重変化', '性', '齢', '斤量', '賞金', '上がり', '人気', 
            'レース名', '日付', '開催', 'クラス', '芝・ダート', '距離', '回り', '馬場', 
            '天気', '場id', '場名', '払戻データ', '枠番'
        ])
        
        print(f"\n合計: {len(df)}行のデータを取得")
        
        # 払戻データの統計
        print("\n--- 払戻データの統計 ---")
        payout_counts = {}
        
        for _, row in df.iterrows():
            try:
                payout_data = json.loads(row['払戻データ'])
                for bet_type, payouts in payout_data.items():
                    if payouts:
                        if bet_type not in payout_counts:
                            payout_counts[bet_type] = 0
                        payout_counts[bet_type] += 1
            except:
                pass
        
        for bet_type, count in payout_counts.items():
            print(f"{bet_type}: {count}レースでデータあり")

if __name__ == "__main__":
    test_payout_scraping()
    test_multiple_races()