#!/usr/bin/env python3
"""
スクレイピングのテスト実行
少数のレースだけ取得して動作確認
"""

import sys
sys.path.append('src')

from data_processing.enhanced_scraping import EnhancedRaceScraper
import json
import pandas as pd

def test_single_race():
    """単一レースのテスト"""
    print("=== 単一レースのテスト ===")
    
    scraper = EnhancedRaceScraper(output_dir="test_data", max_workers=1)
    
    # 2024年1月の東京1Rをテスト（実際に存在するレース）
    race_id = "202405010101"  # 2024年東京1回1日1R
    url = f"https://db.netkeiba.com/race/{race_id}"
    place_code = "05"
    place = "東京"
    
    print(f"レースID: {race_id}")
    print(f"URL: {url}")
    
    # レースデータを取得
    content = scraper.fetch_data(url)
    if content:
        print("✓ HTMLコンテンツ取得成功")
        
        # パース
        race_data = scraper.parse_race_data_enhanced(race_id, content, place_code, place)
        
        if race_data:
            print(f"✓ {len(race_data)}頭のデータを取得")
            
            # 最初の馬のデータを表示
            print("\n--- 1頭目のデータサンプル ---")
            first_horse = race_data[0]
            
            # 基本情報
            print("\n【基本情報】")
            for key in ['馬', '馬番', '枠番', '着順', '騎手', '調教師']:
                if key in first_horse:
                    print(f"  {key}: {first_horse[key]}")
            
            # 日付情報
            print(f"\n【日付】: {first_horse.get('日付', 'なし')}")
            
            # 払戻情報の確認
            print("\n【払戻データ】")
            payout_types = ['払戻_単勝', '払戻_複勝', '払戻_馬連', '払戻_ワイド']
            for ptype in payout_types:
                if ptype in first_horse and first_horse[ptype]:
                    print(f"  {ptype}: {first_horse[ptype]}")
            
            # 全データをJSONで保存
            with open('test_data/sample_race.json', 'w', encoding='utf-8') as f:
                json.dump(race_data, f, ensure_ascii=False, indent=2)
            print("\n✓ test_data/sample_race.json に保存")
            
        else:
            print("✗ レースデータの取得に失敗")
    else:
        print("✗ HTMLコンテンツの取得に失敗")

def test_multiple_races():
    """複数レース（5レースのみ）のテスト"""
    print("\n\n=== 複数レースのテスト（5レースのみ） ===")
    
    scraper = EnhancedRaceScraper(output_dir="test_data", max_workers=2)
    
    # 2024年1月1日の東京の最初の5レースをテスト
    race_data_all = []
    urls_data = []
    
    for race_num in range(1, 6):  # 1R〜5Rまで
        race_id = f"202405010101{race_num:02d}"
        url = f"https://db.netkeiba.com/race/{race_id}"
        urls_data.append((url, race_id, "05", "東京"))
    
    print(f"テストするレース数: {len(urls_data)}")
    
    # 各レースを処理
    for i, (url, race_id, place_code, place) in enumerate(urls_data):
        print(f"\nレース {i+1}/5: {race_id}")
        try:
            result = scraper.process_race_enhanced((url, race_id, place_code, place))
            if result:
                race_data_all.extend(result)
                print(f"  ✓ {len(result)}頭のデータを取得")
            else:
                print(f"  ✗ データなし")
        except Exception as e:
            print(f"  ✗ エラー: {e}")
    
    # DataFrameに変換
    if race_data_all:
        df = pd.DataFrame(race_data_all)
        print(f"\n合計: {len(df)}行のデータを取得")
        
        # カラムの確認
        print("\n【取得できたカラム】")
        for col in df.columns[:20]:  # 最初の20カラムのみ表示
            print(f"  - {col}")
        
        if len(df.columns) > 20:
            print(f"  ... 他 {len(df.columns) - 20} カラム")
        
        # Excelに保存
        df.to_excel('test_data/test_races.xlsx', index=False)
        print("\n✓ test_data/test_races.xlsx に保存")
        
        # 払戻データの確認
        print("\n【払戻データの取得状況】")
        payout_cols = [col for col in df.columns if '払戻' in col]
        for col in payout_cols:
            non_empty = df[col].apply(lambda x: len(x) > 0 if isinstance(x, dict) else False).sum()
            print(f"  {col}: {non_empty}/{len(df)} レースでデータあり")

def main():
    """テスト実行"""
    import os
    os.makedirs('test_data', exist_ok=True)
    
    print("スクレイピングテストを開始します...\n")
    
    # 単一レースのテスト
    test_single_race()
    
    # 複数レースのテスト
    test_multiple_races()
    
    print("\n\nテスト完了！")
    print("test_data/ フォルダを確認してください。")

if __name__ == "__main__":
    main()