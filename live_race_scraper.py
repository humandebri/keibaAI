#!/usr/bin/env python3
"""
ライブレース情報スクレイパー
指定されたrace_idからリアルタイムレースデータを取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import argparse
import sys
import time
import random
from typing import Optional, Dict, List, Any


class LiveRaceScraper:
    """netkeiba.comからライブレース情報を取得するクラス"""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.67"
    ]
    
    PLACE_DICT = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }
    
    def __init__(self, timeout: int = 10):
        """
        Args:
            timeout: リクエストタイムアウト秒数
        """
        self.timeout = timeout
    
    def fetch_race_data(self, race_id: str, max_retries: int = 3) -> Optional[bytes]:
        """指定されたrace_idからレースデータを取得"""
        url = f"https://db.netkeiba.com/race/{race_id}"
        
        retries = 0
        while retries < max_retries:
            try:
                headers = {'User-Agent': random.choice(self.USER_AGENTS)}
                print(f"📡 データ取得中: {url}")
                
                r = requests.get(url, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                
                print(f"✅ データ取得成功")
                return r.content
                
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"❌ 取得エラー (試行 {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    wait_time = random.uniform(1, 3)
                    print(f"⏳ {wait_time:.1f}秒待機...")
                    time.sleep(wait_time)
        
        print(f"❌ データ取得失敗: {race_id}")
        return None
    
    def extract_odds_data(self, soup) -> Dict[str, float]:
        """オッズデータを抽出"""
        odds_data = {}
        
        try:
            # オッズテーブルを探す
            odds_table = soup.find("table", {"class": "race_table_01 nk_tb_common"})
            if not odds_table:
                return odds_data
            
            for row in odds_table.find_all("tr")[1:]:  # ヘッダ行をスキップ
                cols = row.find_all("td")
                if len(cols) >= 13:
                    horse_num = cols[2].text.strip()
                    horse_name = cols[3].text.strip()
                    odds_text = cols[12].text.strip()
                    
                    # オッズを数値に変換
                    try:
                        odds = float(odds_text)
                        odds_data[f"{horse_num}番_{horse_name}"] = odds
                    except (ValueError, TypeError):
                        odds_data[f"{horse_num}番_{horse_name}"] = 0.0
            
        except Exception as e:
            print(f"⚠️ オッズデータ抽出エラー: {e}")
        
        return odds_data
    
    def extract_race_info(self, soup, race_id: str) -> Dict[str, Any]:
        """レース基本情報を抽出"""
        race_info = {
            'race_name': '',
            'date': '',
            'course': '',
            'distance': '',
            'class': '',
            'surface': '',
            'direction': '',
            'weather': '',
            'track_condition': '',
            'place': ''
        }
        
        try:
            # レース名
            title_elem = soup.find("h1")
            if title_elem:
                race_info['race_name'] = title_elem.text.strip()
            
            # 詳細情報
            detail_elems = soup.find_all("p", class_="smalltxt")
            if detail_elems:
                detail_text = detail_elems[0].text.strip()
                parts = detail_text.split()
                if len(parts) >= 3:
                    race_info['date'] = parts[0]
                    race_info['course'] = parts[1]
                    race_info['class'] = parts[2]
            
            # コース情報（距離、芝/ダート、回り）
            span_elems = soup.find_all("span")
            for span in span_elems:
                span_text = span.text.strip()
                if 'm' in span_text and ('芝' in span_text or 'ダ' in span_text):
                    # 距離抽出
                    try:
                        race_info['distance'] = ''.join(filter(str.isdigit, span_text.split('m')[0]))
                    except:
                        pass
                    
                    # 芝/ダート
                    if '芝' in span_text:
                        race_info['surface'] = '芝'
                    elif 'ダ' in span_text:
                        race_info['surface'] = 'ダート'
                    
                    # 回り
                    if '右' in span_text:
                        race_info['direction'] = '右'
                    elif '左' in span_text:
                        race_info['direction'] = '左'
                    
                    # 天気と馬場状態
                    if '晴' in span_text:
                        race_info['weather'] = '晴'
                    elif '曇' in span_text:
                        race_info['weather'] = '曇'
                    elif '雨' in span_text:
                        race_info['weather'] = '雨'
                    
                    if '良' in span_text:
                        race_info['track_condition'] = '良'
                    elif '稍重' in span_text:
                        race_info['track_condition'] = '稍重'
                    elif '重' in span_text:
                        race_info['track_condition'] = '重'
                    elif '不良' in span_text:
                        race_info['track_condition'] = '不良'
            
            # 開催場所
            place_code = race_id[:6][4:6]
            race_info['place'] = self.PLACE_DICT.get(place_code, '不明')
            
            # レースIDを追加
            race_info['race_id'] = race_id
            
        except Exception as e:
            print(f"⚠️ レース情報抽出エラー: {e}")
        
        return race_info
    
    def extract_horse_data(self, soup) -> List[Dict[str, Any]]:
        """出走馬データを抽出"""
        horses_data = []
        
        try:
            main_table = soup.find("table", {"class": "race_table_01 nk_tb_common"})
            if not main_table:
                print("❌ メインテーブルが見つかりません")
                return horses_data
            
            for row in main_table.find_all("tr")[1:]:  # ヘッダ行をスキップ
                cols = row.find_all("td")
                if len(cols) < 15:
                    continue
                
                horse_data = {}
                
                try:
                    # 基本データ
                    horse_data['馬番'] = cols[2].text.strip()
                    horse_data['馬名'] = cols[3].text.strip()
                    horse_data['性齢'] = cols[4].text.strip()
                    horse_data['斤量'] = cols[5].text.strip()
                    horse_data['騎手'] = cols[6].text.strip()
                    
                    # 馬体重処理
                    weight_text = cols[14].text.strip()
                    if '(' in weight_text and ')' in weight_text:
                        weight_parts = weight_text.split('(')
                        horse_data['馬体重'] = weight_parts[0].strip()
                        horse_data['馬体重変化'] = weight_parts[1].replace(')', '').strip()
                    else:
                        horse_data['馬体重'] = weight_text
                        horse_data['馬体重変化'] = '0'
                    
                    # オッズ
                    horse_data['単勝オッズ'] = cols[12].text.strip()
                    
                    # 人気
                    horse_data['人気'] = cols[13].text.strip() if len(cols) > 13 else ''
                    
                    # 枠番（馬番から計算）
                    try:
                        horse_num = int(horse_data['馬番'])
                        horse_data['枠'] = ((horse_num - 1) // 2) + 1
                    except:
                        horse_data['枠'] = 0
                    
                    # 調教師
                    if len(cols) > 18:
                        trainer_elem = cols[18].find('a')
                        horse_data['調教師'] = trainer_elem.text.strip() if trainer_elem else ''
                    else:
                        horse_data['調教師'] = ''
                    
                    horses_data.append(horse_data)
                    
                except Exception as e:
                    print(f"⚠️ 馬データ抽出エラー: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ 出走馬データ抽出エラー: {e}")
        
        return horses_data
    
    def scrape_race(self, race_id: str) -> Optional[Dict[str, Any]]:
        """指定されたrace_idのレースデータを完全取得"""
        print(f"🏇 レースデータ取得開始: {race_id}")
        
        # HTMLデータ取得
        content = self.fetch_race_data(race_id)
        if not content:
            return None
        
        try:
            soup = BeautifulSoup(content, "html.parser", from_encoding="euc-jp")
            
            # レース基本情報
            race_info = self.extract_race_info(soup, race_id)
            
            # 出走馬データ
            horses_data = self.extract_horse_data(soup)
            
            # オッズデータ
            odds_data = self.extract_odds_data(soup)
            
            result = {
                'race_info': race_info,
                'horses': horses_data,
                'odds': odds_data,
                'horse_count': len(horses_data)
            }
            
            print(f"✅ データ取得完了")
            print(f"   レース名: {race_info['race_name']}")
            print(f"   出走頭数: {len(horses_data)}頭")
            print(f"   距離: {race_info['distance']}m")
            print(f"   コース: {race_info['surface']}{race_info['direction']}")
            
            return result
            
        except Exception as e:
            print(f"❌ データ解析エラー: {e}")
            return None
    
    def save_to_csv(self, race_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """レースデータをCSVファイルに保存"""
        if not filename:
            race_id = race_data['race_info']['race_id']
            filename = f"live_race_data_{race_id}.csv"
        
        try:
            # 出走馬データをDataFrameに変換
            df = pd.DataFrame(race_data['horses'])
            
            # レース情報を各行に追加
            race_info = race_data['race_info']
            for key, value in race_info.items():
                if key not in df.columns:
                    df[key] = value
            
            # CSVに保存
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"💾 CSVファイル保存完了: {filename}")
            print(f"   行数: {len(df)}")
            print(f"   列数: {len(df.columns)}")
            
            return filename
            
        except Exception as e:
            print(f"❌ CSV保存エラー: {e}")
            return ""
    
    def save_to_json(self, race_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """レースデータをJSONファイルに保存"""
        if not filename:
            race_id = race_data['race_info']['race_id']
            filename = f"live_race_data_{race_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(race_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 JSONファイル保存完了: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ JSON保存エラー: {e}")
            return ""


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='ライブレース情報スクレイパー')
    parser.add_argument('race_id', help='レースID (例: 202505021212)')
    parser.add_argument('--output', choices=['csv', 'json', 'both'], default='csv', 
                       help='出力形式 (デフォルト: csv)')
    parser.add_argument('--filename', help='出力ファイル名（拡張子なし）')
    parser.add_argument('--timeout', type=int, default=10, help='タイムアウト秒数')
    
    args = parser.parse_args()
    
    # バリデーション
    if len(args.race_id) != 12 or not args.race_id.isdigit():
        print("❌ 無効なレースID形式です。12桁の数字を入力してください。")
        print("   例: 202505021212")
        sys.exit(1)
    
    print("🏇 ライブレーススクレイパー開始")
    print(f"   レースID: {args.race_id}")
    print(f"   出力形式: {args.output}")
    
    # スクレイピング実行
    scraper = LiveRaceScraper(timeout=args.timeout)
    race_data = scraper.scrape_race(args.race_id)
    
    if not race_data:
        print("❌ レースデータの取得に失敗しました")
        sys.exit(1)
    
    # ファイル保存
    saved_files = []
    
    if args.output in ['csv', 'both']:
        csv_filename = args.filename + '.csv' if args.filename else None
        csv_file = scraper.save_to_csv(race_data, csv_filename)
        if csv_file:
            saved_files.append(csv_file)
    
    if args.output in ['json', 'both']:
        json_filename = args.filename + '.json' if args.filename else None
        json_file = scraper.save_to_json(race_data, json_filename)
        if json_file:
            saved_files.append(json_file)
    
    print(f"\n🎉 処理完了！")
    if saved_files:
        print("保存されたファイル:")
        for file in saved_files:
            print(f"  📄 {file}")
    
    # 簡単な統計表示
    print(f"\n📊 レース概要:")
    print(f"   レース名: {race_data['race_info']['race_name']}")
    print(f"   開催日: {race_data['race_info']['date']}")
    print(f"   開催場: {race_data['race_info']['place']}")
    print(f"   距離: {race_data['race_info']['distance']}m")
    print(f"   出走頭数: {race_data['horse_count']}頭")


if __name__ == "__main__":
    main()