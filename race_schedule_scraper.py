#!/usr/bin/env python3
"""
netkeiba.com レース開催日程スクレイパー
今日・明日・今週末のレース一覧を取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict


class RaceScheduleScraper:
    """netkeiba.com からレース開催日程を取得"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://race.netkeiba.com"
        
        # 競馬場コード
        self.course_codes = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
            '05': '東京', '06': '中山', '07': '中京', '08': '京都',
            '09': '阪神', '10': '小倉', '11': '未確定', '12': '東京',
            '13': '盛岡', '14': '大井', '15': '船橋', '16': '川崎', '17': '浦和'
        }
    
    def get_today_races(self) -> pd.DataFrame:
        """今日のレース一覧を取得"""
        today = datetime.now()
        return self._get_races_by_date(today)
    
    def get_tomorrow_races(self) -> pd.DataFrame:
        """明日のレース一覧を取得"""
        tomorrow = datetime.now() + timedelta(days=1)
        return self._get_races_by_date(tomorrow)
    
    def get_weekend_races(self) -> pd.DataFrame:
        """今週末(土日)のレース一覧を取得"""
        today = datetime.now()
        
        # 今週の土曜日を計算
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0 and today.weekday() == 5:  # 今日が土曜日
            saturday = today
        else:
            saturday = today + timedelta(days=days_until_saturday)
        
        sunday = saturday + timedelta(days=1)
        
        # 土日のレースを取得
        saturday_races = self._get_races_by_date(saturday)
        sunday_races = self._get_races_by_date(sunday)
        
        # 結合
        all_races = pd.concat([saturday_races, sunday_races], ignore_index=True)
        return all_races.sort_values(['日付', '競馬場', 'レース番号'])
    
    def _get_races_by_date(self, target_date: datetime) -> pd.DataFrame:
        """指定日のレース一覧を取得"""
        date_str = target_date.strftime('%Y%m%d')
        url = f"{self.base_url}/top/race_list_sub.html?kaisai_date={date_str}"
        
        print(f"📅 {target_date.strftime('%Y年%m月%d日')}のレース情報取得中...")
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            races = self._parse_race_list(soup, target_date)
            print(f"✅ {len(races)}レースを取得")
            return pd.DataFrame(races)
            
        except Exception as e:
            print(f"❌ {target_date.strftime('%Y年%m月%d日')}のレース取得エラー: {e}")
            return pd.DataFrame()
    
    def _parse_race_list(self, soup: BeautifulSoup, target_date: datetime) -> List[Dict]:
        """レース一覧ページから情報を抽出"""
        races = []
        
        # 競馬場ごとのレース情報を探す
        race_sections = soup.find_all('div', class_='race_kaisai_info')
        if not race_sections:
            # 別のパターンを試す
            race_sections = soup.find_all('div', class_='kaisai_info')
        
        for section in race_sections:
            # 競馬場名を取得
            course_name = self._extract_course_name(section)
            if not course_name:
                continue
            
            # そのセクション内のレースリストを取得
            race_items = section.find_all('a', href=re.compile(r'/race/shutuba\.html\?race_id='))
            
            for race_item in race_items:
                race_info = self._extract_race_info(race_item, course_name, target_date)
                if race_info:
                    races.append(race_info)
        
        # レースが見つからない場合、別の方法で探す
        if not races:
            races = self._fallback_race_extraction(soup, target_date)
        
        return races
    
    def _extract_course_name(self, section) -> str:
        """セクションから競馬場名を抽出"""
        # 競馬場名のパターンを複数試す
        patterns = [
            section.find('h3'),
            section.find('h2'),
            section.find('div', class_='kaisai_name'),
            section.find('span', class_='kaisai_name'),
        ]
        
        for elem in patterns:
            if elem:
                text = elem.get_text(strip=True)
                # 競馬場名を抽出（例：「東京競馬場」→「東京」）
                for course_code, course_name in self.course_codes.items():
                    if course_name in text:
                        return course_name
        
        return ""
    
    def _extract_race_info(self, race_item, course_name: str, target_date: datetime) -> Dict:
        """個別レース情報を抽出"""
        try:
            # race_idをURLから抽出
            href = race_item.get('href', '')
            race_id_match = re.search(r'race_id=(\d{12})', href)
            if not race_id_match:
                return None
            
            race_id = race_id_match.group(1)
            
            # レース番号を抽出
            race_text = race_item.get_text(strip=True)
            race_num_match = re.search(r'(\d{1,2})R', race_text)
            race_num = int(race_num_match.group(1)) if race_num_match else 0
            
            # レース名を抽出
            race_name = race_text.replace(f'{race_num}R', '').strip() if race_num else race_text
            
            # 発走時刻を抽出
            time_match = re.search(r'(\d{1,2}):(\d{2})', race_text)
            start_time = f"{time_match.group(1)}:{time_match.group(2)}" if time_match else "不明"
            
            return {
                'race_id': race_id,
                '日付': target_date.strftime('%Y-%m-%d'),
                '曜日': target_date.strftime('%A'),
                '競馬場': course_name,
                'レース番号': race_num,
                'レース名': race_name,
                '発走時刻': start_time,
                'URL': f"{self.base_url}{href}"
            }
            
        except Exception as e:
            print(f"⚠️ レース情報抽出エラー: {e}")
            return None
    
    def _fallback_race_extraction(self, soup: BeautifulSoup, target_date: datetime) -> List[Dict]:
        """フォールバック：別の方法でレースを探す"""
        races = []
        
        # 全てのrace_idリンクを探す
        race_links = soup.find_all('a', href=re.compile(r'race_id=\d{12}'))
        
        for link in race_links:
            href = link.get('href', '')
            race_id_match = re.search(r'race_id=(\d{12})', href)
            if race_id_match:
                race_id = race_id_match.group(1)
                
                # race_idから競馬場を推測
                course_code = race_id[8:10]
                course_name = self.course_codes.get(course_code, '不明')
                
                race_text = link.get_text(strip=True)
                
                races.append({
                    'race_id': race_id,
                    '日付': target_date.strftime('%Y-%m-%d'),
                    '曜日': target_date.strftime('%A'),
                    '競馬場': course_name,
                    'レース番号': 0,
                    'レース名': race_text,
                    '発走時刻': '不明',
                    'URL': f"{self.base_url}{href}"
                })
        
        return races


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='レース開催日程取得')
    parser.add_argument('--target', choices=['today', 'tomorrow', 'weekend'], 
                       default='today', help='取得対象')
    parser.add_argument('--output', type=str, default='race_schedule.csv', 
                       help='出力CSVファイル')
    
    args = parser.parse_args()
    
    scraper = RaceScheduleScraper()
    
    if args.target == 'today':
        races = scraper.get_today_races()
    elif args.target == 'tomorrow':
        races = scraper.get_tomorrow_races()
    else:  # weekend
        races = scraper.get_weekend_races()
    
    if races.empty:
        print("❌ レース情報が取得できませんでした")
        return
    
    # CSV保存
    races.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n💾 レース日程保存: {args.output}")
    
    # 結果表示
    print(f"\n📊 取得結果: {len(races)}レース")
    print("\n🏇 レース一覧:")
    for _, race in races.iterrows():
        print(f"  {race['日付']} {race['競馬場']} {race['レース番号']:2d}R "
              f"{race['発走時刻']} {race['レース名']}")
        print(f"    ID: {race['race_id']}")


if __name__ == "__main__":
    main()