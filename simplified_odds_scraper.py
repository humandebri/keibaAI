#!/usr/bin/env python3
"""
netkeiba.com 簡潔オッズスクレイパー
実際のオッズページ構造に特化した確実な取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from typing import Dict, List, Optional


class SimplifiedOddsScraper:
    """実際のオッズページ構造に基づく確実なスクレイパー"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def scrape_race_with_odds(self, race_id: str) -> pd.DataFrame:
        """基本情報とオッズを確実に取得"""
        print(f"🏇 レース {race_id} のデータ取得開始")
        
        # 1. 出馬表から基本情報を取得
        basic_data = self._get_basic_shutuba_data(race_id)
        if basic_data.empty:
            print("❌ 基本データ取得失敗")
            return pd.DataFrame()
        
        # 2. オッズページから確定オッズを取得
        odds_data = self._get_confirmed_odds(race_id)
        
        # 3. データを統合
        final_data = self._merge_data(basic_data, odds_data)
        
        print(f"✅ 最終データ: {len(final_data)}頭取得完了")
        return final_data
    
    def _get_basic_shutuba_data(self, race_id: str) -> pd.DataFrame:
        """出馬表から基本情報のみ取得"""
        print("📋 出馬表から基本情報取得中...")
        
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(0.5, 1.0))
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='Shutuba_Table')
            
            if not table:
                return pd.DataFrame()
            
            horses_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    # 馬番確認
                    if len(cells) > 1:
                        umaban_text = cells[1].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            horse_data = self._extract_shutuba_basic_data(cells, race_id)
                            if horse_data:
                                horses_data.append(horse_data)
            
            df = pd.DataFrame(horses_data)
            print(f"✓ 基本情報: {len(df)}頭取得成功")
            return df
            
        except Exception as e:
            print(f"❌ 出馬表エラー: {e}")
            return pd.DataFrame()
    
    def _extract_shutuba_basic_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """出馬表から基本情報のみ抽出"""
        try:
            # 枠番（セル0）
            waku = 1
            waku_text = cells[0].get_text(strip=True)
            if waku_text.isdigit() and 1 <= int(waku_text) <= 8:
                waku = int(waku_text)
            
            # 馬番（セル1）
            umaban = int(cells[1].get_text(strip=True))
            
            # 馬名（セル3）
            horse_name = "不明"
            if len(cells) > 3:
                horse_cell = cells[3]
                horse_link = horse_cell.find('a', href=lambda href: href and 'horse' in href)
                if horse_link:
                    horse_name = horse_link.get_text(strip=True)
            
            # 性齢（セル4）
            sei_rei = cells[4].get_text(strip=True) if len(cells) > 4 else "不明"
            
            # 斤量（セル5）
            kinryo = 57.0
            if len(cells) > 5:
                kinryo_text = cells[5].get_text(strip=True)
                if re.match(r'^5[0-9]\.[05]$', kinryo_text):
                    kinryo = float(kinryo_text)
            
            # 騎手（セル6）
            jockey = "不明"
            if len(cells) > 6:
                jockey_cell = cells[6]
                jockey_link = jockey_cell.find('a')
                if jockey_link:
                    jockey = jockey_link.get_text(strip=True)
                else:
                    jockey = jockey_cell.get_text(strip=True)
            
            # 厩舎（セル7）
            trainer = "不明"
            if len(cells) > 7:
                trainer_cell = cells[7]
                trainer_text = trainer_cell.get_text(strip=True)
                trainer = re.sub(r'^(栗東|美浦)', '', trainer_text)
            
            # 馬体重（セル8）
            horse_weight = "不明"
            if len(cells) > 8:
                weight_text = cells[8].get_text(strip=True)
                if re.match(r'\d{3,4}\([+-]?\d+\)', weight_text):
                    horse_weight = weight_text
            
            return {
                'race_id': race_id,
                '枠': waku,
                '馬番': umaban,
                '馬名': horse_name,
                '性齢': sei_rei,
                '騎手': jockey,
                '厩舎': trainer,
                '斤量': kinryo,
                '馬体重': horse_weight,
            }
            
        except Exception:
            return None
    
    def _get_confirmed_odds(self, race_id: str) -> Dict[int, Dict]:
        """確定オッズページから実際のオッズと人気を取得"""
        print("💰 オッズページから確定データ取得中...")
        
        # 複数のオッズページを試行
        odds_urls = [
            f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            f"https://race.netkeiba.com/odds/win.html?race_id={race_id}",
            f"https://race.netkeiba.com/race/odds.html?race_id={race_id}",
        ]
        
        for url in odds_urls:
            try:
                print(f"📊 {url} を確認中...")
                time.sleep(random.uniform(1.0, 2.0))
                
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                odds_data = self._parse_odds_page(soup)
                
                if odds_data:
                    print(f"✓ オッズデータ取得成功: {len(odds_data)}頭")
                    return odds_data
                    
            except Exception as e:
                print(f"⚠️ {url} エラー: {e}")
                continue
        
        print("❌ 全オッズページで取得失敗")
        return {}
    
    def _parse_odds_page(self, soup: BeautifulSoup) -> Dict[int, Dict]:
        """オッズページを解析してデータを抽出"""
        odds_data = {}
        
        # パターン1: 標準的なオッズテーブル
        tables = soup.find_all('table')
        
        for table in tables:
            # ヘッダー行をチェック
            header_row = table.find('tr')
            if header_row:
                header_cells = header_row.find_all(['th', 'td'])
                header_texts = [cell.get_text(strip=True) for cell in header_cells]
                
                # オッズテーブルかどうか判定
                if ('人気' in header_texts and ('オッズ' in ' '.join(header_texts) or '単勝オッズ' in header_texts)):
                    print(f"✓ オッズテーブル発見: {header_texts}")
                    
                    # 列位置を特定
                    popularity_col = -1
                    umaban_col = -1
                    odds_col = -1
                    
                    for i, text in enumerate(header_texts):
                        if '人気' in text:
                            popularity_col = i
                        elif '馬番' in text:
                            umaban_col = i
                        elif '単勝オッズ' in text or text == 'オッズ':
                            odds_col = i
                    
                    # データ行を解析
                    rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) > max(popularity_col, odds_col):
                            
                            # 人気
                            popularity = None
                            if popularity_col >= 0 and popularity_col < len(cells):
                                pop_text = cells[popularity_col].get_text(strip=True)
                                if pop_text.isdigit() and 1 <= int(pop_text) <= 18:
                                    popularity = int(pop_text)
                            
                            # 馬番
                            umaban = None
                            if umaban_col >= 0 and umaban_col < len(cells):
                                umaban_text = cells[umaban_col].get_text(strip=True)
                                if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                                    umaban = int(umaban_text)
                            
                            # オッズ
                            odds = None
                            if odds_col >= 0 and odds_col < len(cells):
                                odds_text = cells[odds_col].get_text(strip=True)
                                if odds_text and odds_text not in ['---.-', '**', '--', '']:
                                    try:
                                        if re.match(r'^\d+\.\d+$', odds_text):
                                            odds = float(odds_text)
                                    except:
                                        pass
                            
                            # データが有効な場合は保存
                            if popularity is not None and umaban is not None:
                                odds_data[umaban] = {
                                    '人気': popularity,
                                    'オッズ': odds
                                }
                                print(f"  馬番{umaban}: {popularity}人気, オッズ{odds}")
        
        # パターン2: JavaScript内のデータを探す
        if not odds_data:
            print("🔍 JavaScript内のオッズデータを探索中...")
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string:
                    content = script.string
                    
                    # オッズ配列パターンを検索
                    odds_patterns = [
                        r'odds.*?=.*?\[([\d\.,\s]+)\]',
                        r'win_odds.*?=.*?\[([\d\.,\s]+)\]',
                        r'popularity.*?=.*?\[([\d,\s]+)\]',
                    ]
                    
                    for pattern in odds_patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            print(f"✓ JavaScript内にオッズデータ発見: {match.group(1)[:100]}")
                            # ここでデータを解析...
        
        return odds_data
    
    def _merge_data(self, basic_data: pd.DataFrame, odds_data: Dict[int, Dict]) -> pd.DataFrame:
        """基本データとオッズデータを統合"""
        if basic_data.empty:
            return pd.DataFrame()
        
        final_data = basic_data.copy()
        final_data['オッズ'] = None
        final_data['人気'] = None
        
        # オッズデータを統合
        for _, row in final_data.iterrows():
            umaban = row['馬番']
            if umaban in odds_data:
                idx = final_data[final_data['馬番'] == umaban].index[0]
                if odds_data[umaban]['オッズ'] is not None:
                    final_data.loc[idx, 'オッズ'] = odds_data[umaban]['オッズ']
                if odds_data[umaban]['人気'] is not None:
                    final_data.loc[idx, '人気'] = odds_data[umaban]['人気']
        
        # 統計
        odds_count = final_data['オッズ'].notna().sum()
        pop_count = final_data['人気'].notna().sum()
        print(f"📊 統合結果: オッズ{odds_count}頭、人気{pop_count}頭")
        
        return final_data


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='簡潔netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202505021211)')
    parser.add_argument('--output', type=str, default='simplified_race_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = SimplifiedOddsScraper()
    race_data = scraper.scrape_race_with_odds(args.race_id)
    
    if race_data.empty:
        print("❌ データ取得に失敗しました")
        return
    
    # CSV保存
    race_data.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n💾 データ保存: {args.output}")
    
    # 結果表示
    print(f"\n📊 取得結果: {len(race_data)}頭")
    print("\n🏇 出馬表:")
    for _, horse in race_data.iterrows():
        odds_str = f"{horse['オッズ']}倍" if pd.notna(horse['オッズ']) else "未設定"
        pop_str = f"{horse['人気']}人気" if pd.notna(horse['人気']) else "未設定"
        print(f"  {horse['枠']}枠{horse['馬番']:2d}番 {horse['馬名']:15s} "
              f"{horse['騎手']:8s} {horse['厩舎']:8s} {horse['馬体重']:10s} "
              f"{odds_str:8s} {pop_str}")


if __name__ == "__main__":
    main()