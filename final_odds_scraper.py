#!/usr/bin/env python3
"""
netkeiba.com 最終版オッズスクレイパー
オッズテーブル構造に基づく確実な取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from typing import Dict, List, Optional


class FinalOddsScraper:
    """判明したオッズテーブル構造に基づく確実なスクレイパー"""
    
    PLACE_DICT = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.67",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    ]
    
    def __init__(self):
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self):
        """セッションを設定"""
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://race.netkeiba.com/',
        })
    
    def scrape_complete_race_data(self, race_id: str) -> pd.DataFrame:
        """完全なレースデータを取得（基本情報+オッズ+人気）"""
        print(f"🏇 レース情報取得中: {race_id}")
        
        # 1. 基本情報を出馬表から取得
        basic_data = self._get_basic_data_from_shutuba(race_id)
        
        # 2. オッズと人気をオッズページから取得
        odds_data = self._get_odds_from_odds_page(race_id)
        
        # 3. データを統合
        final_data = self._merge_basic_and_odds_data(basic_data, odds_data, race_id)
        
        if final_data.empty:
            print("❌ データ取得に失敗しました")
            return pd.DataFrame()
        
        print(f"✅ {len(final_data)}頭の完全データを取得")
        return final_data
    
    def _get_basic_data_from_shutuba(self, race_id: str) -> Optional[pd.DataFrame]:
        """出馬表から基本情報を取得"""
        print("📋 出馬表から基本情報取得中...")
        
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(0.5, 1.5))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Shutuba_Tableを探す
            table = soup.find('table', class_='Shutuba_Table')
            if not table:
                print("❌ Shutuba_Tableが見つかりません")
                return None
            
            horses_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    # 馬番チェック（セル1）
                    if len(cells) > 1:
                        umaban_text = cells[1].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            horse_data = self._extract_basic_horse_data(cells, race_id)
                            if horse_data:
                                horses_data.append(horse_data)
            
            if horses_data:
                print(f"✓ 基本情報取得成功: {len(horses_data)}頭")
                return pd.DataFrame(horses_data)
            else:
                print("❌ 基本情報の抽出に失敗")
                return None
                
        except Exception as e:
            print(f"❌ 出馬表取得エラー: {e}")
            return None
    
    def _get_odds_from_odds_page(self, race_id: str) -> Optional[pd.DataFrame]:
        """オッズページから確実にオッズと人気を取得"""
        print("💰 オッズページからオッズ・人気取得中...")
        
        url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(1.0, 2.0))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 発見されたテーブル構造に基づいて解析
            tables = soup.find_all('table')
            
            for table in tables:
                # テーブルのヘッダーをチェック
                header_row = table.find('tr')
                if header_row:
                    header_cells = header_row.find_all(['th', 'td'])
                    header_texts = [cell.get_text(strip=True) for cell in header_cells]
                    
                    # 単勝オッズテーブルかチェック
                    if ('単勝オッズ' in header_texts or 
                        ('人気' in header_texts and 'オッズ' in ' '.join(header_texts))):
                        
                        print(f"✓ オッズテーブル発見: {header_texts}")
                        odds_data = self._extract_odds_from_table(table, race_id)
                        if odds_data:
                            return pd.DataFrame(odds_data)
            
            print("❌ オッズテーブルが見つかりません")
            return None
            
        except Exception as e:
            print(f"❌ オッズページエラー: {e}")
            return None
    
    def _extract_basic_horse_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """基本的な馬データを抽出"""
        try:
            # 枠番（セル0）
            waku = 1
            waku_cell = cells[0]
            waku_text = waku_cell.get_text(strip=True)
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
                elif 'HorseInfo' in str(horse_cell.get('class', [])):
                    horse_name = horse_cell.get_text(strip=True)
            
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
                if 'Jockey' in str(jockey_cell.get('class', [])):
                    jockey_link = jockey_cell.find('a')
                    if jockey_link:
                        jockey = jockey_link.get_text(strip=True)
                    else:
                        jockey = jockey_cell.get_text(strip=True)
            
            # 厩舎（セル7）
            trainer = "不明"
            if len(cells) > 7:
                trainer_cell = cells[7]
                if 'Trainer' in str(trainer_cell.get('class', [])):
                    trainer_link = trainer_cell.find('a')
                    if trainer_link:
                        trainer = trainer_link.get_text(strip=True)
                    else:
                        trainer = trainer_cell.get_text(strip=True)
                    # 地域プレフィックスを除去
                    trainer = re.sub(r'^(栗東|美浦|笠松|金沢|園田|姫路|高知|佐賀|門別|盛岡|水沢|浦和|船橋|大井|川崎)', '', trainer)
            
            # 馬体重（セル8）
            horse_weight = "不明"
            if len(cells) > 8:
                weight_cell = cells[8]
                if 'Weight' in str(weight_cell.get('class', [])):
                    weight_text = weight_cell.get_text(strip=True)
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
            
        except Exception as e:
            print(f"❌ 基本データ抽出エラー: {e}")
            return None
    
    def _extract_odds_from_table(self, table, race_id: str) -> List[Dict]:
        """確実なオッズテーブル解析"""
        odds_data = []
        
        try:
            rows = table.find_all('tr')
            
            # ヘッダー行を解析して列位置を特定
            header_row = rows[0] if rows else None
            if not header_row:
                return odds_data
            
            header_cells = header_row.find_all(['th', 'td'])
            header_texts = [cell.get_text(strip=True) for cell in header_cells]
            
            # 列位置を特定
            popularity_col = -1
            umaban_col = -1
            horse_name_col = -1
            odds_col = -1
            
            for i, text in enumerate(header_texts):
                if '人気' in text:
                    popularity_col = i
                elif '馬番' in text:
                    umaban_col = i
                elif '馬名' in text:
                    horse_name_col = i
                elif '単勝オッズ' in text or 'オッズ' in text:
                    odds_col = i
            
            print(f"列位置特定: 人気={popularity_col}, 馬番={umaban_col}, 馬名={horse_name_col}, オッズ={odds_col}")
            
            # データ行を解析
            for row in rows[1:]:  # ヘッダーをスキップ
                cells = row.find_all(['td', 'th'])
                if len(cells) >= max(popularity_col, umaban_col, horse_name_col, odds_col) + 1:
                    
                    # 人気
                    popularity = None
                    if popularity_col >= 0:
                        pop_text = cells[popularity_col].get_text(strip=True)
                        if pop_text.isdigit() and 1 <= int(pop_text) <= 18:
                            popularity = int(pop_text)
                    
                    # 馬番
                    umaban = None
                    if umaban_col >= 0:
                        umaban_text = cells[umaban_col].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            umaban = int(umaban_text)
                    
                    # 馬名
                    horse_name = None
                    if horse_name_col >= 0:
                        horse_name = cells[horse_name_col].get_text(strip=True)
                    
                    # オッズ
                    odds = None
                    if odds_col >= 0:
                        odds_text = cells[odds_col].get_text(strip=True)
                        if odds_text and odds_text not in ['---.-', '**', '--', '']:
                            try:
                                if re.match(r'^\d+\.\d+$', odds_text):
                                    odds = float(odds_text)
                            except:
                                pass
                    
                    # データが有効な場合のみ追加
                    if popularity is not None:
                        data_item = {
                            'race_id': race_id,
                            '人気': popularity,
                            'オッズ': odds
                        }
                        
                        if umaban is not None:
                            data_item['馬番'] = umaban
                        if horse_name:
                            data_item['馬名'] = horse_name
                        
                        odds_data.append(data_item)
                        print(f"✓ オッズデータ: 人気{popularity}, 馬番{umaban}, オッズ{odds}")
            
            print(f"✓ オッズデータ抽出成功: {len(odds_data)}件")
            return odds_data
            
        except Exception as e:
            print(f"❌ オッズテーブル解析エラー: {e}")
            return odds_data
    
    def _merge_basic_and_odds_data(self, basic_data: Optional[pd.DataFrame], 
                                  odds_data: Optional[pd.DataFrame], 
                                  race_id: str) -> pd.DataFrame:
        """基本データとオッズデータを統合"""
        
        if basic_data is None or basic_data.empty:
            print("❌ 基本データがありません")
            return pd.DataFrame()
        
        # 基本データをベースとする
        final_data = basic_data.copy()
        
        # オッズ・人気のデフォルト値を設定
        final_data['オッズ'] = None
        final_data['人気'] = None
        
        # オッズデータがある場合は統合
        if odds_data is not None and not odds_data.empty:
            print("🔗 データ統合中...")
            
            for _, odds_row in odds_data.iterrows():
                # 馬番で一致する行を探す
                if '馬番' in odds_row and pd.notna(odds_row['馬番']):
                    mask = final_data['馬番'] == odds_row['馬番']
                    if mask.any():
                        if pd.notna(odds_row['オッズ']):
                            final_data.loc[mask, 'オッズ'] = odds_row['オッズ']
                        if pd.notna(odds_row['人気']):
                            final_data.loc[mask, '人気'] = odds_row['人気']
                
                # 馬名で一致する行を探す（フォールバック）
                elif '馬名' in odds_row and odds_row['馬名']:
                    mask = final_data['馬名'] == odds_row['馬名']
                    if mask.any():
                        if pd.notna(odds_row['オッズ']):
                            final_data.loc[mask, 'オッズ'] = odds_row['オッズ']
                        if pd.notna(odds_row['人気']):
                            final_data.loc[mask, '人気'] = odds_row['人気']
        
        # 統計情報を表示
        odds_count = final_data['オッズ'].notna().sum()
        pop_count = final_data['人気'].notna().sum()
        total_count = len(final_data)
        
        print(f"📊 統合結果: 全{total_count}頭中、オッズ{odds_count}頭、人気{pop_count}頭")
        
        return final_data


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='最終版netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202406020311)')
    parser.add_argument('--output', type=str, default='final_race_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = FinalOddsScraper()
    race_data = scraper.scrape_complete_race_data(args.race_id)
    
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