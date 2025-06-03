#!/usr/bin/env python3
"""
改良版netkeiba.comスクレイピングシステム
一般的なnetkeiba.comスクレイピング手法を参考にした効率的な実装
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from typing import Dict, List, Optional, Tuple
import urllib.parse


class ImprovedNetkeibaScrapor:
    """改良版netkeiba.comスクレイパー"""
    
    def __init__(self):
        self.session = requests.Session()
        self._setup_session()
        self.base_url = "https://race.netkeiba.com"
        
        # 競馬場コード
        self.place_codes = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
        }
        
    def _setup_session(self):
        """セッション設定を最適化"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        self.session.headers.update(headers)
        
    def scrape_race_complete(self, race_id: str) -> pd.DataFrame:
        """完全なレース情報をスクレイピング"""
        print(f"🏇 改良版スクレイピング開始: {race_id}")
        
        # レース情報を解析
        race_info = self._parse_race_id(race_id)
        print(f"📍 {race_info['place']} {race_info['meeting']}回{race_info['day']}日目 {race_info['race_num']}R")
        
        # 1. 出馬表ページから基本情報取得
        print("📋 出馬表ページから基本情報取得中...")
        shutuba_data = self._scrape_shutuba_page(race_id)
        
        # 2. オッズページから確定オッズ取得
        print("💰 オッズページから確定オッズ取得中...")
        odds_data = self._scrape_odds_page(race_id)
        
        # 3. 結果ページから実績データ取得（該当する場合）
        print("🏆 結果ページから実績データ取得中...")
        result_data = self._scrape_result_page(race_id)
        
        # 4. データ統合
        final_data = self._merge_all_data(shutuba_data, odds_data, result_data, race_id)
        
        print(f"✅ 改良版スクレイピング完了: {len(final_data)}頭")
        return final_data
    
    def _parse_race_id(self, race_id: str) -> Dict:
        """レースIDから情報を抽出"""
        year = race_id[:4]
        place_code = race_id[4:6]
        meeting = race_id[6:8]
        day = race_id[8:10]
        race_num = race_id[10:12]
        
        return {
            'year': year,
            'place_code': place_code,
            'place': self.place_codes.get(place_code, f"不明({place_code})"),
            'meeting': meeting,
            'day': day,
            'race_num': race_num
        }
    
    def _scrape_shutuba_page(self, race_id: str) -> pd.DataFrame:
        """出馬表ページをスクレイピング"""
        url = f"{self.base_url}/race/shutuba.html?race_id={race_id}"
        
        try:
            # リクエスト間隔を調整
            time.sleep(random.uniform(1.0, 2.0))
            
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Shutuba_Tableを検索
            shutuba_table = soup.find('table', class_='Shutuba_Table')
            if not shutuba_table:
                print("❌ Shutuba_Tableが見つかりません")
                return pd.DataFrame()
            
            print("✓ Shutuba_Table発見、データ抽出中...")
            
            horses_data = []
            rows = shutuba_table.find_all('tr')
            
            for row_idx, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                
                # ヘッダー行をスキップ
                if len(cells) < 8 or not cells[1].get_text(strip=True).isdigit():
                    continue
                
                horse_data = self._extract_shutuba_horse_data(cells, race_id, row_idx)
                if horse_data:
                    horses_data.append(horse_data)
            
            df = pd.DataFrame(horses_data)
            print(f"✓ 出馬表データ取得: {len(df)}頭")
            return df
            
        except Exception as e:
            print(f"❌ 出馬表スクレイピングエラー: {e}")
            return pd.DataFrame()
    
    def _extract_shutuba_horse_data(self, cells: List, race_id: str, row_idx: int) -> Optional[Dict]:
        """出馬表から馬データを抽出（改良版）"""
        try:
            # デバッグ用: セル内容を確認
            if row_idx <= 3:  # 最初の3頭のみデバッグ出力
                cell_contents = [f"'{cell.get_text(strip=True)}'" for cell in cells[:10]]
                print(f"  行{row_idx}: {cell_contents}")
            
            # 基本データ抽出
            data = {}
            
            # 枠番（セル0または1）
            waku = self._extract_waku(cells)
            data['枠'] = waku
            
            # 馬番（セル1または2）
            umaban = self._extract_umaban(cells)
            if not umaban:
                return None
            data['馬番'] = umaban
            
            # 馬名（通常セル3、リンクまたはテキスト）
            horse_name = self._extract_horse_name(cells)
            data['馬名'] = horse_name
            
            # 性齢（通常セル4）
            sei_rei = self._extract_sei_rei(cells)
            data['性齢'] = sei_rei
            
            # 斤量（通常セル5）
            kinryo = self._extract_kinryo(cells)
            data['斤量'] = kinryo
            
            # 騎手（通常セル6）
            jockey = self._extract_jockey(cells)
            data['騎手'] = jockey
            
            # 厩舎（通常セル7）
            trainer = self._extract_trainer(cells)
            data['厩舎'] = trainer
            
            # 馬体重（通常セル8）
            horse_weight = self._extract_horse_weight(cells)
            data['馬体重'] = horse_weight
            
            # レースID追加
            data['race_id'] = race_id
            
            return data
            
        except Exception as e:
            print(f"⚠️ 行{row_idx}のデータ抽出エラー: {e}")
            return None
    
    def _extract_waku(self, cells: List) -> int:
        """枠番を抽出"""
        # セル0から試行
        for i in [0, 1]:
            if i < len(cells):
                text = cells[i].get_text(strip=True)
                if text.isdigit() and 1 <= int(text) <= 8:
                    return int(text)
                
                # CSS classから推測
                classes = cells[i].get('class', [])
                for cls in classes:
                    if 'Waku' in cls:
                        match = re.search(r'(\d)', cls)
                        if match and 1 <= int(match.group(1)) <= 8:
                            return int(match.group(1))
        
        return 1  # デフォルト値
    
    def _extract_umaban(self, cells: List) -> Optional[int]:
        """馬番を抽出"""
        for i in [1, 2, 0]:  # 通常はセル1、場合によってはセル2またはセル0
            if i < len(cells):
                text = cells[i].get_text(strip=True)
                if text.isdigit() and 1 <= int(text) <= 18:
                    return int(text)
        return None
    
    def _extract_horse_name(self, cells: List) -> str:
        """馬名を抽出"""
        # セル3～5の範囲で馬名を探す
        for i in range(3, min(6, len(cells))):
            cell = cells[i]
            
            # リンクから馬名を取得（優先）
            horse_link = cell.find('a', href=lambda href: href and 'horse' in href)
            if horse_link:
                name = horse_link.get_text(strip=True)
                if name and len(name) > 1:
                    return name
            
            # クラス名で判定
            if 'HorseInfo' in str(cell.get('class', [])) or 'Horse_Name' in str(cell.get('class', [])):
                name = cell.get_text(strip=True)
                if name and len(name) > 1:
                    return name
            
            # spanタグ内の馬名
            span = cell.find('span')
            if span:
                name = span.get_text(strip=True)
                if name and len(name) > 1:
                    return name
        
        return "不明"
    
    def _extract_sei_rei(self, cells: List) -> str:
        """性齢を抽出"""
        for i in range(4, min(7, len(cells))):
            text = cells[i].get_text(strip=True)
            # 性齢のパターン: 牡3, 牝4, セ5 など
            if re.match(r'^[牡牝セ][0-9]$', text):
                return text
        return "不明"
    
    def _extract_kinryo(self, cells: List) -> float:
        """斤量を抽出"""
        for i in range(5, min(8, len(cells))):
            text = cells[i].get_text(strip=True)
            # 斤量のパターン: 57.0, 54.5 など
            if re.match(r'^5[0-9]\.[05]$', text):
                try:
                    return float(text)
                except:
                    pass
        return 57.0  # デフォルト値
    
    def _extract_jockey(self, cells: List) -> str:
        """騎手を抽出"""
        for i in range(6, min(9, len(cells))):
            cell = cells[i]
            
            # リンクから騎手名を取得
            jockey_link = cell.find('a', href=lambda href: href and 'jockey' in href)
            if jockey_link:
                name = jockey_link.get_text(strip=True)
                if name:
                    return name
            
            # クラス名で判定
            if 'Jockey' in str(cell.get('class', [])):
                name = cell.get_text(strip=True)
                if name and len(name) > 1:
                    return name
        
        return "不明"
    
    def _extract_trainer(self, cells: List) -> str:
        """厩舎を抽出"""
        for i in range(7, min(10, len(cells))):
            cell = cells[i]
            
            # リンクから厩舎名を取得
            trainer_link = cell.find('a', href=lambda href: href and 'trainer' in href)
            if trainer_link:
                name = trainer_link.get_text(strip=True)
                if name:
                    # 地域プレフィックスを除去
                    return re.sub(r'^(栗東|美浦|笠松|金沢|園田|姫路|高知|佐賀|門別|盛岡|水沢|浦和|船橋|大井|川崎)', '', name)
            
            # クラス名で判定
            if 'Trainer' in str(cell.get('class', [])):
                name = cell.get_text(strip=True)
                if name and len(name) > 1:
                    return re.sub(r'^(栗東|美浦)', '', name)
        
        return "不明"
    
    def _extract_horse_weight(self, cells: List) -> str:
        """馬体重を抽出"""
        for i in range(8, min(12, len(cells))):
            cell = cells[i]
            
            text = cell.get_text(strip=True)
            # 馬体重のパターン: 456(+2), 480(-4) など
            if re.match(r'\d{3,4}\([+-]?\d+\)', text):
                return text
            
            # クラス名で判定
            if 'Weight' in str(cell.get('class', [])):
                text = cell.get_text(strip=True)
                if re.match(r'\d{3,4}', text):
                    return text
        
        return "不明"
    
    def _scrape_odds_page(self, race_id: str) -> Dict[int, Dict]:
        """オッズページをスクレイピング"""
        odds_data = {}
        
        # 複数のオッズページを試行
        odds_urls = [
            f"{self.base_url}/odds/index.html?race_id={race_id}",
            f"{self.base_url}/odds/win.html?race_id={race_id}",
        ]
        
        for url in odds_urls:
            try:
                time.sleep(random.uniform(1.5, 2.5))
                
                response = self.session.get(url, timeout=20)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # オッズテーブルを探す
                odds_tables = soup.find_all('table')
                
                for table in odds_tables:
                    table_odds = self._extract_odds_from_table(table, race_id)
                    if table_odds:
                        print(f"✓ オッズテーブルから{len(table_odds)}頭のデータ取得")
                        odds_data.update(table_odds)
                        break
                
                if odds_data:
                    break
                    
            except Exception as e:
                print(f"⚠️ オッズページ {url} エラー: {e}")
        
        return odds_data
    
    def _extract_odds_from_table(self, table, race_id: str) -> Dict[int, Dict]:
        """テーブルからオッズデータを抽出"""
        odds_data = {}
        
        try:
            # ヘッダー行をチェック
            header_row = table.find('tr')
            if not header_row:
                return odds_data
            
            header_cells = header_row.find_all(['th', 'td'])
            header_texts = [cell.get_text(strip=True) for cell in header_cells]
            
            # オッズテーブルかどうか判定
            has_odds = any('オッズ' in text for text in header_texts)
            has_popularity = any('人気' in text for text in header_texts)
            
            if not (has_odds or has_popularity):
                return odds_data
            
            # 列位置を特定
            popularity_col = -1
            umaban_col = -1
            odds_col = -1
            
            for i, text in enumerate(header_texts):
                if '人気' in text:
                    popularity_col = i
                elif '馬番' in text:
                    umaban_col = i
                elif 'オッズ' in text:
                    odds_col = i
            
            # データ行を解析
            rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) <= max(popularity_col, umaban_col, odds_col):
                    continue
                
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
                
                # データ保存
                if popularity and umaban:
                    odds_data[umaban] = {
                        '人気': popularity,
                        'オッズ': odds
                    }
        
        except Exception:
            pass
        
        return odds_data
    
    def _scrape_result_page(self, race_id: str) -> Dict:
        """結果ページをスクレイピング（該当する場合）"""
        result_data = {}
        
        url = f"{self.base_url}/race/result.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(1.0, 2.0))
            
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return result_data
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 結果テーブルを探す
            result_table = soup.find('table', class_='race_table_01')
            if result_table:
                print("✓ 結果ページを発見（過去レース）")
                # 結果データの抽出ロジックを実装
                # ここでは簡略化
                result_data['has_result'] = True
            
        except Exception as e:
            print(f"⚠️ 結果ページエラー: {e}")
        
        return result_data
    
    def _merge_all_data(self, shutuba_data: pd.DataFrame, 
                       odds_data: Dict[int, Dict], 
                       result_data: Dict, 
                       race_id: str) -> pd.DataFrame:
        """全データを統合"""
        if shutuba_data.empty:
            return pd.DataFrame()
        
        final_data = shutuba_data.copy()
        
        # オッズと人気のデフォルト値
        final_data['オッズ'] = None
        final_data['人気'] = None
        
        # オッズデータを統合
        if odds_data:
            for _, row in final_data.iterrows():
                umaban = row['馬番']
                if umaban in odds_data:
                    idx = final_data[final_data['馬番'] == umaban].index[0]
                    
                    if odds_data[umaban]['オッズ'] is not None:
                        final_data.loc[idx, 'オッズ'] = odds_data[umaban]['オッズ']
                    if odds_data[umaban]['人気'] is not None:
                        final_data.loc[idx, '人気'] = odds_data[umaban]['人気']
        
        # 統計情報
        odds_count = final_data['オッズ'].notna().sum()
        pop_count = final_data['人気'].notna().sum()
        print(f"📊 データ統合結果: オッズ{odds_count}頭、人気{pop_count}頭")
        
        return final_data


def main():
    """改良版スクレイパー実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='改良版netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202505021211)')
    parser.add_argument('--output', type=str, default='improved_race_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = ImprovedNetkeibaScrapor()
    race_data = scraper.scrape_race_complete(args.race_id)
    
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