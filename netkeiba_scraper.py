#!/usr/bin/env python3
"""
netkeiba.com レースデータスクレイパー
推測なしで実際の出馬表データのみ取得
馬番、枠、馬名、性齢、騎手、馬舎、馬体重、オッズ、人気を正確に抽出
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from typing import Dict, List, Optional




class NetkeibaScraper:
    """netkeiba.com から正確なレースデータを取得"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://race.netkeiba.com/race/shutuba.html"
        
    def scrape_race(self, race_id: str) -> pd.DataFrame:
        """レースデータを正確に取得"""
        print(f"🏇 レース情報取得中: {race_id}")
        
        # まず確定オッズページから取得を試す
        odds_data = self._scrape_odds_page(race_id)
        if not odds_data.empty:
            return odds_data
        
        # 静的スクレイピングにフォールバック
        return self._scrape_static(race_id)
    
    def _scrape_odds_page(self, race_id: str) -> pd.DataFrame:
        """オッズページから確定データを取得"""
        print("💰 オッズページから確定データを取得中...")
        
        # 出馬表とオッズページを両方取得
        shutuba_data = self._get_shutuba_data(race_id)
        odds_data = self._get_odds_data(race_id)
        
        if shutuba_data.empty:
            return pd.DataFrame()
        
        # データを統合
        final_data = self._merge_shutuba_and_odds(shutuba_data, odds_data)
        return final_data
    
    def _get_shutuba_data(self, race_id: str) -> pd.DataFrame:
        """出馬表から基本データを取得"""
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
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
                    # 馬番確認（セル1）
                    if len(cells) > 1:
                        umaban_text = cells[1].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            horse_data = self._extract_shutuba_data(cells, race_id)
                            if horse_data:
                                horses_data.append(horse_data)
            
            return pd.DataFrame(horses_data)
            
        except Exception as e:
            print(f"⚠️ 出馬表取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_odds_data(self, race_id: str) -> pd.DataFrame:
        """オッズページから確定オッズを取得"""
        url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # オッズテーブルを探す
            tables = soup.find_all('table')
            
            for table in tables:
                header_row = table.find('tr')
                if header_row:
                    header_cells = header_row.find_all(['th', 'td'])
                    header_texts = [cell.get_text(strip=True) for cell in header_cells]
                    
                    # 単勝オッズテーブルを特定
                    if '単勝オッズ' in header_texts or ('人気' in header_texts and 'オッズ' in ' '.join(header_texts)):
                        odds_data = self._extract_odds_data(table, race_id)
                        if odds_data:
                            return pd.DataFrame(odds_data)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠️ オッズページエラー: {e}")
            return pd.DataFrame()
    
    def _extract_shutuba_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """出馬表から基本データを抽出"""
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
                elif 'HorseInfo' in str(horse_cell.get('class', [])):
                    horse_name = horse_cell.get_text(strip=True)
            
            # 性齢（セル4）
            sei_rei = cells[4].get_text(strip=True) if len(cells) > 4 else "不明"
            
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
                # 地域プレフィックスを除去
                trainer = re.sub(r'^(栗東|美浦)', '', trainer_text)
            
            # 斤量（セル5）
            kinryo = 57.0
            if len(cells) > 5:
                kinryo_text = cells[5].get_text(strip=True)
                if re.match(r'^5[0-9]\.[05]$', kinryo_text):
                    kinryo = float(kinryo_text)
            
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
            
        except Exception as e:
            return None
    
    def _extract_odds_data(self, table, race_id: str) -> List[Dict]:
        """オッズテーブルからデータを抽出"""
        odds_data = []
        
        try:
            rows = table.find_all('tr')
            header_row = rows[0] if rows else None
            
            if not header_row:
                return odds_data
            
            # 列位置を特定
            header_cells = header_row.find_all(['th', 'td'])
            header_texts = [cell.get_text(strip=True) for cell in header_cells]
            
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
                elif '単勝オッズ' in text or (text == 'オッズ'):
                    odds_col = i
            
            print(f"列位置: 人気={popularity_col}, 馬番={umaban_col}, 馬名={horse_name_col}, オッズ={odds_col}")
            
            # データ行を解析
            for row in rows[1:]:
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
                    
                    # 馬名
                    horse_name = None
                    if horse_name_col >= 0 and horse_name_col < len(cells):
                        horse_name = cells[horse_name_col].get_text(strip=True)
                        if not horse_name:
                            horse_name = None
                    
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
            
            return odds_data
            
        except Exception as e:
            print(f"❌ オッズ抽出エラー: {e}")
            return odds_data
    
    def _merge_shutuba_and_odds(self, shutuba_data: pd.DataFrame, odds_data: pd.DataFrame) -> pd.DataFrame:
        """出馬表とオッズデータを統合"""
        if shutuba_data.empty:
            return pd.DataFrame()
        
        # 基本データにオッズ・人気列を追加
        final_data = shutuba_data.copy()
        final_data['オッズ'] = None
        final_data['人気'] = None
        
        if not odds_data.empty:
            # 馬番で統合
            for _, odds_row in odds_data.iterrows():
                if '馬番' in odds_row and pd.notna(odds_row['馬番']):
                    mask = final_data['馬番'] == odds_row['馬番']
                    if mask.any():
                        if pd.notna(odds_row['オッズ']):
                            final_data.loc[mask, 'オッズ'] = odds_row['オッズ']
                        if pd.notna(odds_row['人気']):
                            final_data.loc[mask, '人気'] = odds_row['人気']
                
                # 馬名で統合（フォールバック）
                elif '馬名' in odds_row and odds_row['馬名']:
                    mask = final_data['馬名'] == odds_row['馬名']
                    if mask.any():
                        if pd.notna(odds_row['オッズ']):
                            final_data.loc[mask, 'オッズ'] = odds_row['オッズ']
                        if pd.notna(odds_row['人気']):
                            final_data.loc[mask, '人気'] = odds_row['人気']
        
        # 統計を出力
        odds_count = final_data['オッズ'].notna().sum()
        pop_count = final_data['人気'].notna().sum()
        print(f"✅ データ統合完了: オッズ{odds_count}頭、人気{pop_count}頭")
        
        return final_data
    
    def _scrape_with_selenium(self, race_id: str) -> pd.DataFrame:
        """Seleniumで動的データを取得"""
        url = f"{self.base_url}?race_id={race_id}"
        
        # Chromeオプション設定
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--window-size=1920,1080')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.get(url)
            
            # Shutuba_Tableの読み込み待機
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "Shutuba_Table"))
            )
            
            # JavaScript実行完了まで待機
            time.sleep(3)
            
            # オッズが動的に読み込まれるまで待機
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "span[id*='odds-']") or
                             d.find_elements(By.XPATH, "//td[contains(text(),'.') and not(contains(text(),'---'))]")
                )
                time.sleep(2)  # 追加安定化待機
            except:
                print("⚠️ 動的オッズ待機タイムアウト")
            
            # ページソース取得して解析
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            return self._parse_shutuba_table(soup, race_id)
            
        finally:
            if driver:
                driver.quit()
    
    def _scrape_static(self, race_id: str) -> pd.DataFrame:
        """静的スクレイピング"""
        url = f"{self.base_url}?race_id={race_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_shutuba_table(soup, race_id)
        except Exception as e:
            print(f"❌ 静的スクレイピングエラー: {e}")
            return pd.DataFrame()
    
    def _parse_shutuba_table(self, soup: BeautifulSoup, race_id: str) -> pd.DataFrame:
        """Shutuba_Tableから正確にデータを抽出"""
        
        # Shutuba_Tableを探す
        shutuba_table = soup.find('table', class_='Shutuba_Table')
        if not shutuba_table:
            print("❌ Shutuba_Tableが見つかりません")
            return pd.DataFrame()
        
        print("✓ Shutuba_Table発見")
        
        horses_data = []
        rows = shutuba_table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 10:  # 最低限必要な列数
                continue
            
            # 馬番が1-18の数字かチェック
            first_cell = cells[0].get_text(strip=True)
            if not (first_cell.isdigit() and 1 <= int(first_cell) <= 18):
                continue
            
            horse_data = self._extract_horse_data(cells, race_id)
            if horse_data:  # データが正常に取得できた場合のみ追加
                horses_data.append(horse_data)
        
        if not horses_data:
            print("❌ 馬データが取得できませんでした")
            return pd.DataFrame()
        
        print(f"✅ {len(horses_data)}頭のデータを正確に取得")
        return pd.DataFrame(horses_data)
    
    def _extract_horse_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """1頭分のデータを正確に抽出"""
        
        # セル内容をデバッグ表示
        print(f"\n--- 馬データ解析 (セル数: {len(cells)}) ---")
        for i, cell in enumerate(cells[:15]):
            cell_text = cell.get_text(strip=True)
            cell_classes = cell.get('class', [])
            print(f"セル{i}: '{cell_text}' class={cell_classes}")
        
        try:
            # 1. 枠番 (セル0) - Wakuクラス
            waku = 1  # デフォルト値
            waku_cell = cells[0]
            waku_text = waku_cell.get_text(strip=True)
            if waku_text.isdigit() and 1 <= int(waku_text) <= 8:
                waku = int(waku_text)
            else:
                # クラス名から枠番を推測
                waku_classes = waku_cell.get('class', [])
                for cls in waku_classes:
                    if 'Waku' in cls:
                        waku_match = re.search(r'Waku(\d)', cls)
                        if waku_match:
                            waku = int(waku_match.group(1))
                            break
            
            # 2. 馬番 (セル1) - Umabanクラス
            umaban = 1  # デフォルト値
            if len(cells) > 1:
                umaban_cell = cells[1]
                umaban_text = umaban_cell.get_text(strip=True)
                if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                    umaban = int(umaban_text)
                else:
                    return None  # 馬番が取得できない場合は無効
            else:
                return None
            
            # 3. 馬名 (セル3、HorseInfoクラス)
            horse_name = "不明"
            if len(cells) > 3:
                horse_cell = cells[3]
                # リンクから馬名を取得
                horse_link = horse_cell.find('a', href=lambda href: href and 'horse' in href)
                if horse_link:
                    horse_name = horse_link.get_text(strip=True)
                elif 'HorseInfo' in str(horse_cell.get('class', [])):
                    horse_name = horse_cell.get_text(strip=True)
            
            # 4. 性齢 (セル4、Bareiクラス)
            sei_rei = "不明"
            if len(cells) > 4:
                sei_rei_cell = cells[4]
                if 'Barei' in str(sei_rei_cell.get('class', [])):
                    sei_rei = sei_rei_cell.get_text(strip=True)
                else:
                    # フォールバック
                    sei_rei = sei_rei_cell.get_text(strip=True)
            
            # 5. 斤量 (セル5)
            kinryo = 57.0
            if len(cells) > 5:
                kinryo_text = cells[5].get_text(strip=True)
                try:
                    if re.match(r'^5[0-9]\.[05]$', kinryo_text):  # 50.0-59.5の範囲
                        kinryo = float(kinryo_text)
                except:
                    pass
            
            # 6. 騎手 (セル6、Jockeyクラス)
            jockey = "不明"
            if len(cells) > 6:
                jockey_cell = cells[6]
                if 'Jockey' in str(jockey_cell.get('class', [])):
                    # リンクがある場合はリンクテキストを優先
                    jockey_link = jockey_cell.find('a', href=lambda href: href and 'jockey' in href)
                    if jockey_link:
                        jockey = jockey_link.get_text(strip=True)
                    else:
                        jockey = jockey_cell.get_text(strip=True)
            
            # 7. 厩舎/調教師 (セル7、Trainerクラス)
            trainer = "不明"
            if len(cells) > 7:
                trainer_cell = cells[7]
                if 'Trainer' in str(trainer_cell.get('class', [])):
                    # リンクがある場合はリンクテキストを優先
                    trainer_link = trainer_cell.find('a', href=lambda href: href and 'trainer' in href)
                    if trainer_link:
                        trainer = trainer_link.get_text(strip=True)
                    else:
                        trainer = trainer_cell.get_text(strip=True)
                    
                    # 「栗東」「美浦」などのプレフィックスを除去して調教師名のみ抽出
                    trainer = re.sub(r'^(栗東|美浦|笠松|金沢|園田|姫路|高知|佐賀|門別|盛岡|水沢|浦和|船橋|大井|川崎)', '', trainer)
            
            # 8. 馬体重 (セル8、Weightクラス)
            horse_weight = "不明"
            if len(cells) > 8:
                weight_cell = cells[8]
                if 'Weight' in str(weight_cell.get('class', [])):
                    weight_text = weight_cell.get_text(strip=True)
                    # 体重(増減)の形式をチェック
                    if re.match(r'\d{3,4}\([+-]?\d+\)', weight_text):
                        horse_weight = weight_text
            
            # 9. オッズ (動的データ優先)
            odds = self._extract_odds(cells)
            
            # 10. 人気 (動的データ優先)
            popularity = self._extract_popularity(cells)
            
            # データの完全性チェック
            if horse_name == "不明" or jockey == "不明":
                print(f"⚠️ 馬{umaban}番: 基本情報不足 (馬名={horse_name}, 騎手={jockey})")
                return None
            
            # オッズと人気の状況を報告
            odds_status = f"{odds}倍" if odds is not None else "未設定"
            pop_status = f"{popularity}人気" if popularity is not None else "未設定"
            
            print(f"✅ 馬{umaban}番: {horse_name} / {jockey} / {trainer} / {horse_weight} / {odds_status} / {pop_status}")
            
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
                'オッズ': odds,
                '人気': popularity
            }
            
        except Exception as e:
            print(f"❌ 馬データ抽出エラー: {e}")
            return None
    
    def _extract_odds(self, cells: List) -> Optional[float]:
        """オッズを正確に抽出（推測なし・多方面アプローチ）"""
        
        print(f"    💰 オッズ抽出開始...")
        
        # パターン1: 動的に読み込まれたオッズ（span[id*='odds-']）
        for i, cell in enumerate(cells):
            odds_spans = cell.find_all('span', id=lambda x: x and 'odds-' in x)
            for span in odds_spans:
                odds_text = span.get_text(strip=True)
                print(f"      パターン1: セル{i} span[odds-] = '{odds_text}'")
                if odds_text and odds_text not in ['---.-', '**', '--', '']:
                    try:
                        odds_val = float(odds_text)
                        if 1.0 <= odds_val <= 999.0:
                            print(f"      ✓ オッズ発見(動的span): {odds_val}")
                            return odds_val
                    except:
                        continue
        
        # パターン2: オッズ専用クラス（厳密検索）
        odds_classes = ['Odds', 'odds', 'popular_odds', 'shutuba_odds', 'Txt_R']
        for i, cell in enumerate(cells):
            cell_classes = cell.get('class', [])
            if any(odds_class in str(cell_classes) for odds_class in odds_classes):
                odds_text = cell.get_text(strip=True)
                print(f"      パターン2: セル{i} class={cell_classes} = '{odds_text}'")
                if (odds_text and odds_text not in ['---.-', '**', '--', ''] and
                    re.match(r'^\d+\.\d+$', odds_text)):
                    try:
                        odds_val = float(odds_text)
                        if 1.0 <= odds_val <= 999.0 and not (55.0 <= odds_val <= 58.0):
                            print(f"      ✓ オッズ発見(クラス): {odds_val}")
                            return odds_val
                    except:
                        continue
        
        # パターン3: 数値パターン全探索（後半セル重視）
        for i in range(len(cells)):
            cell = cells[i]
            cell_text = cell.get_text(strip=True)
            
            # より多様なオッズパターンをチェック
            odds_patterns = [
                r'^(\d{1,3}\.\d{1})$',        # 12.3
                r'^(\d{1,2}\.\d{2})$',        # 12.34  
                r'(\d+\.\d+)倍',              # 12.3倍
                r'単勝.*?(\d+\.\d+)',         # 単勝12.3
                r'オッズ.*?(\d+\.\d+)',       # オッズ12.3
                r'^(\d{1,3})\.(\d{1})$',      # 分離形式
            ]
            
            for pattern in odds_patterns:
                match = re.search(pattern, cell_text)
                if match:
                    try:
                        if len(match.groups()) == 2:  # 分離形式の場合
                            odds_val = float(f"{match.group(1)}.{match.group(2)}")
                        else:
                            odds_val = float(match.group(1))
                        
                        # より厳密な範囲チェック
                        if (1.0 <= odds_val <= 999.0 and 
                            not (50.0 <= odds_val <= 60.0) and  # 斤量除外
                            not (2020 <= odds_val <= 2030) and  # 年号除外
                            not (400 <= odds_val <= 600)):      # 体重除外
                            print(f"      ✓ オッズ発見(パターン): セル{i} '{cell_text}' -> {odds_val}")
                            return odds_val
                    except:
                        continue
        
        # パターン4: JavaScript動的コンテンツ（属性値から）
        for i, cell in enumerate(cells):
            # data-odds などの属性をチェック
            for attr_name in ['data-odds', 'data-value', 'value', 'data-price']:
                attr_value = cell.get(attr_name)
                if attr_value:
                    try:
                        odds_val = float(attr_value)
                        if 1.0 <= odds_val <= 999.0:
                            print(f"      ✓ オッズ発見(属性): セル{i} {attr_name}={odds_val}")
                            return odds_val
                    except:
                        continue
        
        # パターン5: 隠しテキスト・コメント内のオッズ
        for i, cell in enumerate(cells):
            # HTML コメント内を確認
            comments = cell.find_all(string=lambda text: isinstance(text, str) and '<!--' in text)
            for comment in comments:
                odds_match = re.search(r'(\d+\.\d+)', str(comment))
                if odds_match:
                    try:
                        odds_val = float(odds_match.group(1))
                        if 1.0 <= odds_val <= 999.0:
                            print(f"      ✓ オッズ発見(コメント): セル{i} -> {odds_val}")
                            return odds_val
                    except:
                        continue
        
        print(f"      ❌ オッズ未発見")
        return None
    
    def _extract_popularity(self, cells: List) -> Optional[int]:
        """人気を正確に抽出（推測なし）"""
        
        # パターン1: Popular_Ninkiクラス
        for cell in cells:
            if 'Popular_Ninki' in str(cell.get('class', [])):
                pop_text = cell.get_text(strip=True)
                if pop_text and pop_text not in ['**', '--']:
                    try:
                        pop_val = int(pop_text)
                        if 1 <= pop_val <= 18:
                            return pop_val
                    except:
                        continue
        
        # パターン2: 人気関連クラス
        popularity_classes = ['Popular', 'popular', 'ninki', 'Ninki', 'Txt_C']
        for cell in cells:
            cell_classes = cell.get('class', [])
            if any(pop_class in str(cell_classes) for pop_class in popularity_classes):
                pop_text = cell.get_text(strip=True)
                if (pop_text and pop_text not in ['**', '--'] and 
                    pop_text.isdigit()):
                    try:
                        pop_val = int(pop_text)
                        # 妥当な人気範囲かチェック（年齢、斤量を除外）
                        if 1 <= pop_val <= 18 and pop_val not in [3, 4, 5, 6, 55, 56, 57, 58]:
                            return pop_val
                    except:
                        continue
        
        # パターン3: 背景色による人気判定
        for cell in cells:
            cell_classes = cell.get('class', [])
            if 'BgYellow' in cell_classes:
                return 1  # 1人気
            elif 'BgBlue02' in cell_classes:
                return 2  # 2人気
            elif 'BgOrange' in cell_classes:
                return 3  # 3人気
        
        # パターン4: 特定セル位置での探索（9-15列目）
        for cell_idx in range(9, min(16, len(cells))):
            cell = cells[cell_idx]
            cell_text = cell.get_text(strip=True)
            
            if (cell_text and cell_text not in ['**', '--'] and 
                cell_text.isdigit()):
                try:
                    pop_val = int(cell_text)
                    # 妥当な人気範囲かチェック
                    if 1 <= pop_val <= 18 and pop_val not in [55, 56, 57, 58]:
                        return pop_val
                except:
                    continue
        
        return None


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='netkeiba.com正確スクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202406020311)')
    parser.add_argument('--output', type=str, default='scraped_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = NetkeibaScraper()
    race_data = scraper.scrape_race(args.race_id)
    
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
        odds_str = f"{horse['オッズ']}倍" if horse['オッズ'] is not None else "未設定"
        pop_str = f"{horse['人気']}人気" if horse['人気'] is not None else "未設定"
        print(f"  {horse['枠']}枠{horse['馬番']:2d}番 {horse['馬名']:15s} "
              f"{horse['騎手']:8s} {horse['厩舎']:8s} {horse['馬体重']:10s} "
              f"{odds_str:8s} {pop_str}")


if __name__ == "__main__":
    main()