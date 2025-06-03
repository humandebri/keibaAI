#!/usr/bin/env python3
"""
レースID自動予測システム
使い方: python auto_race_predictor.py 202505021201
"""

import argparse
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
import json
from io import StringIO
import html

# Seleniumのインポート（オプション）
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class NetkeibaRaceScraper:
    """netkeiba.comからレース情報を自動取得"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://race.netkeiba.com/race/shutuba.html"
        
        # 競馬場コード対応表（検証版）
        self.course_codes = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
            '05': '東京', '06': '中山', '07': '中京', '08': '京都',
            '09': '阪神', '10': '小倉', '11': '未確定', '12': '東京',
            '13': '盛岡', '14': '大井', '15': '船橋', '16': '川崎', '17': '浦和'
        }
    
    def parse_race_id(self, race_id: str) -> dict:
        """レースIDを解析"""
        if len(race_id) != 12:
            raise ValueError(f"レースIDは12桁で入力してください: {race_id}")
        
        return {
            'year': race_id[:4],
            'month': race_id[4:6],
            'day': race_id[6:8],
            'course_code': race_id[8:10],
            'race_num': race_id[10:12],
            'course_name': self.course_codes.get(race_id[8:10], '不明')
        }
    
    def scrape_race_data(self, race_id: str) -> pd.DataFrame:
        """レースデータをスクレイピング"""
        print(f"🏇 レース情報取得中: {race_id}")
        
        race_info = self.parse_race_id(race_id)
        print(f"   {race_info['year']}年{race_info['month']}月{race_info['day']}日")
        print(f"   {race_info['course_name']} {race_info['race_num']}R")
        
        # まず静的スクレイピングを試す（安定性重視）
        static_result = self._scrape_static(race_id, race_info)
        
        # 動的スクレイピングを優先実行
        if SELENIUM_AVAILABLE:
            try:
                dynamic_result = self._scrape_with_selenium_fast(race_id, race_info)
                if not dynamic_result.empty:
                    print("✅ 動的データ取得成功")
                    return dynamic_result
            except KeyboardInterrupt:
                return static_result
            except Exception as e:
                pass
        
        # 静的データは補完用
        if not static_result.empty:
            return static_result
        
        # SeleniumなしまたはChromeDriver問題
        if not SELENIUM_AVAILABLE:
            print("💡 完全なオッズ・人気取得には以下が必要:")
            print("   1. pip install selenium")
            print("   2. ChromeDriverのインストール")
        
        return static_result
    
    def _scrape_with_selenium(self, race_id: str, race_info: dict) -> pd.DataFrame:
        """Seleniumを使った動的スクレイピング"""
        url = f"{self.base_url}?race_id={race_id}"
        
        # Chromeオプション設定
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # ヘッドレスモード
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # オッズが読み込まれるまで待機
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "Shutuba_Table"))
            )
            
            # 少し待機してJavaScriptの実行を待つ
            time.sleep(3)
            
            # ページのHTMLを取得
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # レース情報を抽出
            race_data = self._extract_race_info_dynamic(soup, race_id, race_info)
            
            return race_data
            
        finally:
            if driver:
                driver.quit()
    
    def _scrape_with_selenium_fast(self, race_id: str, race_info: dict) -> pd.DataFrame:
        """高速・安定版Seleniumスクレイピング"""
        url = f"{self.base_url}?race_id={race_id}"
        
        # Chromeオプション設定（軽量化）
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-images')  # 画像読み込み無効
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        
        driver = None
        try:
            # タイムアウト設定付きでドライバー起動
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)  # タイムアウト延長
            driver.implicitly_wait(10)
            
            driver.get(url)
            
            # オッズテーブルの読み込み待機
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "Shutuba_Table"))
                )
                
                # JavaScript実行完了まで待機
                time.sleep(3)
                
                # オッズが動的に読み込まれるまで追加待機
                try:
                    # 複数のオッズ要素パターンを待機
                    WebDriverWait(driver, 10).until(
                        lambda d: (
                            d.find_elements(By.CSS_SELECTOR, "span[id*='odds-']") or
                            d.find_elements(By.CSS_SELECTOR, ".Odds") or
                            d.find_elements(By.CSS_SELECTOR, ".odds") or
                            d.find_elements(By.XPATH, "//td[contains(text(),'.')]")
                        )
                    )
                    print("✓ 動的オッズ要素確認")
                    time.sleep(2)  # 追加安定化待機
                except:
                    print("⚠️ 動的オッズ待機タイムアウト")
                    
            except:
                pass
            
            # ページソース取得
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # レース情報を抽出
            race_data = self._extract_race_info_dynamic(soup, race_id, race_info)
            
            if race_data.empty:
                print("   動的データ取得失敗")
                return pd.DataFrame()
            
            return race_data
            
        except Exception as e:
            print(f"   動的スクレイピングエラー: {str(e)[:50]}...")
            return pd.DataFrame()
            
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as quit_error:
                    print(f"   ドライバー終了エラー: {quit_error}")
                    pass
    
    def _scrape_static(self, race_id: str, race_info: dict) -> pd.DataFrame:
        """静的スクレイピング（従来の方法）"""
        url = f"{self.base_url}?race_id={race_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # レース情報を取得
            race_data = self._extract_race_info(soup, race_id, race_info)
            
            if race_data.empty:
                print("❌ レースデータが見つかりません")
                return pd.DataFrame()
            
            print(f"✅ {len(race_data)}頭の出馬表を取得")
            return race_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ ネットワークエラー: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ スクレイピングエラー: {e}")
            return pd.DataFrame()
    
    def _extract_race_info(self, soup: BeautifulSoup, race_id: str, race_info: dict) -> pd.DataFrame:
        """レース情報を抽出（2024年構造対応）"""
        horses_data = []
        
        # レース条件を取得
        race_conditions = self._get_race_conditions(soup)
        
        # 優先：netkeiba構造特化抽出
        horses_data = self._extract_shutuba_table_directly(soup, race_id, race_info, race_conditions)
        
        # フォールバック：pandas read_htmlを試す
        if not horses_data:
            print("⚠️ 専用抽出失敗、pandas read_htmlを試行")
            try:
                tables = pd.read_html(StringIO(str(soup)), header=0)
                
                for table in tables:
                    if len(table.columns) >= 6 and len(table) >= 3:
                        print(f"📋 テーブル発見: {len(table)}行, {len(table.columns)}列")
                        print(f"   列名: {list(table.columns)[:5]}")
                        
                        horses_data = self._parse_dataframe_horses(table, race_id, race_info, race_conditions)
                        if horses_data:
                            print(f"✓ pandas read_htmlで{len(horses_data)}頭のデータ抽出成功")
                            break
            
            except Exception as e:
                print(f"⚠️ pandas read_html失敗: {e}")
        
        if not horses_data:
            print("⚠️ 出馬表が見つかりません")
            print("💡 サンプルデータで試してください: --sample")
            return pd.DataFrame()
        
        return pd.DataFrame(horses_data)
    
    def _extract_race_info_dynamic(self, soup: BeautifulSoup, race_id: str, race_info: dict) -> pd.DataFrame:
        """動的に読み込まれたレース情報を抽出"""
        horses_data = []
        
        # レース条件を取得
        race_conditions = self._get_race_conditions(soup)
        
        # 動的データ対応の抽出
        horses_data = self._extract_shutuba_table_dynamic(soup, race_id, race_info, race_conditions)
        
        if not horses_data:
            print("⚠️ 動的出馬表が見つかりません")
            return pd.DataFrame()
        
        return pd.DataFrame(horses_data)
    
    def _extract_shutuba_table_dynamic(self, soup: BeautifulSoup, race_id: str, race_info: dict, race_conditions: dict) -> list:
        """動的データを含むShutuba_Tableから抽出"""
        horses_data = []
        
        # Shutuba_Tableを探す
        shutuba_table = soup.find('table', class_='Shutuba_Table')
        if not shutuba_table:
            return horses_data
        
        print("✓ 動的Shutuba_Table発見")
        
        # 各行を解析
        rows = shutuba_table.find_all('tr')
        header_skipped = False
        
        for row in rows:
            # ヘッダー行をスキップ
            if not header_skipped:
                if row.find('th'):
                    continue
                header_skipped = True
            
            # 馬データ行を探す
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                first_cell = cells[0].get_text(strip=True)
                is_horse_row = first_cell.isdigit() and 1 <= int(first_cell) <= 18
                
                if is_horse_row:
                    try:
                        horse_data = self._parse_shutuba_row_dynamic(row, race_id, race_info, race_conditions)
                        if horse_data:
                            horses_data.append(horse_data)
                    except Exception as e:
                        print(f"⚠️ 動的馬情報解析エラー: {e}")
                        continue
        
        # 重複データを削除（同じ馬番）
        unique_horses = {}
        for horse in horses_data:
            horse_num = horse['馬番']
            if horse_num not in unique_horses:
                unique_horses[horse_num] = horse
        
        final_horses = list(unique_horses.values())
        print(f"✓ {len(final_horses)}頭の動的馬データを抽出")
        return final_horses
    
    def _parse_shutuba_row_dynamic(self, row, race_id: str, race_info: dict, race_conditions: dict) -> dict:
        """動的データを含むShutuba_Table行から抽出"""
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 6:
            return None
        
        # 馬番（1列目）
        horse_num_cell = cells[0]
        horse_num_text = horse_num_cell.get_text(strip=True)
        if not horse_num_text.isdigit():
            return None
        horse_num = int(horse_num_text)
        
        # 馬名（HorseInfoクラスまたはリンク）
        horse_name = "不明"
        for cell in cells[1:5]:
            horse_info_link = cell.find('a', href=lambda href: href and 'horse' in href)
            if horse_info_link:
                horse_name = horse_info_link.get_text(strip=True)
                break
        
        # 騎手名（角括弧で囲まれた形式: [騎手名]）
        jockey_name = "不明"
        for cell in cells[1:8]:
            cell_text = cell.get_text(strip=True)
            jockey_match = re.search(r'\[([^\]]+)\]', cell_text)
            if jockey_match:
                jockey_name = jockey_match.group(1)
                break
            
            jockey_link = cell.find('a', href=lambda href: href and 'jockey' in href)
            if jockey_link:
                jockey_name = jockey_link.get_text(strip=True)
                break
        
        # オッズ（動的に読み込まれたデータから）
        odds = 99.9
        
        # パターン1: odds-X_Y形式のspan（動的データ）
        for cell in cells:
            odds_span = cell.find('span', id=lambda x: x and 'odds-' in x)
            if odds_span:
                odds_text = odds_span.get_text(strip=True)
                try:
                    if odds_text and odds_text != '' and '.' in odds_text:
                        odds = float(odds_text)
                        break
                except:
                    continue
        
        # パターン2: セル内の数字を直接探す
        if odds == 99.9:
            for cell in cells[6:12]:
                cell_text = cell.get_text(strip=True)
                odds_match = re.search(r'(\d+\.\d+)', cell_text)
                if odds_match:
                    try:
                        extracted_odds = float(odds_match.group(1))
                        if 1.0 <= extracted_odds <= 999.0:
                            odds = extracted_odds
                            break
                    except:
                        continue
        
        # 人気（動的に読み込まれたデータから）
        popularity = 99
        
        # パターン1: Popular_Ninkiクラス（動的データ）
        for cell in cells:
            if 'Popular_Ninki' in str(cell.get('class', [])):
                pop_text = cell.get_text(strip=True)
                try:
                    if pop_text.isdigit() and 1 <= int(pop_text) <= 18:
                        popularity = int(pop_text)
                        break
                except:
                    continue
        
        # パターン2: 背景色クラス（動的に追加される）
        if popularity == 99:
            for cell in cells:
                cell_classes = cell.get('class', [])
                if 'BgYellow' in cell_classes:
                    popularity = 1
                    break
                elif 'BgBlue02' in cell_classes:
                    popularity = 2
                    break
                elif 'BgOrange' in cell_classes:
                    popularity = 3
                    break
        
        # パターン3: 数字を直接探す
        if popularity == 99:
            for cell in cells[8:14]:
                cell_text = cell.get_text(strip=True)
                try:
                    if cell_text.isdigit() and 1 <= int(cell_text) <= 18:
                        popularity = int(cell_text)
                        break
                except:
                    continue
        
        print(f"  動的馬{horse_num}番: {horse_name} / {jockey_name} / {odds}倍 / {popularity}人気")
        
        return {
            'race_id': race_id,
            '馬': horse_name,
            '馬番': horse_num,
            '騎手': jockey_name,
            'オッズ': odds,
            '人気': popularity,
            '年齢': 4,
            '性': '牡',
            '斤量': 57.0,
            '体重': '480(0)',
            '体重変化': 0,
            '距離': race_conditions['距離'],
            'クラス': race_conditions['クラス'],
            '芝・ダート': race_conditions['芝・ダート'],
            '馬場': race_conditions['馬場'],
            '天気': race_conditions['天気'],
            'レース名': race_conditions['レース名'],
            '開催': race_info['course_name'],
            '場名': race_info['course_name']
        }
    
    def _extract_shutuba_table_directly(self, soup: BeautifulSoup, race_id: str, race_info: dict, race_conditions: dict) -> list:
        """Shutuba_Tableから直接データを抽出"""
        horses_data = []
        
        # 複数のテーブルパターンで出馬表を探索
        tables_found = []
        
        # パターン1: Shutuba_Table
        shutuba_table = soup.find('table', class_='Shutuba_Table')
        if shutuba_table:
            tables_found.append(('Shutuba_Table', shutuba_table))
            print("✓ Shutuba_Table発見")
        
        # パターン2: 他のテーブルクラス
        other_patterns = [
            ('table', {'class': 'race_table_01'}),
            ('table', {'class': 'nk_tb_common'}),
            ('table', {'id': 'shutsuba_table'}),
        ]
        
        for tag, attrs in other_patterns:
            table = soup.find(tag, attrs)
            if table:
                tables_found.append((str(attrs), table))
        
        # 全テーブルから馬データを抽出
        for table_name, table in tables_found:
            table_horses = self._extract_horses_from_table(table, race_id, race_info, race_conditions)
            horses_data.extend(table_horses)
        
        if not horses_data:
            # フォールバック: 汎用テーブル検索
            print("⚠️ 専用テーブルで馬データが見つからず、汎用検索を開始")
            return self._fallback_table_extraction(soup, race_id, race_info, race_conditions)
        
        # 馬名ベースでユニークにする（同じ馬名は1頭とみなす）
        unique_horses = {}
        for horse in horses_data:
            horse_name = horse['馬']
            if horse_name not in unique_horses:
                unique_horses[horse_name] = horse
            else:
                # より完全な情報を持つデータを優先
                existing = unique_horses[horse_name]
                if self._calculate_horse_data_score(horse) > self._calculate_horse_data_score(existing):
                    unique_horses[horse_name] = horse
        
        final_horses = list(unique_horses.values())
        
        # 馬番で再順序付け（馬番が重複している場合があるので馬名も考慮）
        final_horses.sort(key=lambda x: (x['馬番'], x['馬']))
        
        # 馬番を1-18で連番に修正（重複を解消）
        for i, horse in enumerate(final_horses, 1):
            horse['馬番'] = i
        
        print(f"✓ {len(final_horses)}頭のデータを取得（馬番1-{len(final_horses)}で連番化）")
        
        return final_horses
    
    def _calculate_horse_data_score(self, horse_data: dict) -> int:
        """馬データの完全性をスコア化"""
        score = 0
        
        # 馬名の質
        if horse_data['馬'] != "不明":
            score += 10
            
        # 騎手名の質
        if horse_data['騎手'] != "不明":
            score += 10
            
        # オッズの質
        if horse_data['オッズ'] != 99.9:
            score += 20
            
        # 人気の質
        if horse_data['人気'] != 99:
            score += 20
            
        # 有名騎手ボーナス
        famous_jockeys = ['武豊', '川田', 'ルメール', 'デムーロ', '福永', '岩田', '坂井']
        if any(jockey in horse_data['騎手'] for jockey in famous_jockeys):
            score += 5
            
        return score
    
    def _extract_horses_from_table(self, table, race_id: str, race_info: dict, race_conditions: dict) -> list:
        """テーブルから馬データを抽出"""
        horses_data = []
        
        # 全ての行を解析
        rows = table.find_all('tr')
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:
                continue
            
            # 1列目の内容をチェック
            first_cell = cells[0].get_text(strip=True)
            
            # 馬番判定（1-18の数字があれば馬データ行とみなす）
            if first_cell.isdigit():
                horse_num = int(first_cell)
                if 1 <= horse_num <= 18:
                    try:
                        horse_data = self._parse_shutuba_row(row, race_id, race_info, race_conditions)
                        if horse_data:
                            horses_data.append(horse_data)
                    except Exception as e:
                        continue
        
        return horses_data
    
    def _parse_shutuba_row(self, row, race_id: str, race_info: dict, race_conditions: dict) -> dict:
        """Shutuba_Table行からデータを正確に抽出"""
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 6:  # 最低限必要な列数
            return None
        
        # デバッグ: セル内容を表示
        print(f"  デバッグ: セル数={len(cells)}")
        for i, cell in enumerate(cells[:15]):  # 最初の15セルを表示
            cell_text = cell.get_text(strip=True)
            cell_classes = cell.get('class', [])
            print(f"    セル{i}: '{cell_text}' class={cell_classes}")
        
        # 馬番（1列目）
        horse_num_cell = cells[0]
        horse_num_text = horse_num_cell.get_text(strip=True)
        if not horse_num_text.isdigit():
            return None
        horse_num = int(horse_num_text)
        
        # 馬名（HorseInfoクラスまたはリンク）
        horse_name = "不明"
        # HorseInfoクラスを含むセルを探す
        for cell in cells[1:5]:  # 2-5列目で探す
            horse_info_link = cell.find('a', href=lambda href: href and 'horse' in href)
            if horse_info_link:
                horse_name = horse_info_link.get_text(strip=True)
                break
            
            # HorseInfoクラスのセルを探す
            if 'HorseInfo' in str(cell.get('class', [])):
                horse_name = cell.get_text(strip=True)
                break
        
        # 騎手名（角括弧で囲まれた形式: [騎手名]）
        jockey_name = "不明"
        for cell in cells[1:8]:  # 2-8列目で探す
            cell_text = cell.get_text(strip=True)
            # [騎手名] の形式を探す
            jockey_match = re.search(r'\[([^\]]+)\]', cell_text)
            if jockey_match:
                jockey_name = jockey_match.group(1)
                break
            
            # 騎手リンクを探す
            jockey_link = cell.find('a', href=lambda href: href and 'jockey' in href)
            if jockey_link:
                jockey_name = jockey_link.get_text(strip=True)
                break
        
        # オッズ（実際のHTMLテーブル構造に基づく正確な抽出）
        odds = 99.9
        
        # netkeiba.comの実際の構造に基づき、適切なセル位置でオッズを探す
        # デバッグ出力から判明: セル5は斤量(57.0)、セル9は未確定オッズ('---.-')
        
        # パターン1: 特定のセル位置でオッズを探す（最も確実）
        odds_candidate_cells = [8, 9, 10, 11, 12, 13, 14]  # オッズが入る可能性の高いセル位置
        for cell_idx in odds_candidate_cells:
            if cell_idx < len(cells):
                cell = cells[cell_idx]
                cell_text = cell.get_text(strip=True)
                
                # '---.-'や'**'は未確定データなのでスキップ
                if cell_text in ['---.-', '**', '--', '']:
                    continue
                    
                # オッズらしき数値パターンをチェック
                if re.match(r'^\d+\.\d+$', cell_text):
                    try:
                        extracted_odds = float(cell_text)
                        # 57.0は斤量、4.0は年齢の可能性があるので除外
                        if 1.0 <= extracted_odds <= 999.0 and extracted_odds != 57.0 and extracted_odds != 4.0:
                            odds = extracted_odds
                            print(f"      ✓ オッズ発見: セル{cell_idx} = {odds}")
                            break
                    except:
                        continue
        
        # パターン2: odds-X_Y形式のspan（動的データ）
        if odds == 99.9:
            for cell in cells:
                odds_span = cell.find('span', id=lambda x: x and 'odds-' in x)
                if odds_span:
                    odds_text = odds_span.get_text(strip=True)
                    try:
                        if odds_text and '.' in odds_text and odds_text not in ['---.-', '**']:
                            extracted_odds = float(odds_text)
                            if 1.0 <= extracted_odds <= 999.0 and extracted_odds != 57.0:
                                odds = extracted_odds
                                break
                    except:
                        continue
        
        # パターン3: クラス名でオッズセルを探す
        if odds == 99.9:
            odds_classes = ['Odds', 'odds', 'popular_odds', 'shutuba_odds', 'Txt_R']
            for cell in cells:
                cell_classes = cell.get('class', [])
                if any(odds_class in str(cell_classes) for odds_class in odds_classes):
                    odds_text = cell.get_text(strip=True)
                    try:
                        if (odds_text and '.' in odds_text and 
                            odds_text not in ['---.-', '**'] and
                            re.match(r'^\d+\.\d+$', odds_text)):
                            extracted_odds = float(odds_text)
                            if 1.0 <= extracted_odds <= 999.0 and extracted_odds != 57.0:
                                odds = extracted_odds
                                print(f"      ✓ オッズ発見(クラス): {odds}")
                                break
                    except:
                        continue
        
        # 人気（実際のHTMLテーブル構造に基づく正確な抽出）
        popularity = 99
        
        # netkeiba.comの実際の構造に基づき、適切なセル位置で人気を探す
        # デバッグ出力から判明: セル10に '**' があったのは未確定人気
        
        # パターン1: 特定のセル位置で人気を探す（最も確実）
        popularity_candidate_cells = [9, 10, 11, 12, 13, 14, 15]  # 人気が入る可能性の高いセル位置
        for cell_idx in popularity_candidate_cells:
            if cell_idx < len(cells):
                cell = cells[cell_idx]
                cell_text = cell.get_text(strip=True)
                
                # '**'や'--'は未確定データなのでスキップ
                if cell_text in ['**', '--', '---.-', '']:
                    continue
                    
                # 人気らしき数値パターンをチェック
                if cell_text.isdigit():
                    try:
                        extracted_pop = int(cell_text)
                        # 57は斤量、4は年齢の可能性があるので除外
                        if 1 <= extracted_pop <= 18 and extracted_pop not in [57, 4]:
                            popularity = extracted_pop
                            print(f"      ✓ 人気発見: セル{cell_idx} = {popularity}")
                            break
                    except:
                        continue
        
        # パターン2: Popular_Ninkiクラス（動的データ）
        if popularity == 99:
            for cell in cells:
                if 'Popular_Ninki' in str(cell.get('class', [])):
                    pop_text = cell.get_text(strip=True)
                    try:
                        if pop_text.isdigit() and 1 <= int(pop_text) <= 18:
                            popularity = int(pop_text)
                            print(f"      ✓ 人気発見(Popular_Ninki): {popularity}")
                            break
                    except:
                        continue
        
        # パターン3: 人気関連のクラス名
        if popularity == 99:
            popularity_classes = ['Popular', 'popular', 'ninki', 'Ninki', 'rank', 'Rank', 'Txt_C']
            for cell in cells:
                cell_classes = cell.get('class', [])
                if any(pop_class in str(cell_classes) for pop_class in popularity_classes):
                    pop_text = cell.get_text(strip=True)
                    try:
                        if (pop_text.isdigit() and 1 <= int(pop_text) <= 18 and 
                            int(pop_text) not in [57, 4]):  # 斤量と年齢を除外
                            popularity = int(pop_text)
                            print(f"      ✓ 人気発見(クラス): {popularity}")
                            break
                    except:
                        continue
        
        # パターン4: 背景色クラス
        if popularity == 99:
            for cell in cells:
                cell_classes = cell.get('class', [])
                if 'BgYellow' in cell_classes:  # 1人気
                    popularity = 1
                    break
                elif 'BgBlue02' in cell_classes:  # 2人気
                    popularity = 2
                    break
                elif 'BgOrange' in cell_classes:  # 3人気
                    popularity = 3
                    break
        
        # **推定値は使用せず、スクレイピングで取得できたデータのみ使用**
        # オッズまたは人気が取得できなかった場合はその馬のデータを無効とする
        if odds == 99.9 and popularity == 99:
            print(f"  ❌ 馬{horse_num}番: {horse_name} / {jockey_name} / オッズ・人気取得失敗 - データ無効")
            return None  # データが不完全な場合はNoneを返す
        elif odds == 99.9:
            print(f"  ⚠️ 馬{horse_num}番: {horse_name} / {jockey_name} / オッズ取得失敗 / {popularity}人気（人気のみ取得）")
            return None  # オッズが取得できない場合は無効
        elif popularity == 99:
            print(f"  ⚠️ 馬{horse_num}番: {horse_name} / {jockey_name} / {odds}倍 / 人気取得失敗（オッズのみ取得）")
            return None  # 人気が取得できない場合は無効
        else:
            print(f"  ✅ 馬{horse_num}番: {horse_name} / {jockey_name} / {odds}倍 / {popularity}人気（実測完全データ）")
        
        return {
            'race_id': race_id,
            '馬': horse_name,
            '馬番': horse_num,
            '騎手': jockey_name,
            'オッズ': odds,
            '人気': popularity,
            '年齢': 4,
            '性': '牡',
            '斤量': 57.0,
            '体重': '480(0)',
            '体重変化': 0,
            '距離': race_conditions['距離'],
            'クラス': race_conditions['クラス'],
            '芝・ダート': race_conditions['芝・ダート'],
            '馬場': race_conditions['馬場'],
            '天気': race_conditions['天気'],
            'レース名': race_conditions['レース名'],
            '開催': race_info['course_name'],
            '場名': race_info['course_name']
        }
    
    def _fallback_table_extraction(self, soup: BeautifulSoup, race_id: str, race_info: dict, race_conditions: dict) -> list:
        """フォールバック：汎用テーブル抽出"""
        horses_data = []
        
        horse_table = self._find_horse_table(soup)
        if not horse_table:
            return horses_data
        
        rows = horse_table.find_all('tr')[1:]  # ヘッダー行をスキップ
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 6:
                continue
            
            try:
                horse_data = self._parse_horse_row(cells, race_id, race_info, race_conditions)
                if horse_data:
                    horses_data.append(horse_data)
            except Exception as e:
                print(f"⚠️ 馬情報解析エラー: {e}")
                continue
        
        return horses_data
    
    def _find_horse_table(self, soup: BeautifulSoup):
        """複数のパターンで出馬表テーブルを探す"""
        # 2025年netkeiba構造対応
        patterns = [
            ('table', {'class': 'Shutuba_Table'}),  # 最新構造
            ('table', {'class': 'race_table_01'}),
            ('table', {'class': 'nk_tb_common'}),
            ('table', {'summary': '出馬表'}),
            ('table', {'id': 'shutsuba_table'}),
            ('div', {'class': 'race_table_wrapper'}),
        ]
        
        for tag, attrs in patterns:
            table = soup.find(tag, attrs)
            if table:
                if tag == 'div':
                    table = table.find('table')
                if table:
                    print(f"✓ テーブル発見: {attrs}")
                    return table
        
        # 汎用的にテーブルを探す
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) >= 3:  # 最低限の行数
                cells = rows[1].find_all(['td', 'th']) if len(rows) > 1 else []
                if len(cells) >= 6:  # 最低限の列数
                    print(f"✓ 汎用テーブル発見: {len(rows)}行, {len(cells)}列")
                    return table
        
        return None
    
    def _parse_dataframe_horses(self, df, race_id: str, race_info: dict, race_conditions: dict) -> list:
        """DataFrameから馬データを解析"""
        horses_data = []
        
        try:
            for idx, row in df.iterrows():
                # 行をリストに変換
                cells = [str(val) for val in row.values]
                
                # 馬番を探す
                horse_num = None
                for cell in cells[:3]:
                    if cell.isdigit() and 1 <= int(cell) <= 18:
                        horse_num = int(cell)
                        break
                
                if not horse_num:
                    continue
                
                # 基本情報を抽出
                horse_name = self._extract_horse_name_from_cells(cells)
                jockey = self._extract_jockey_from_cells(cells)
                odds = self._extract_odds_from_cells(cells)
                popularity = self._extract_popularity_from_cells(cells)
                
                if horse_name and jockey:
                    horse_data = {
                        'race_id': race_id,
                        '馬': horse_name,
                        '馬番': horse_num,
                        '騎手': jockey,
                        'オッズ': odds,
                        '人気': popularity,
                        '年齢': 4,
                        '性': '牡',
                        '斤量': 57.0,
                        '体重': '480(0)',
                        '体重変化': 0,
                        '距離': race_conditions['距離'],
                        'クラス': race_conditions['クラス'],
                        '芝・ダート': race_conditions['芝・ダート'],
                        '馬場': race_conditions['馬場'],
                        '天気': race_conditions['天気'],
                        'レース名': race_conditions['レース名'],
                        '開催': race_info['course_name'],
                        '場名': race_info['course_name']
                    }
                    horses_data.append(horse_data)
        
        except Exception as e:
            print(f"⚠️ DataFrame解析エラー: {e}")
        
        return horses_data
    
    def _extract_horse_name_from_cells(self, cells: list) -> str:
        """セルから馬名を抽出"""
        # 列名に「馬名」があるか確認
        for i, cell in enumerate(cells):
            if i == 3 and cell and cell != 'nan' and len(str(cell)) > 1:  # 4列目（馬名列）
                return str(cell)
        
        # フォールバック：馬名らしきセルを探す
        for i, cell in enumerate(cells[1:6]):  # 2-6列目
            cell_str = str(cell)
            if (cell_str and cell_str != 'nan' and len(cell_str) > 1 and 
                not cell_str.isdigit() and '騎手' not in cell_str and
                not any(keyword in cell_str for keyword in ['牡', '牝', '騙', '歳'])):
                return cell_str
        return "不明"
    
    def _extract_jockey_from_cells(self, cells: list) -> str:
        """セルから騎手名を抽出"""
        # 騎手名は通常馬名の次の列
        if len(cells) > 4:
            jockey_cell = str(cells[4])
            if jockey_cell and jockey_cell != 'nan' and len(jockey_cell) > 1:
                return jockey_cell
        
        # 有名騎手キーワード検索
        jockey_keywords = ['武豊', '川田', '福永', 'ルメール', 'デムーロ', '岩田', '松山', '藤岡', '坂井', '池添']
        for cell in cells:
            cell_str = str(cell)
            if any(keyword in cell_str for keyword in jockey_keywords):
                return cell_str
        
        # フォールバック：騎手らしきセルを探す
        for i, cell in enumerate(cells[3:8]):  # 4-8列目
            cell_str = str(cell)
            if (cell_str and cell_str != 'nan' and len(cell_str) > 1 and 
                not cell_str.isdigit() and 
                not any(keyword in cell_str for keyword in ['牡', '牝', '騙', '歳', 'kg'])):
                return cell_str
        return "不明"
    
    def _extract_odds_from_cells(self, cells: list) -> float:
        """セルからオッズを抽出"""
        for cell in cells:
            try:
                if '.' in cell and float(cell) > 1.0 and float(cell) < 999.0:
                    return float(cell)
            except:
                continue
        return 99.9
    
    def _extract_popularity_from_cells(self, cells: list) -> int:
        """セルから人気を抽出"""
        for cell in cells:
            try:
                if cell.isdigit() and 1 <= int(cell) <= 18:
                    return int(cell)
            except:
                continue
        return 99
    
    def _get_race_conditions(self, soup: BeautifulSoup) -> dict:
        """レース条件を取得（改良版）"""
        conditions = {
            '距離': 1600,
            'クラス': 5,
            '芝・ダート': 0,  # 0=芝, 1=ダート
            '馬場': 0,       # 0=良
            '天気': 0,       # 0=晴
            'レース名': '不明',
            '開催': '不明',
            '場名': '不明'
        }
        
        try:
            # レース名（複数パターンで取得）
            race_title_patterns = [
                soup.find('h1', class_='raceTitle'),
                soup.find('h1'),
                soup.find('div', class_='raceName'),
                soup.find('span', class_='raceName')
            ]
            
            for title_elem in race_title_patterns:
                if title_elem:
                    race_name = title_elem.get_text(strip=True)
                    if race_name and race_name != '':
                        conditions['レース名'] = race_name
                        break
            
            # レース条件文字列を複数パターンで探す
            race_data_patterns = [
                soup.find('div', class_='race_data'),
                soup.find('div', class_='raceData'),
                soup.find('span', class_='race_condition'),
                soup.find('p', class_='raceCondition')
            ]
            
            race_text = ""
            for data_elem in race_data_patterns:
                if data_elem:
                    race_text = data_elem.get_text()
                    break
            
            # 全体のHTMLからも距離情報を探す
            if not race_text:
                race_text = str(soup)
            
            # 距離を抽出（複数パターン）
            distance_patterns = [
                r'(\d{4})m',
                r'(\d{4})メートル', 
                r'(\d{4})M',
                r'距離.*?(\d{4})',
                r'(\d{4}).*?m'
            ]
            
            for pattern in distance_patterns:
                distance_match = re.search(pattern, race_text)
                if distance_match:
                    distance = int(distance_match.group(1))
                    if 1000 <= distance <= 4000:  # 妥当な距離範囲
                        conditions['距離'] = distance
                        break
            
            # 芝・ダートを判定
            if 'ダート' in race_text or 'ダ' in race_text or 'D' in race_text:
                conditions['芝・ダート'] = 1
            
            # クラスを推定
            if 'G1' in race_text or 'GⅠ' in race_text or 'GI' in race_text:
                conditions['クラス'] = 8
            elif 'G2' in race_text or 'GⅡ' in race_text or 'GII' in race_text:
                conditions['クラス'] = 7
            elif 'G3' in race_text or 'GⅢ' in race_text or 'GIII' in race_text:
                conditions['クラス'] = 6
            elif 'オープン' in race_text or 'OP' in race_text or 'Open' in race_text:
                conditions['クラス'] = 6
            elif '3勝' in race_text:
                conditions['クラス'] = 5
            elif '2勝' in race_text:
                conditions['クラス'] = 4
            elif '1勝' in race_text:
                conditions['クラス'] = 3
            elif '新馬' in race_text:
                conditions['クラス'] = 1
            elif '未勝利' in race_text:
                conditions['クラス'] = 2
        
        except Exception as e:
            print(f"⚠️ レース条件解析エラー: {e}")
        
        return conditions
    
    def _parse_horse_row(self, cells, race_id: str, race_info: dict, race_conditions: dict) -> dict:
        """馬の行データを解析"""
        try:
            # セル数により構造を判定
            if len(cells) < 8:
                return None
            
            # 基本情報を抽出
            horse_num = self._extract_text(cells[0])  # 馬番
            if not horse_num.isdigit():
                return None
            
            horse_name = self._extract_text(cells[1])  # 馬名
            jockey = self._extract_text(cells[2])      # 騎手
            
            # オッズと人気（サイトにより位置が異なる）
            odds = self._extract_odds(cells[3:6])
            popularity = self._extract_popularity(cells[3:6])
            
            # 馬の詳細情報
            age_sex = self._extract_text(cells[4] if len(cells) > 4 else cells[-2])
            weight = self._extract_text(cells[5] if len(cells) > 5 else cells[-1])
            
            # 年齢と性別を分離
            age, sex = self._parse_age_sex(age_sex)
            
            # 斤量と体重を分離
            carry_weight, horse_weight, weight_change = self._parse_weights(weight)
            
            return {
                'race_id': race_id,
                '馬': horse_name,
                '馬番': int(horse_num),
                '騎手': jockey,
                'オッズ': odds,
                '人気': popularity,
                '年齢': age,
                '性': sex,
                '斤量': carry_weight,
                '体重': horse_weight,
                '体重変化': weight_change,
                '距離': race_conditions['距離'],
                'クラス': race_conditions['クラス'],
                '芝・ダート': race_conditions['芝・ダート'],
                '馬場': race_conditions['馬場'],
                '天気': race_conditions['天気'],
                'レース名': race_conditions['レース名'],
                '開催': race_info['course_name'],
                '場名': race_info['course_name']
            }
            
        except Exception as e:
            print(f"⚠️ 馬データ解析エラー: {e}")
            return None
    
    def _extract_text(self, cell) -> str:
        """セルからテキストを抽出"""
        if cell is None:
            return ""
        return cell.get_text(strip=True)
    
    def _extract_odds(self, cells) -> float:
        """オッズを抽出"""
        for cell in cells:
            text = self._extract_text(cell)
            if re.match(r'^\d+\.\d+$', text):
                return float(text)
        return 99.9  # デフォルト値
    
    def _extract_popularity(self, cells) -> int:
        """人気を抽出"""
        for cell in cells:
            text = self._extract_text(cell)
            if re.match(r'^\d+$', text) and 1 <= int(text) <= 18:
                return int(text)
        return 99  # デフォルト値
    
    def _parse_age_sex(self, age_sex_text: str) -> tuple:
        """年齢・性別文字列を解析"""
        try:
            # "4牡" のような形式
            match = re.match(r'(\d+)([牡牝騙])', age_sex_text)
            if match:
                age = int(match.group(1))
                sex = match.group(2)
                return age, sex
        except:
            pass
        return 4, '牡'  # デフォルト値
    
    def _parse_weights(self, weight_text: str) -> tuple:
        """重量情報を解析"""
        try:
            # "57kg 480(+2)" のような形式
            parts = weight_text.split()
            
            # 斤量
            carry_weight = 55.0
            if parts and 'kg' in parts[0]:
                carry_weight = float(parts[0].replace('kg', ''))
            
            # 体重と変化
            horse_weight = "480(0)"
            weight_change = 0
            
            if len(parts) > 1:
                weight_part = parts[1]
                # "480(+2)" のような形式
                weight_match = re.match(r'(\d+)\(([+-]?\d+)\)', weight_part)
                if weight_match:
                    horse_weight = weight_part
                    weight_change = int(weight_match.group(2))
            
            return carry_weight, horse_weight, weight_change
            
        except:
            return 55.0, "480(0)", 0
    
    # 推定機能を削除 - 実際のスクレイピングデータのみ使用


def create_sample_race_data(race_id: str) -> pd.DataFrame:
    """テスト用サンプルレースデータを作成"""
    scraper = NetkeibaRaceScraper()
    race_info = scraper.parse_race_id(race_id)
    
    # サンプル馬データ
    sample_horses = [
        {'name': 'サンプル馬1', 'jockey': '武豊', 'odds': 2.3, 'popularity': 1},
        {'name': 'サンプル馬2', 'jockey': '川田将雅', 'odds': 3.8, 'popularity': 2},
        {'name': 'サンプル馬3', 'jockey': '福永祐一', 'odds': 5.2, 'popularity': 3},
        {'name': 'サンプル馬4', 'jockey': 'ルメール', 'odds': 6.8, 'popularity': 4},
        {'name': 'サンプル馬5', 'jockey': 'デムーロ', 'odds': 8.5, 'popularity': 5},
        {'name': 'サンプル馬6', 'jockey': '岩田康誠', 'odds': 12.3, 'popularity': 6},
        {'name': 'サンプル馬7', 'jockey': '松山弘平', 'odds': 15.8, 'popularity': 7},
        {'name': 'サンプル馬8', 'jockey': '藤岡佑介', 'odds': 18.9, 'popularity': 8}
    ]
    
    horses_data = []
    for i, horse in enumerate(sample_horses, 1):
        horses_data.append({
            'race_id': race_id,
            '馬': horse['name'],
            '馬番': i,
            '騎手': horse['jockey'],
            'オッズ': horse['odds'],
            '人気': horse['popularity'],
            '年齢': 4,
            '性': '牡',
            '斤量': 57.0,
            '体重': f"48{i}(+{i-4})",
            '体重変化': i-4,
            '距離': 1600,
            'クラス': 6,
            '芝・ダート': 0,
            '馬場': 0,
            '天気': 0,
            'レース名': f'{race_info["course_name"]}記念',
            '開催': race_info['course_name'],
            '場名': race_info['course_name']
        })
    
    return pd.DataFrame(horses_data)

def main():
    parser = argparse.ArgumentParser(description='レースID自動予測システム')
    parser.add_argument('race_id', type=str, 
                       help='レースID (例: 202505021201)')
    parser.add_argument('--output', type=str, default='races.csv',
                       help='出力CSVファイル名')
    parser.add_argument('--predict', action='store_true',
                       help='取得後すぐに予測実行')
    parser.add_argument('--strategy', type=str, 
                       choices=['conservative', 'standard', 'aggressive'],
                       default='standard',
                       help='予測戦略')
    parser.add_argument('--sample', action='store_true',
                       help='スクレイピングの代わりにサンプルデータを使用')
    
    args = parser.parse_args()
    
    print("🤖 競馬AI自動予測システム")
    print("=" * 50)
    
    # レースIDの形式チェック
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        print("   例: 202505021201")
        return
    
    # レースデータ取得
    if args.sample:
        print("📝 サンプルデータを生成中...")
        race_data = create_sample_race_data(args.race_id)
    else:
        # スクレイパー初期化
        scraper = NetkeibaRaceScraper()
        race_data = scraper.scrape_race_data(args.race_id)
        
        if race_data.empty:
            print("❌ スクレイピングに失敗しました。サンプルデータで試してください")
            print(f"   python auto_race_predictor.py {args.race_id} --sample --predict")
            return
    
    # CSVファイル保存
    race_data.to_csv(args.output, index=False, encoding='utf-8')
    print(f"💾 レースデータ保存: {args.output}")
    
    # データ内容を表示
    print(f"\n📊 取得データ:")
    print(f"   レース: {race_data.iloc[0]['レース名']}")
    print(f"   開催: {race_data.iloc[0]['開催']}")
    print(f"   距離: {race_data.iloc[0]['距離']}m")
    print(f"   出走頭数: {len(race_data)}頭")
    
    print("\n🏇 出馬表:")
    for _, horse in race_data.iterrows():
        print(f"   {horse['馬番']:2d}番 {horse['馬']:12s} "
              f"{horse['騎手']:8s} {horse['オッズ']:5.1f}倍 "
              f"{horse['人気']:2d}人気")
    
    # 予測実行（オプション）
    if args.predict:
        print(f"\n🔮 予測実行中...")
        
        try:
            import subprocess
            result = subprocess.run([
                'python', 'predict_races.py', args.output,
                '--strategy', args.strategy
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 予測完了")
                # 結果の一部を表示
                if result.stdout:
                    lines = result.stdout.split('\n')
                    in_results = False
                    for line in lines:
                        if '🏇 レース予測結果' in line:
                            in_results = True
                        if in_results:
                            print(line)
                        if '💡 推奨ベット' in line:
                            break
            else:
                print(f"❌ 予測エラー: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 予測実行エラー: {e}")
            print(f"手動実行: python predict_races.py {args.output}")

if __name__ == "__main__":
    main()