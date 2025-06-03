#!/usr/bin/env python3
"""
netkeiba.com 改良版レースデータスクレイパー
オッズと人気を確実に取得する特化型スクレイパー
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from typing import Dict, List, Optional
import json

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


class ImprovedRaceScraper:
    """netkeiba.com からオッズと人気を確実に取得する改良版スクレイパー"""
    
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
        """セッションをセットアップ"""
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def scrape_race_data(self, race_id: str, retry_count: int = 3) -> pd.DataFrame:
        """レースデータを取得（複数手法でオッズ・人気を確実に取得）"""
        print(f"🏇 レース情報取得中: {race_id}")
        
        # 手法1: オッズAPI経由での取得を試す
        odds_data = self._try_odds_api(race_id)
        
        # 手法2: 動的スクレイピング（Selenium）
        dynamic_data = None
        if SELENIUM_AVAILABLE:
            dynamic_data = self._scrape_with_selenium(race_id)
        
        # 手法3: 静的スクレイピング（改良版）
        static_data = self._scrape_static_improved(race_id)
        
        # データを統合して最良の結果を選択
        final_data = self._merge_data_sources(odds_data, dynamic_data, static_data, race_id)
        
        if final_data.empty:
            print("❌ 全ての手法でデータ取得に失敗")
            return pd.DataFrame()
        
        print(f"✅ {len(final_data)}頭のデータを取得")
        return final_data
    
    def _try_odds_api(self, race_id: str) -> Optional[pd.DataFrame]:
        """オッズAPIまたはAjax経由でのデータ取得を試行"""
        try:
            # netkeiba.comのオッズ取得用のエンドポイントを試す
            ajax_urls = [
                f"https://race.netkeiba.com/api/api_get_jockey_result.html?race_id={race_id}",
                f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
                f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}",
            ]
            
            for url in ajax_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        # JSONレスポンスの場合
                        if 'application/json' in response.headers.get('content-type', ''):
                            data = response.json()
                            parsed_data = self._parse_json_response(data, race_id)
                            if parsed_data is not None:
                                print("✓ API経由でオッズデータ取得成功")
                                return parsed_data
                        
                        # HTMLレスポンスの場合
                        soup = BeautifulSoup(response.content, 'html.parser')
                        parsed_data = self._parse_odds_page(soup, race_id)
                        if parsed_data is not None:
                            print("✓ オッズページからデータ取得成功")
                            return parsed_data
                            
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ API取得エラー: {e}")
            return None
    
    def _scrape_with_selenium(self, race_id: str) -> Optional[pd.DataFrame]:
        """Seleniumを使った動的スクレイピング（改良版）"""
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        # Chromeオプション設定（より安定した設定）
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-javascript')  # JavaScriptを無効化してロード高速化
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={random.choice(self.USER_AGENTS)}')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(15)
            driver.implicitly_wait(5)
            
            # 複数のURLパターンを試す
            urls_to_try = [
                f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
                f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
                f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}",  # 地方競馬用
            ]
            
            for url in urls_to_try:
                try:
                    driver.get(url)
                    
                    # テーブルの読み込み待機
                    WebDriverWait(driver, 10).until(
                        EC.any_of(
                            EC.presence_of_element_located((By.CLASS_NAME, "Shutuba_Table")),
                            EC.presence_of_element_located((By.CLASS_NAME, "race_table_01")),
                            EC.presence_of_element_located((By.ID, "race_table"))
                        )
                    )
                    
                    # JavaScript実行完了まで待機
                    time.sleep(2)
                    
                    # ページソース取得して解析
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    data = self._parse_shutuba_table_improved(soup, race_id)
                    
                    if data is not None and not data.empty:
                        print("✓ Selenium動的スクレイピング成功")
                        return data
                        
                except Exception as url_error:
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Seleniumスクレイピングエラー: {e}")
            return None
            
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _scrape_static_improved(self, race_id: str) -> Optional[pd.DataFrame]:
        """改良版静的スクレイピング"""
        urls_to_try = [
            f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
            f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}",  # 地方競馬用
        ]
        
        for url in urls_to_try:
            try:
                # ランダムな待機時間
                time.sleep(random.uniform(0.5, 2.0))
                
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                data = self._parse_shutuba_table_improved(soup, race_id)
                
                if data is not None and not data.empty:
                    print("✓ 静的スクレイピング成功")
                    return data
                    
            except Exception as e:
                continue
        
        print("⚠️ 静的スクレイピング失敗")
        return None
    
    def _parse_shutuba_table_improved(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """改良版Shutuba_Table解析"""
        
        # 複数のテーブルパターンを試す
        table_selectors = [
            'table.Shutuba_Table',
            'table.race_table_01',
            'table[summary*="出馬表"]',
            'table[id*="race"]',
            'div.race_table_wrapper table',
        ]
        
        table = None
        for selector in table_selectors:
            table = soup.select_one(selector)
            if table:
                print(f"✓ テーブル発見: {selector}")
                break
        
        if not table:
            print("❌ 出馬表テーブルが見つかりません")
            return None
        
        horses_data = []
        rows = table.find_all('tr')
        
        print(f"テーブル行数: {len(rows)}")
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if len(cells) < 8:  # 最低限必要な列数
                continue
            
            # 馬番チェック
            first_cell_text = cells[0].get_text(strip=True)
            if not first_cell_text.isdigit():
                continue
            
            horse_num = int(first_cell_text)
            if not (1 <= horse_num <= 18):
                continue
            
            # セル内容をデバッグ表示
            print(f"\n--- 馬{horse_num}番データ解析 ---")
            for i, cell in enumerate(cells[:12]):
                cell_text = cell.get_text(strip=True)
                cell_classes = cell.get('class', [])
                print(f"セル{i}: '{cell_text}' class={cell_classes}")
            
            # 馬データを抽出
            horse_data = self._extract_horse_data_improved(cells, race_id, horse_num)
            if horse_data:
                horses_data.append(horse_data)
        
        if not horses_data:
            print("❌ 馬データの抽出に失敗")
            return None
        
        return pd.DataFrame(horses_data)
    
    def _extract_horse_data_improved(self, cells: List, race_id: str, horse_num: int) -> Optional[Dict]:
        """改良版馬データ抽出"""
        try:
            # 基本情報の抽出
            data = {
                'race_id': race_id,
                '馬番': horse_num,
                '枠': self._extract_waku(cells),
                '馬名': self._extract_horse_name(cells),
                '性齢': self._extract_age_sex(cells),
                '騎手': self._extract_jockey(cells),
                '馬舎': self._extract_trainer(cells),
                '斤量': self._extract_weight_carried(cells),
                '馬体重': self._extract_horse_weight(cells),
                'オッズ': self._extract_odds_improved(cells),
                '人気': self._extract_popularity_improved(cells)
            }
            
            # データの妥当性チェック
            if data['馬名'] == "不明" or data['騎手'] == "不明":
                print(f"⚠️ 馬{horse_num}番: 基本情報不足")
                return None
            
            # オッズと人気の状況を報告
            odds_status = "取得" if data['オッズ'] is not None else "未設定"
            pop_status = "取得" if data['人気'] is not None else "未設定"
            
            print(f"✓ 馬{horse_num}番: {data['馬名']} / {data['騎手']} / オッズ{odds_status} / 人気{pop_status}")
            
            return data
            
        except Exception as e:
            print(f"❌ 馬{horse_num}番データ抽出エラー: {e}")
            return None
    
    def _extract_waku(self, cells: List) -> int:
        """枠番を抽出"""
        # 1列目または枠関連クラスから抽出
        for i, cell in enumerate(cells[:3]):
            if 'Waku' in str(cell.get('class', [])):
                text = cell.get_text(strip=True)
                if text.isdigit() and 1 <= int(text) <= 8:
                    return int(text)
        
        # フォールバック: 馬番から推測
        horse_num = int(cells[1].get_text(strip=True)) if len(cells) > 1 else 1
        return ((horse_num - 1) // 2) + 1
    
    def _extract_horse_name(self, cells: List) -> str:
        """馬名を抽出"""
        for cell in cells[2:6]:
            # HorseInfoクラスまたは馬へのリンク
            if 'HorseInfo' in str(cell.get('class', [])):
                return cell.get_text(strip=True)
            
            horse_link = cell.find('a', href=lambda href: href and 'horse' in href)
            if horse_link:
                return horse_link.get_text(strip=True)
        
        return "不明"
    
    def _extract_age_sex(self, cells: List) -> str:
        """性齢を抽出"""
        for cell in cells[3:7]:
            if 'Barei' in str(cell.get('class', [])):
                return cell.get_text(strip=True)
        
        return "不明"
    
    def _extract_jockey(self, cells: List) -> str:
        """騎手を抽出"""
        for cell in cells[4:8]:
            # Jockeyクラスまたは騎手へのリンク
            if 'Jockey' in str(cell.get('class', [])):
                jockey_link = cell.find('a')
                if jockey_link:
                    return jockey_link.get_text(strip=True)
                return cell.get_text(strip=True)
            
            jockey_link = cell.find('a', href=lambda href: href and 'jockey' in href)
            if jockey_link:
                return jockey_link.get_text(strip=True)
        
        return "不明"
    
    def _extract_trainer(self, cells: List) -> str:
        """調教師を抽出"""
        for cell in cells[5:9]:
            if 'Trainer' in str(cell.get('class', [])):
                trainer_link = cell.find('a')
                if trainer_link:
                    return trainer_link.get_text(strip=True)
                return cell.get_text(strip=True)
        
        return "不明"
    
    def _extract_weight_carried(self, cells: List) -> float:
        """斤量を抽出"""
        for cell in cells[4:8]:
            text = cell.get_text(strip=True)
            if re.match(r'^5[0-9]\.[05]$', text):  # 50.0-59.5の範囲
                return float(text)
        
        return 57.0  # デフォルト値
    
    def _extract_horse_weight(self, cells: List) -> str:
        """馬体重を抽出"""
        for cell in cells[6:10]:
            if 'Weight' in str(cell.get('class', [])):
                return cell.get_text(strip=True)
            
            text = cell.get_text(strip=True)
            if re.match(r'\d{3,4}\([+-]?\d+\)', text):  # 体重(増減)の形式
                return text
        
        return "不明"
    
    def _extract_odds_improved(self, cells: List) -> Optional[float]:
        """改良版オッズ抽出"""
        
        # パターン1: 動的に読み込まれたオッズspan
        for cell in cells:
            odds_spans = cell.find_all('span', id=lambda x: x and 'odds-' in x)
            for span in odds_spans:
                odds_text = span.get_text(strip=True)
                if odds_text and odds_text not in ['---.-', '**', '--']:
                    try:
                        return float(odds_text)
                    except:
                        continue
        
        # パターン2: オッズ専用クラス
        odds_classes = ['Odds', 'odds', 'Popular', 'Txt_R']
        for cell in cells:
            cell_classes = cell.get('class', [])
            if any(odds_class in str(cell_classes) for odds_class in odds_classes):
                odds_text = cell.get_text(strip=True)
                if (odds_text and odds_text not in ['---.-', '**', '--'] and
                    re.match(r'^\d+\.\d+$', odds_text)):
                    try:
                        odds_val = float(odds_text)
                        if 1.0 <= odds_val <= 999.0 and not (50.0 <= odds_val <= 60.0):
                            return odds_val
                    except:
                        continue
        
        # パターン3: 数値パターンマッチング
        for cell in cells[7:]:  # 後半のセルを重点的に
            text = cell.get_text(strip=True)
            # より厳密なオッズパターン
            odds_patterns = [
                r'^(\d{1,3}\.\d{1})$',        # 基本パターン: 12.3
                r'^(\d{1,2}\.\d{2})$',        # 詳細パターン: 12.34
                r'単勝.*?(\d+\.\d+)',         # 単勝オッズ
                r'オッズ.*?(\d+\.\d+)',       # オッズ表記
            ]
            
            for pattern in odds_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        odds_val = float(match.group(1))
                        # より厳密な範囲チェック
                        if (1.0 <= odds_val <= 999.0 and 
                            not (50.0 <= odds_val <= 60.0) and  # 斤量除外
                            not (2020 <= odds_val <= 2030)):    # 年号除外
                            return odds_val
                    except:
                        continue
        
        return None
    
    def _extract_popularity_improved(self, cells: List) -> Optional[int]:
        """改良版人気抽出"""
        
        # パターン1: Popular_Ninkiクラス
        for cell in cells:
            if 'Popular_Ninki' in str(cell.get('class', [])):
                pop_text = cell.get_text(strip=True)
                if pop_text and pop_text not in ['**', '--', '***']:
                    try:
                        pop_val = int(pop_text)
                        if 1 <= pop_val <= 18:
                            return pop_val
                    except:
                        continue
        
        # パターン2: 人気関連クラス
        popularity_classes = ['Popular', 'popular', 'ninki', 'Ninki', 'rank', 'Rank']
        for cell in cells:
            cell_classes = cell.get('class', [])
            if any(pop_class in str(cell_classes) for pop_class in popularity_classes):
                pop_text = cell.get_text(strip=True)
                if (pop_text and pop_text not in ['**', '--', '***'] and 
                    pop_text.isdigit()):
                    try:
                        pop_val = int(pop_text)
                        if 1 <= pop_val <= 18:
                            return pop_val
                    except:
                        continue
        
        # パターン3: 背景色による人気判定
        for cell in cells:
            cell_classes = cell.get('class', [])
            if 'BgYellow' in cell_classes or 'bg-yellow' in cell_classes:
                return 1  # 1人気
            elif 'BgBlue' in cell_classes or 'bg-blue' in cell_classes:
                return 2  # 2人気
            elif 'BgOrange' in cell_classes or 'bg-orange' in cell_classes:
                return 3  # 3人気
        
        # パターン4: テキストパターンマッチング
        for cell in cells[8:]:  # 後半のセルを重点的に
            text = cell.get_text(strip=True)
            
            popularity_patterns = [
                r'^(\d{1,2})$',              # 単純な数字
                r'(\d{1,2})人気',            # 「5人気」形式
                r'人気(\d{1,2})',            # 「人気5」形式
                r'(\d{1,2})番人気',          # 「5番人気」形式
            ]
            
            for pattern in popularity_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        pop_val = int(match.group(1))
                        if 1 <= pop_val <= 18:
                            return pop_val
                    except:
                        continue
        
        return None
    
    def _parse_json_response(self, data: Dict, race_id: str) -> Optional[pd.DataFrame]:
        """JSON APIレスポンスの解析"""
        # 実装は必要に応じて追加
        return None
    
    def _parse_odds_page(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """オッズページの解析"""
        # 実装は必要に応じて追加
        return None
    
    def _merge_data_sources(self, odds_data: Optional[pd.DataFrame], 
                           dynamic_data: Optional[pd.DataFrame], 
                           static_data: Optional[pd.DataFrame], 
                           race_id: str) -> pd.DataFrame:
        """複数のデータソースを統合"""
        
        # 最も完全なデータを選択
        candidates = [data for data in [dynamic_data, static_data, odds_data] if data is not None]
        
        if not candidates:
            return pd.DataFrame()
        
        # オッズと人気の取得数でスコアを計算
        best_data = None
        best_score = -1
        
        for data in candidates:
            score = 0
            if not data.empty:
                # オッズ取得数をカウント
                odds_count = data['オッズ'].notna().sum()
                popularity_count = data['人気'].notna().sum()
                
                score = odds_count * 2 + popularity_count  # オッズを重視
                
                print(f"データソース評価: オッズ{odds_count}頭, 人気{popularity_count}頭, スコア{score}")
                
                if score > best_score:
                    best_score = score
                    best_data = data
        
        return best_data if best_data is not None else pd.DataFrame()


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='改良版netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202406020311)')
    parser.add_argument('--output', type=str, default='improved_race_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = ImprovedRaceScraper()
    race_data = scraper.scrape_race_data(args.race_id)
    
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
              f"{horse['騎手']:10s} {odds_str:8s} {pop_str}")


if __name__ == "__main__":
    main()