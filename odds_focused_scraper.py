#!/usr/bin/env python3
"""
netkeiba.com オッズ特化スクレイパー（Selenium不使用）
オッズと人気を確実に取得する専用スクレイパー
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
import json
from typing import Dict, List, Optional


class OddsFocusedScraper:
    """オッズ取得に特化したスクレイパー（Selenium不使用）"""
    
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
        """高度なセッション設定"""
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://race.netkeiba.com/',
        })
        
        # Cookieの初期設定
        self.session.get('https://race.netkeiba.com/', timeout=10)
    
    def scrape_race_with_odds(self, race_id: str) -> pd.DataFrame:
        """オッズを含む完全なレースデータを取得"""
        print(f"🏇 レース情報取得中: {race_id}")
        
        # 手法1: オッズ専用ページからの取得
        odds_data = self._get_odds_from_odds_page(race_id)
        
        # 手法2: 出馬表ページからの取得
        shutuba_data = self._get_data_from_shutuba_page(race_id)
        
        # 手法3: モバイル版からの取得（シンプルな構造）
        mobile_data = self._get_data_from_mobile_page(race_id)
        
        # 手法4: Ajax/API エンドポイントからの取得
        api_data = self._get_data_from_api(race_id)
        
        # データを統合
        final_data = self._merge_all_data(odds_data, shutuba_data, mobile_data, api_data, race_id)
        
        if final_data.empty:
            print("❌ 全ての手法でデータ取得に失敗")
            return pd.DataFrame()
        
        print(f"✅ {len(final_data)}頭のデータを取得")
        return final_data
    
    def _get_odds_from_odds_page(self, race_id: str) -> Optional[pd.DataFrame]:
        """オッズ専用ページからデータを取得"""
        print("📊 オッズページからの取得を試行...")
        
        odds_urls = [
            f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            f"https://race.netkeiba.com/odds/win.html?race_id={race_id}",
            f"https://race.netkeiba.com/api/api_get_jockey_result.html?race_id={race_id}",
        ]
        
        for url in odds_urls:
            try:
                time.sleep(random.uniform(1.0, 2.0))  # レート制限対策
                
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    print(f"✓ オッズページアクセス成功: {url}")
                    
                    # JSON レスポンスの場合
                    if 'application/json' in response.headers.get('content-type', ''):
                        try:
                            data = response.json()
                            parsed = self._parse_odds_json(data, race_id)
                            if parsed is not None:
                                return parsed
                        except:
                            pass
                    
                    # HTML レスポンスの場合
                    soup = BeautifulSoup(response.content, 'html.parser')
                    parsed = self._parse_odds_html(soup, race_id)
                    if parsed is not None:
                        return parsed
                        
            except Exception as e:
                print(f"⚠️ オッズページエラー: {e}")
                continue
        
        return None
    
    def _get_data_from_shutuba_page(self, race_id: str) -> Optional[pd.DataFrame]:
        """出馬表ページから基本データを取得"""
        print("📋 出馬表ページからの取得を試行...")
        
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(0.5, 1.5))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_shutuba_page(soup, race_id)
            
        except Exception as e:
            print(f"⚠️ 出馬表ページエラー: {e}")
            return None
    
    def _get_data_from_mobile_page(self, race_id: str) -> Optional[pd.DataFrame]:
        """モバイル版からシンプルなデータを取得"""
        print("📱 モバイル版からの取得を試行...")
        
        # モバイル用User-Agentに変更
        mobile_headers = self.session.headers.copy()
        mobile_headers['User-Agent'] = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15"
        
        mobile_urls = [
            f"https://sp.netkeiba.com/race/shutuba.html?race_id={race_id}",
            f"https://m.netkeiba.com/race/shutuba.html?race_id={race_id}",
        ]
        
        for url in mobile_urls:
            try:
                response = self.session.get(url, headers=mobile_headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    parsed = self._parse_mobile_page(soup, race_id)
                    if parsed is not None:
                        return parsed
            except:
                continue
        
        return None
    
    def _get_data_from_api(self, race_id: str) -> Optional[pd.DataFrame]:
        """Ajax/API エンドポイントからデータを取得"""
        print("🔌 API エンドポイントからの取得を試行...")
        
        api_endpoints = [
            f"https://race.netkeiba.com/api/api_get_odds.html?race_id={race_id}",
            f"https://race.netkeiba.com/api/race_before.html?race_id={race_id}",
            f"https://race.netkeiba.com/race/odds_ajax.html?race_id={race_id}",
        ]
        
        for endpoint in api_endpoints:
            try:
                # Ajax リクエスト用ヘッダー
                ajax_headers = self.session.headers.copy()
                ajax_headers.update({
                    'X-Requested-With': 'XMLHttpRequest',
                    'Referer': f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
                })
                
                response = self.session.get(endpoint, headers=ajax_headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"✓ API レスポンス取得: {endpoint}")
                    
                    # JSON の場合
                    try:
                        data = response.json()
                        parsed = self._parse_api_json(data, race_id)
                        if parsed is not None:
                            return parsed
                    except:
                        pass
                    
                    # HTML/text の場合
                    if response.text:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        parsed = self._parse_api_html(soup, race_id)
                        if parsed is not None:
                            return parsed
                            
            except Exception as e:
                print(f"⚠️ API エラー {endpoint}: {e}")
                continue
        
        return None
    
    def _parse_odds_json(self, data: Dict, race_id: str) -> Optional[pd.DataFrame]:
        """オッズJSONデータの解析"""
        try:
            if 'odds' in data or 'horse' in data:
                # データ構造に応じて解析
                horses_data = []
                
                # 一般的なJSONレスポンス構造を想定
                if isinstance(data, dict) and 'data' in data:
                    data = data['data']
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            horse_data = self._extract_from_json_item(item, race_id)
                            if horse_data:
                                horses_data.append(horse_data)
                
                if horses_data:
                    print(f"✓ JSON からデータ抽出成功: {len(horses_data)}頭")
                    return pd.DataFrame(horses_data)
        except:
            pass
        
        return None
    
    def _parse_odds_html(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """オッズページHTMLの解析"""
        try:
            # オッズテーブルを探す
            odds_tables = soup.find_all('table', class_=['odds_table', 'OddsTable', 'race_table'])
            
            for table in odds_tables:
                horses_data = []
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # ヘッダーをスキップ
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        horse_data = self._extract_from_odds_row(cells, race_id)
                        if horse_data:
                            horses_data.append(horse_data)
                
                if horses_data:
                    print(f"✓ オッズHTMLからデータ抽出成功: {len(horses_data)}頭")
                    return pd.DataFrame(horses_data)
        except:
            pass
        
        return None
    
    def _parse_shutuba_page(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """出馬表ページの解析（基本情報）"""
        try:
            table = soup.find('table', class_='Shutuba_Table')
            if not table:
                return None
            
            horses_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    # 馬番チェック
                    first_cell = cells[0].get_text(strip=True)
                    if first_cell.isdigit() and 1 <= int(first_cell) <= 8:
                        # 馬番は実際にはセル1にある
                        if len(cells) > 1:
                            umaban_text = cells[1].get_text(strip=True)
                            if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                                horse_data = self._extract_basic_horse_data(cells, race_id)
                                if horse_data:
                                    horses_data.append(horse_data)
            
            if horses_data:
                print(f"✓ 出馬表からデータ抽出成功: {len(horses_data)}頭")
                return pd.DataFrame(horses_data)
        except:
            pass
        
        return None
    
    def _parse_mobile_page(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """モバイルページの解析"""
        # モバイル版の簡単な構造を解析
        try:
            # モバイル版特有のクラスを探す
            mobile_selectors = [
                'div.horse_list',
                'table.horse_table',
                'div.race_horse',
                'ul.horse_list'
            ]
            
            for selector in mobile_selectors:
                elements = soup.select(selector)
                if elements:
                    # モバイル版からのデータ抽出ロジック
                    pass
        except:
            pass
        
        return None
    
    def _parse_api_json(self, data: Dict, race_id: str) -> Optional[pd.DataFrame]:
        """API JSONレスポンスの解析"""
        return self._parse_odds_json(data, race_id)
    
    def _parse_api_html(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """API HTMLレスポンスの解析"""
        return self._parse_odds_html(soup, race_id)
    
    def _extract_from_json_item(self, item: Dict, race_id: str) -> Optional[Dict]:
        """JSON アイテムからデータ抽出"""
        try:
            return {
                'race_id': race_id,
                '枠': item.get('waku', 1),
                '馬番': item.get('umaban', item.get('horse_number', 1)),
                '馬名': item.get('horse_name', item.get('name', '不明')),
                '性齢': item.get('age_sex', '不明'),
                '騎手': item.get('jockey', item.get('jockey_name', '不明')),
                '厩舎': item.get('trainer', '不明'),
                '斤量': float(item.get('weight', 57.0)),
                '馬体重': item.get('horse_weight', '不明'),
                'オッズ': float(item.get('odds', item.get('win_odds'))) if item.get('odds') else None,
                '人気': int(item.get('popularity', item.get('rank'))) if item.get('popularity') else None
            }
        except:
            return None
    
    def _extract_from_odds_row(self, cells: List, race_id: str) -> Optional[Dict]:
        """オッズテーブル行からデータ抽出"""
        try:
            umaban = int(cells[0].get_text(strip=True))
            horse_name = cells[1].get_text(strip=True) if len(cells) > 1 else "不明"
            odds_text = cells[2].get_text(strip=True) if len(cells) > 2 else None
            
            odds = None
            if odds_text and re.match(r'^\d+\.\d+$', odds_text):
                odds = float(odds_text)
            
            return {
                'race_id': race_id,
                '枠': ((umaban - 1) // 2) + 1,
                '馬番': umaban,
                '馬名': horse_name,
                '性齢': '不明',
                '騎手': '不明',
                '厩舎': '不明',
                '斤量': 57.0,
                '馬体重': '不明',
                'オッズ': odds,
                '人気': None
            }
        except:
            return None
    
    def _extract_basic_horse_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """基本的な馬データの抽出"""
        try:
            # 前回のロジックを再利用
            waku = 1
            waku_cell = cells[0]
            waku_text = waku_cell.get_text(strip=True)
            if waku_text.isdigit() and 1 <= int(waku_text) <= 8:
                waku = int(waku_text)
            
            umaban = int(cells[1].get_text(strip=True))
            horse_name = cells[3].get_text(strip=True) if len(cells) > 3 else "不明"
            sei_rei = cells[4].get_text(strip=True) if len(cells) > 4 else "不明"
            kinryo = float(cells[5].get_text(strip=True)) if len(cells) > 5 and re.match(r'^5[0-9]\.[05]$', cells[5].get_text(strip=True)) else 57.0
            jockey = cells[6].get_text(strip=True) if len(cells) > 6 else "不明"
            trainer = cells[7].get_text(strip=True) if len(cells) > 7 else "不明"
            
            # 厩舎名から地域プレフィックスを除去
            trainer = re.sub(r'^(栗東|美浦|笠松|金沢|園田|姫路|高知|佐賀|門別|盛岡|水沢|浦和|船橋|大井|川崎)', '', trainer)
            
            horse_weight = "不明"
            if len(cells) > 8:
                weight_text = cells[8].get_text(strip=True)
                if re.match(r'\d{3,4}\([+-]?\d+\)', weight_text):
                    horse_weight = weight_text
            
            # オッズと人気を積極的に探す
            odds = self._aggressive_odds_search(cells)
            popularity = self._aggressive_popularity_search(cells)
            
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
        except:
            return None
    
    def _aggressive_odds_search(self, cells: List) -> Optional[float]:
        """積極的なオッズ検索"""
        # 全セルを総当たりでチェック
        for cell in cells:
            text = cell.get_text(strip=True)
            
            # 数値パターンを広く探す
            patterns = [
                r'^(\d{1,3}\.\d{1,2})$',
                r'(\d+\.\d+)倍',
                r'単勝.*?(\d+\.\d+)',
                r'win.*?(\d+\.\d+)',
                r'オッズ.*?(\d+\.\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        odds_val = float(match.group(1))
                        if 1.0 <= odds_val <= 999.0 and not (50.0 <= odds_val <= 60.0):
                            return odds_val
                    except:
                        continue
        
        return None
    
    def _aggressive_popularity_search(self, cells: List) -> Optional[int]:
        """積極的な人気検索"""
        # 全セルを総当たりでチェック
        for cell in cells:
            text = cell.get_text(strip=True)
            
            # 人気パターンを探す
            patterns = [
                r'^(\d{1,2})$',
                r'(\d{1,2})人気',
                r'人気(\d{1,2})',
                r'rank.*?(\d{1,2})',
                r'pop.*?(\d{1,2})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        pop_val = int(match.group(1))
                        if 1 <= pop_val <= 18:
                            return pop_val
                    except:
                        continue
        
        return None
    
    def _merge_all_data(self, odds_data: Optional[pd.DataFrame], 
                       shutuba_data: Optional[pd.DataFrame], 
                       mobile_data: Optional[pd.DataFrame], 
                       api_data: Optional[pd.DataFrame], 
                       race_id: str) -> pd.DataFrame:
        """全データソースを統合"""
        
        # 最も完全なデータソースを選択
        data_sources = [
            ("API", api_data),
            ("オッズページ", odds_data), 
            ("出馬表", shutuba_data),
            ("モバイル", mobile_data)
        ]
        
        best_data = None
        best_score = -1
        
        for name, data in data_sources:
            if data is not None and not data.empty:
                # スコアを計算（オッズと人気の取得数で評価）
                odds_count = data['オッズ'].notna().sum() if 'オッズ' in data.columns else 0
                pop_count = data['人気'].notna().sum() if '人気' in data.columns else 0
                horse_count = len(data)
                
                score = odds_count * 3 + pop_count * 2 + horse_count
                print(f"データソース評価 {name}: 馬{horse_count}頭, オッズ{odds_count}頭, 人気{pop_count}頭, スコア{score}")
                
                if score > best_score:
                    best_score = score
                    best_data = data
        
        return best_data if best_data is not None else pd.DataFrame()


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='オッズ特化netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202406020311)')
    parser.add_argument('--output', type=str, default='odds_focused_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = OddsFocusedScraper()
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
        odds_str = f"{horse['オッズ']}倍" if horse['オッズ'] is not None else "未設定"
        pop_str = f"{horse['人気']}人気" if horse['人気'] is not None else "未設定"
        print(f"  {horse['枠']}枠{horse['馬番']:2d}番 {horse['馬名']:15s} "
              f"{horse['騎手']:8s} {horse['厩舎']:8s} {horse['馬体重']:10s} "
              f"{odds_str:8s} {pop_str}")


if __name__ == "__main__":
    main()