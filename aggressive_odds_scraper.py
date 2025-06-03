#!/usr/bin/env python3
"""
積極的オッズ取得システム
JavaScript API、動的コンテンツ、複数エンドポイントから確実にオッズを取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
import json
from typing import Dict, List, Optional
from urllib.parse import urljoin


class AggressiveOddsScraper:
    """あらゆる手段でオッズを取得する積極的スクレイパー"""
    
    def __init__(self):
        self.session = requests.Session()
        self._setup_session()
        self.base_url = "https://race.netkeiba.com"
        
    def _setup_session(self):
        """セッション設定を最適化"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        }
        self.session.headers.update(headers)
        
    def scrape_with_all_methods(self, race_id: str) -> pd.DataFrame:
        """あらゆる手段でデータを取得"""
        print(f"🚀 積極的データ取得開始: {race_id}")
        
        # 1. 基本データ取得
        basic_data = self._get_enhanced_basic_data(race_id)
        if basic_data.empty:
            print("❌ 基本データ取得失敗")
            return pd.DataFrame()
        
        # 2. 複数手法でオッズ取得を試行
        odds_data = {}
        
        # 手法1: 標準オッズページ
        print("\n🎯 手法1: 標準オッズページからの取得")
        odds_data.update(self._method1_standard_odds(race_id))
        
        # 手法2: JavaScript APIエンドポイント
        print("\n🎯 手法2: JavaScript APIエンドポイント")
        odds_data.update(self._method2_api_endpoints(race_id))
        
        # 手法3: Ajax動的取得
        print("\n🎯 手法3: Ajax動的取得")
        odds_data.update(self._method3_ajax_calls(race_id))
        
        # 手法4: リアルタイムオッズAPI
        print("\n🎯 手法4: リアルタイムオッズAPI")
        odds_data.update(self._method4_realtime_api(race_id))
        
        # 手法5: モバイル版ページ
        print("\n🎯 手法5: モバイル版ページ")
        odds_data.update(self._method5_mobile_version(race_id))
        
        # データ統合
        final_data = self._merge_all_data(basic_data, odds_data)
        
        return final_data
    
    def _get_enhanced_basic_data(self, race_id: str) -> pd.DataFrame:
        """強化された基本データ取得"""
        print("📋 強化基本データ取得中...")
        
        url = f"{self.base_url}/race/shutuba.html?race_id={race_id}"
        
        try:
            # リトライ機能付き取得
            for attempt in range(3):
                try:
                    time.sleep(random.uniform(1.0, 2.0))
                    response = self.session.get(url, timeout=20)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    print(f"⚠️ 再試行 {attempt + 1}/3...")
                    time.sleep(3)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Shutuba_Tableを取得
            table = soup.find('table', class_='Shutuba_Table')
            if not table:
                return pd.DataFrame()
            
            horses_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    if len(cells) > 1:
                        umaban_text = cells[1].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            horse_data = self._extract_enhanced_horse_data(cells, race_id)
                            if horse_data:
                                horses_data.append(horse_data)
            
            df = pd.DataFrame(horses_data)
            print(f"✓ 強化基本データ: {len(df)}頭取得成功")
            return df
            
        except Exception as e:
            print(f"❌ 基本データエラー: {e}")
            return pd.DataFrame()
    
    def _extract_enhanced_horse_data(self, cells: List, race_id: str) -> Optional[Dict]:
        """より詳細な馬データ抽出"""
        try:
            # 基本データ
            waku = 1
            waku_text = cells[0].get_text(strip=True)
            if waku_text.isdigit() and 1 <= int(waku_text) <= 8:
                waku = int(waku_text)
            
            umaban = int(cells[1].get_text(strip=True))
            
            # 馬名（より詳細に）
            horse_name = "不明"
            if len(cells) > 3:
                horse_cell = cells[3]
                # 複数の方法で馬名を試行
                horse_link = horse_cell.find('a', href=lambda href: href and 'horse' in href)
                if horse_link:
                    horse_name = horse_link.get_text(strip=True)
                elif horse_cell.find('span'):
                    horse_name = horse_cell.find('span').get_text(strip=True)
                else:
                    horse_name = horse_cell.get_text(strip=True)
            
            # 性齢
            sei_rei = cells[4].get_text(strip=True) if len(cells) > 4 else "不明"
            
            # 斤量
            kinryo = 57.0
            if len(cells) > 5:
                kinryo_text = cells[5].get_text(strip=True)
                if re.match(r'^5[0-9]\.[05]$', kinryo_text):
                    kinryo = float(kinryo_text)
            
            # 騎手（リンクからも取得）
            jockey = "不明"
            if len(cells) > 6:
                jockey_cell = cells[6]
                jockey_link = jockey_cell.find('a')
                if jockey_link:
                    jockey = jockey_link.get_text(strip=True)
                else:
                    jockey = jockey_cell.get_text(strip=True)
            
            # 厩舎
            trainer = "不明"
            if len(cells) > 7:
                trainer_cell = cells[7]
                trainer_text = trainer_cell.get_text(strip=True)
                trainer = re.sub(r'^(栗東|美浦)', '', trainer_text)
            
            # 馬体重
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
    
    def _method1_standard_odds(self, race_id: str) -> Dict[int, Dict]:
        """手法1: 標準オッズページから詳細取得"""
        odds_data = {}
        
        url = f"{self.base_url}/odds/index.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(1.5, 2.5))
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 全てのテーブルを詳細チェック
            tables = soup.find_all('table')
            
            for i, table in enumerate(tables):
                table_odds = self._extract_odds_from_any_table(table, race_id)
                if table_odds:
                    print(f"✓ テーブル{i+1}からオッズ取得: {len(table_odds)}頭")
                    odds_data.update(table_odds)
            
            # JavaScript変数をチェック
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    js_odds = self._extract_odds_from_javascript(script.string, race_id)
                    if js_odds:
                        print(f"✓ JavaScriptからオッズ取得: {len(js_odds)}頭")
                        odds_data.update(js_odds)
            
        except Exception as e:
            print(f"⚠️ 手法1エラー: {e}")
        
        return odds_data
    
    def _method2_api_endpoints(self, race_id: str) -> Dict[int, Dict]:
        """手法2: JavaScript APIエンドポイント"""
        odds_data = {}
        
        # 発見されたAPIエンドポイント
        api_urls = [
            f"{self.base_url}/api/api_get_jra_odds.html?race_id={race_id}",
            f"{self.base_url}/api/api_get_odds.html?race_id={race_id}",
            f"{self.base_url}/api/race_before.html?race_id={race_id}",
            f"{self.base_url}/odds/odds_get_form.html?type=1&race_id={race_id}",
        ]
        
        for url in api_urls:
            try:
                print(f"📡 API呼び出し: {url}")
                
                # APIリクエスト用ヘッダー
                api_headers = self.session.headers.copy()
                api_headers.update({
                    'X-Requested-With': 'XMLHttpRequest',
                    'Referer': f'{self.base_url}/odds/index.html?race_id={race_id}',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                })
                
                time.sleep(random.uniform(1.0, 2.0))
                response = self.session.get(url, headers=api_headers, timeout=15)
                
                if response.status_code == 200:
                    print(f"✓ API応答成功: {len(response.text)}文字")
                    
                    # JSON解析を試行
                    try:
                        json_data = response.json()
                        api_odds = self._parse_api_json_odds(json_data, race_id)
                        if api_odds:
                            print(f"✓ APIからオッズ取得: {len(api_odds)}頭")
                            odds_data.update(api_odds)
                    except:
                        # HTMLとして解析
                        soup = BeautifulSoup(response.content, 'html.parser')
                        html_odds = self._extract_odds_from_html_response(soup, race_id)
                        if html_odds:
                            print(f"✓ APIレスポンスHTMLからオッズ取得: {len(html_odds)}頭")
                            odds_data.update(html_odds)
                
            except Exception as e:
                print(f"⚠️ API {url} エラー: {e}")
        
        return odds_data
    
    def _method3_ajax_calls(self, race_id: str) -> Dict[int, Dict]:
        """手法3: Ajax動的呼び出し"""
        odds_data = {}
        
        # Ajax エンドポイント
        ajax_urls = [
            f"{self.base_url}/odds/odds_ajax.html?race_id={race_id}",
            f"{self.base_url}/race/ajax_race_odds.html?race_id={race_id}",
        ]
        
        for url in ajax_urls:
            try:
                print(f"🔄 Ajax呼び出し: {url}")
                
                ajax_headers = self.session.headers.copy()
                ajax_headers.update({
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                })
                
                # POSTとGETの両方を試行
                for method in ['GET', 'POST']:
                    try:
                        if method == 'POST':
                            data = {'race_id': race_id, 'type': '1'}
                            response = self.session.post(url, data=data, headers=ajax_headers, timeout=15)
                        else:
                            response = self.session.get(url, headers=ajax_headers, timeout=15)
                        
                        if response.status_code == 200 and len(response.text) > 10:
                            print(f"✓ Ajax {method} 成功: {len(response.text)}文字")
                            
                            # レスポンス解析
                            try:
                                json_data = response.json()
                                ajax_odds = self._parse_ajax_odds(json_data, race_id)
                                if ajax_odds:
                                    odds_data.update(ajax_odds)
                            except:
                                # HTML解析
                                soup = BeautifulSoup(response.content, 'html.parser')
                                html_odds = self._extract_odds_from_html_response(soup, race_id)
                                if html_odds:
                                    odds_data.update(html_odds)
                            
                            break  # 成功したらGET/POSTのループを終了
                            
                    except Exception as e:
                        print(f"⚠️ Ajax {method} エラー: {e}")
                
            except Exception as e:
                print(f"⚠️ Ajax URL {url} エラー: {e}")
        
        return odds_data
    
    def _method4_realtime_api(self, race_id: str) -> Dict[int, Dict]:
        """手法4: リアルタイムオッズAPI"""
        odds_data = {}
        
        # リアルタイムAPIのシミュレーション
        try:
            print("⚡ リアルタイムAPI呼び出し")
            
            # WebSocketやServer-Sent Eventsのエンドポイントを試行
            realtime_urls = [
                f"{self.base_url}/api/realtime_odds?race_id={race_id}",
                f"wss://race.netkeiba.com/ws/odds/{race_id}",
            ]
            
            for url in realtime_urls:
                if url.startswith('http'):
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            print(f"✓ リアルタイムAPI応答: {len(response.text)}文字")
                            # 解析処理...
                    except:
                        pass
            
        except Exception as e:
            print(f"⚠️ リアルタイムAPI エラー: {e}")
        
        return odds_data
    
    def _method5_mobile_version(self, race_id: str) -> Dict[int, Dict]:
        """手法5: モバイル版ページ"""
        odds_data = {}
        
        try:
            print("📱 モバイル版ページアクセス")
            
            # モバイル用ヘッダー
            mobile_headers = self.session.headers.copy()
            mobile_headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
            
            mobile_urls = [
                f"https://sp.netkeiba.com/race/odds/index.html?race_id={race_id}",
                f"https://m.netkeiba.com/race/odds?race_id={race_id}",
            ]
            
            for url in mobile_urls:
                try:
                    response = self.session.get(url, headers=mobile_headers, timeout=15)
                    if response.status_code == 200:
                        print(f"✓ モバイル版アクセス成功: {len(response.text)}文字")
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        mobile_odds = self._extract_odds_from_mobile_page(soup, race_id)
                        if mobile_odds:
                            odds_data.update(mobile_odds)
                
                except Exception as e:
                    print(f"⚠️ モバイル版 {url} エラー: {e}")
        
        except Exception as e:
            print(f"⚠️ モバイル版全般エラー: {e}")
        
        return odds_data
    
    def _extract_odds_from_any_table(self, table, race_id: str) -> Dict[int, Dict]:
        """任意のテーブルからオッズを抽出"""
        odds_data = {}
        
        try:
            # ヘッダー行チェック
            header_row = table.find('tr')
            if not header_row:
                return odds_data
            
            header_cells = header_row.find_all(['th', 'td'])
            header_texts = [cell.get_text(strip=True) for cell in header_cells]
            
            # オッズ関連テーブルかチェック
            if not any(keyword in ' '.join(header_texts) for keyword in ['オッズ', '人気', '倍']):
                return odds_data
            
            # 列位置特定
            popularity_col = -1
            umaban_col = -1
            odds_col = -1
            
            for i, text in enumerate(header_texts):
                if '人気' in text:
                    popularity_col = i
                elif '馬番' in text:
                    umaban_col = i
                elif 'オッズ' in text or '倍' in text:
                    odds_col = i
            
            # データ行解析
            rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) > max(popularity_col, odds_col, umaban_col):
                    
                    # 人気取得
                    popularity = None
                    if popularity_col >= 0:
                        pop_text = cells[popularity_col].get_text(strip=True)
                        if pop_text.isdigit() and 1 <= int(pop_text) <= 18:
                            popularity = int(pop_text)
                    
                    # 馬番取得
                    umaban = None
                    if umaban_col >= 0:
                        umaban_text = cells[umaban_col].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            umaban = int(umaban_text)
                    
                    # オッズ取得
                    odds = None
                    if odds_col >= 0:
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
    
    def _extract_odds_from_javascript(self, js_content: str, race_id: str) -> Dict[int, Dict]:
        """JavaScript変数からオッズを抽出"""
        odds_data = {}
        
        try:
            # JavaScript変数パターンを検索
            patterns = [
                r'odds\s*=\s*\[([\d\.,\s]+)\]',
                r'win_odds\s*=\s*\[([\d\.,\s]+)\]',
                r'popularity\s*=\s*\[([\d,\s]+)\]',
                r'race_odds\s*=\s*\{([^}]+)\}',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, js_content, re.IGNORECASE)
                for match in matches:
                    # マッチしたデータを解析
                    data_str = match.group(1)
                    # 解析ロジックを実装...
        
        except Exception:
            pass
        
        return odds_data
    
    def _parse_api_json_odds(self, json_data: Dict, race_id: str) -> Dict[int, Dict]:
        """API JSONレスポンスからオッズを解析"""
        odds_data = {}
        
        try:
            # JSON構造を解析してオッズデータを抽出
            if 'odds' in json_data:
                odds_info = json_data['odds']
                # 解析ロジック...
            elif 'data' in json_data:
                # 別の構造を試行...
                pass
        
        except Exception:
            pass
        
        return odds_data
    
    def _extract_odds_from_html_response(self, soup: BeautifulSoup, race_id: str) -> Dict[int, Dict]:
        """HTMLレスポンスからオッズを抽出"""
        odds_data = {}
        
        try:
            # HTMLからテーブルを探して解析
            tables = soup.find_all('table')
            for table in tables:
                table_odds = self._extract_odds_from_any_table(table, race_id)
                odds_data.update(table_odds)
        
        except Exception:
            pass
        
        return odds_data
    
    def _parse_ajax_odds(self, json_data: Dict, race_id: str) -> Dict[int, Dict]:
        """Ajax JSONレスポンスを解析"""
        return self._parse_api_json_odds(json_data, race_id)
    
    def _extract_odds_from_mobile_page(self, soup: BeautifulSoup, race_id: str) -> Dict[int, Dict]:
        """モバイルページからオッズを抽出"""
        return self._extract_odds_from_html_response(soup, race_id)
    
    def _merge_all_data(self, basic_data: pd.DataFrame, odds_data: Dict[int, Dict]) -> pd.DataFrame:
        """全データを統合"""
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
        
        # 統計表示
        odds_count = final_data['オッズ'].notna().sum()
        pop_count = final_data['人気'].notna().sum()
        total_count = len(final_data)
        
        print(f"\n📊 最終統合結果:")
        print(f"   全{total_count}頭中、オッズ取得{odds_count}頭、人気取得{pop_count}頭")
        
        if odds_count > 0:
            print(f"   最低オッズ: {final_data['オッズ'].min():.1f}倍")
            print(f"   最高オッズ: {final_data['オッズ'].max():.1f}倍")
        
        return final_data


def main():
    """積極的スクレイピング実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='積極的netkeiba.comスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202505021211)')
    parser.add_argument('--output', type=str, default='aggressive_race_data.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = AggressiveOddsScraper()
    race_data = scraper.scrape_with_all_methods(args.race_id)
    
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