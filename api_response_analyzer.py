#!/usr/bin/env python3
"""
API レスポンス詳細分析ツール
取得できたAPIレスポンスの内容を詳細に分析
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random


class ApiResponseAnalyzer:
    """APIレスポンスを詳細分析"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        })
    
    def analyze_all_apis(self, race_id: str):
        """全APIエンドポイントを詳細分析"""
        print(f"🔬 API詳細分析開始: {race_id}")
        
        # 成功したAPIエンドポイント
        successful_apis = [
            f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}",
            f"https://race.netkeiba.com/odds/odds_get_form.html?type=1&race_id={race_id}",
        ]
        
        for api_url in successful_apis:
            print(f"\n{'='*60}")
            print(f"🎯 詳細分析: {api_url}")
            print(f"{'='*60}")
            
            self._analyze_single_api(api_url, race_id)
    
    def _analyze_single_api(self, url: str, race_id: str):
        """単一APIエンドポイントを詳細分析"""
        try:
            # 複数の方法でリクエスト
            methods = [
                {'method': 'GET', 'headers': {}},
                {'method': 'GET', 'headers': {'X-Requested-With': 'XMLHttpRequest'}},
                {'method': 'POST', 'headers': {'X-Requested-With': 'XMLHttpRequest'}, 'data': {'race_id': race_id}},
            ]
            
            for i, config in enumerate(methods):
                print(f"\n📡 方法{i+1}: {config['method']} リクエスト")
                
                try:
                    headers = self.session.headers.copy()
                    headers.update(config.get('headers', {}))
                    
                    if config['method'] == 'POST':
                        response = self.session.post(url, 
                                                   data=config.get('data', {}), 
                                                   headers=headers, 
                                                   timeout=15)
                    else:
                        response = self.session.get(url, headers=headers, timeout=15)
                    
                    print(f"   ステータス: {response.status_code}")
                    print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
                    print(f"   Content-Length: {len(response.text)}")
                    
                    if response.status_code == 200:
                        self._analyze_response_content(response, f"方法{i+1}")
                    
                    time.sleep(1)  # 負荷軽減
                    
                except Exception as e:
                    print(f"   ❌ エラー: {e}")
        
        except Exception as e:
            print(f"❌ API分析エラー: {e}")
    
    def _analyze_response_content(self, response, method_name: str):
        """レスポンス内容を詳細分析"""
        content = response.text
        print(f"\n🔍 {method_name} - レスポンス内容分析:")
        
        # 1. JSON解析を試行
        try:
            json_data = json.loads(content)
            print(f"   ✅ JSON形式検出")
            print(f"   📊 JSON構造: {self._analyze_json_structure(json_data)}")
            
            # JSONの中身を詳細表示
            print(f"   📝 JSON内容:")
            formatted_json = json.dumps(json_data, ensure_ascii=False, indent=2)
            print(f"      {formatted_json}")
            
            # オッズらしきデータを探す
            odds_data = self._find_odds_in_json(json_data)
            if odds_data:
                print(f"   🎯 オッズデータ発見: {odds_data}")
            
        except json.JSONDecodeError:
            print(f"   ❌ JSON形式ではありません")
        
        # 2. HTML解析を試行
        if '<' in content and '>' in content:
            print(f"   ✅ HTML形式検出")
            soup = BeautifulSoup(content, 'html.parser')
            
            # テーブルを探す
            tables = soup.find_all('table')
            print(f"   📊 テーブル数: {len(tables)}")
            
            for j, table in enumerate(tables):
                print(f"      テーブル{j+1}:")
                rows = table.find_all('tr')
                for k, row in enumerate(rows[:3]):  # 最初の3行のみ
                    cells = row.find_all(['td', 'th'])
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    print(f"         行{k+1}: {cell_texts}")
            
            # スクリプトタグを探す
            scripts = soup.find_all('script')
            print(f"   📊 スクリプトタグ数: {len(scripts)}")
            
            for j, script in enumerate(scripts):
                if script.string and ('odds' in script.string.lower() or '倍' in script.string):
                    print(f"      スクリプト{j+1} (オッズ関連):")
                    print(f"         {script.string[:200]}...")
        
        # 3. テキスト形式の分析
        print(f"   📝 テキスト内容 (最初の500文字):")
        print(f"      {content[:500]}")
        
        # 4. オッズらしき数値パターンを検索
        odds_patterns = [
            r'\d+\.\d+倍',
            r'odds.*?\d+\.\d+',
            r'人気.*?\d+',
            r'\d{1,3}\.\d{1,2}(?=\s|$|,)',
        ]
        
        print(f"   🔍 オッズパターン検索:")
        for pattern in odds_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"      パターン '{pattern}': {matches[:5]}")  # 最初の5個
        
        # 5. 特殊文字・エンコーディング確認
        print(f"   📊 文字エンコーディング情報:")
        print(f"      文字数: {len(content)}")
        print(f"      バイト数: {len(content.encode('utf-8'))}")
        
        # 特殊文字の検索
        special_chars = ['---.-', '**', '>>>', '---', '∞']
        for char in special_chars:
            count = content.count(char)
            if count > 0:
                print(f"      '{char}': {count}個")
    
    def _analyze_json_structure(self, json_data):
        """JSON構造を分析"""
        if isinstance(json_data, dict):
            keys = list(json_data.keys())
            return f"辞書型, キー: {keys[:5]}" + ("..." if len(keys) > 5 else "")
        elif isinstance(json_data, list):
            return f"リスト型, 要素数: {len(json_data)}"
        else:
            return f"プリミティブ型: {type(json_data).__name__}"
    
    def _find_odds_in_json(self, json_data):
        """JSON内からオッズデータを探す"""
        odds_keywords = ['odds', 'オッズ', '倍', 'popular', '人気', 'win', 'place']
        
        def search_recursive(obj, path=""):
            results = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # キー名がオッズ関連かチェック
                    if any(keyword in key.lower() for keyword in odds_keywords):
                        results.append(f"{current_path}: {value}")
                    
                    # 再帰的に探索
                    results.extend(search_recursive(value, current_path))
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    results.extend(search_recursive(item, current_path))
            
            elif isinstance(obj, (str, int, float)):
                # 数値がオッズらしいかチェック
                if isinstance(obj, (int, float)) and 1.0 <= obj <= 999.0:
                    results.append(f"{path}: {obj} (オッズ候補)")
                elif isinstance(obj, str):
                    # 文字列内のオッズパターンをチェック
                    if re.search(r'\d+\.\d+', str(obj)):
                        results.append(f"{path}: {obj} (オッズ候補)")
            
            return results
        
        return search_recursive(json_data)


def main():
    """APIレスポンス分析実行"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用法: python api_response_analyzer.py <race_id>")
        sys.exit(1)
    
    race_id = sys.argv[1]
    
    if not race_id.isdigit() or len(race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        sys.exit(1)
    
    analyzer = ApiResponseAnalyzer()
    analyzer.analyze_all_apis(race_id)


if __name__ == "__main__":
    main()