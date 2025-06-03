#!/usr/bin/env python3
"""
netkeiba.com のオッズ構造を詳細に調査するデバッグツール
"""

import requests
from bs4 import BeautifulSoup
import re
import json

def debug_odds_structure(race_id: str):
    """オッズの取得構造を詳細に調査"""
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    print(f"🔍 レース {race_id} のオッズ構造を調査中...")
    
    # 1. 基本出馬表ページのHTMLを詳細解析
    print("\n1️⃣ 出馬表ページの解析...")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    response = session.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # JavaScriptやscriptタグ内のオッズ情報を探す
    scripts = soup.find_all('script')
    print(f"スクリプトタグ数: {len(scripts)}")
    
    for i, script in enumerate(scripts):
        if script.string:
            content = script.string
            # オッズらしき数値パターンを探す
            if 'odds' in content.lower() or '倍' in content:
                print(f"スクリプト{i}にオッズ関連コード発見:")
                print(content[:500] + "..." if len(content) > 500 else content)
                print("-" * 50)
    
    # 2. オッズ専用ページの解析
    print("\n2️⃣ オッズ専用ページの解析...")
    odds_urls = [
        f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        f"https://race.netkeiba.com/odds/win.html?race_id={race_id}",
    ]
    
    for url in odds_urls:
        print(f"\n📊 {url}")
        try:
            response = session.get(url)
            print(f"ステータス: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # オッズテーブルを探す
                tables = soup.find_all('table')
                print(f"テーブル数: {len(tables)}")
                
                for j, table in enumerate(tables):
                    if j < 3:  # 最初の3つのテーブルを詳細表示
                        print(f"テーブル{j}:")
                        rows = table.find_all('tr')
                        for k, row in enumerate(rows[:5]):  # 最初の5行
                            cells = row.find_all(['td', 'th'])
                            cell_texts = [cell.get_text(strip=True) for cell in cells]
                            print(f"  行{k}: {cell_texts}")
                        print()
                
                # オッズらしき数値を全検索
                text_content = soup.get_text()
                odds_patterns = [
                    r'\d+\.\d+倍',
                    r'単勝.*?\d+\.\d+',
                    r'オッズ.*?\d+\.\d+',
                    r'\d{1,3}\.\d{1,2}(?=\s*[^\d])',
                ]
                
                for pattern in odds_patterns:
                    matches = re.findall(pattern, text_content)
                    if matches:
                        print(f"パターン '{pattern}' でマッチ: {matches[:10]}")  # 最初の10個
                
        except Exception as e:
            print(f"エラー: {e}")
    
    # 3. Ajax/API エンドポイントの調査
    print("\n3️⃣ Ajax/APIエンドポイントの調査...")
    api_urls = [
        f"https://race.netkeiba.com/api/api_get_odds.html?race_id={race_id}",
        f"https://race.netkeiba.com/api/race_before.html?race_id={race_id}",
        f"https://race.netkeiba.com/race/odds_ajax.html?race_id={race_id}",
    ]
    
    for url in api_urls:
        print(f"\n🔌 {url}")
        try:
            headers = session.headers.copy()
            headers['X-Requested-With'] = 'XMLHttpRequest'
            headers['Referer'] = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
            
            response = session.get(url, headers=headers)
            print(f"ステータス: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            
            if response.status_code == 200:
                content = response.text[:1000]  # 最初の1000文字
                print(f"レスポンス内容: {content}")
                
                # JSON解析を試す
                try:
                    data = response.json()
                    print(f"JSON データ: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
                except:
                    print("JSON解析失敗")
                    
        except Exception as e:
            print(f"エラー: {e}")
    
    # 4. ページのHTMLを全保存（詳細解析用）
    print("\n4️⃣ HTMLコンテンツの保存...")
    with open(f"debug_html_{race_id}.html", 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"HTMLを debug_html_{race_id}.html に保存しました")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用法: python debug_odds_structure.py <race_id>")
        sys.exit(1)
    
    race_id = sys.argv[1]
    debug_odds_structure(race_id)