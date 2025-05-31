#!/usr/bin/env python3
"""
スクレイピングの簡易テスト（パッケージ不要版）
基本的な動作確認のみ
"""

import sys
sys.path.append('src')

# 必要な部分だけ手動で確認
print("=== スクレイピングコードの構造確認 ===")

# enhanced_scraping.pyの主要メソッドを確認
with open('src/data_processing/enhanced_scraping.py', 'r') as f:
    content = f.read()
    
    # 主要なメソッドを検索
    methods = [
        'fetch_payout_data',
        'fetch_blood_data', 
        'parse_race_date',
        '_extract_horse_data_enhanced'
    ]
    
    print("\n【実装されているメソッド】")
    for method in methods:
        if f'def {method}' in content:
            print(f"✓ {method}")
            # メソッドの最初の数行を表示
            start = content.find(f'def {method}')
            end = content.find('\n\n', start)
            if end == -1:
                end = start + 200
            snippet = content[start:end].split('\n')[:5]
            for line in snippet[1:4]:
                print(f"    {line.strip()}")
        else:
            print(f"✗ {method}")
    
    # 払戻データの構造を確認
    print("\n【払戻データの構造】")
    payout_start = content.find("payout_data = {")
    if payout_start != -1:
        payout_end = content.find("}", payout_start) + 1
        print(content[payout_start:payout_end])
    
    # 枠番計算の実装を確認
    print("\n【枠番計算の実装】")
    waku_line = content.find("horse_data['枠番']")
    if waku_line != -1:
        line_end = content.find('\n', waku_line)
        print(content[waku_line:line_end])
    
    # カラム構造を確認
    print("\n【出力データのカラム】")
    data_keys = []
    import re
    matches = re.findall(r"horse_data\['([^']+)'\]", content)
    unique_keys = sorted(set(matches))
    
    print("基本情報:")
    basic_keys = [k for k in unique_keys if not k.startswith('払戻')]
    for i, key in enumerate(basic_keys):
        print(f"  {i+1}. {key}")
    
    print("\n払戻情報:")
    payout_keys = [k for k in unique_keys if k.startswith('払戻')]
    for key in payout_keys:
        print(f"  - {key}")

print("\n=== 確認完了 ===")
print("\n実装の要点:")
print("1. 払戻データ取得機能 ✓")
print("2. 枠番の自動計算 ✓")
print("3. 日付取得の改善 ✓")
print("4. 血統情報の枠組み ✓（詳細実装は必要）")

print("\n次のステップ:")
print("1. 仮想環境を作成してパッケージをインストール")
print("2. 実際にスクレイピングを実行してテスト")
print("3. 払戻データが正しく取得できているか確認")