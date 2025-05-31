#!/usr/bin/env python3
"""
払戻データ取得の最終テスト
"""

from bs4 import BeautifulSoup

# 先ほどのデバッグで確認したHTMLの構造を再現
html_sample = """
<table class="pay_table_01">
<tr><td>7</td><td>140</td><td>1</td></tr>
<tr><td>7
6
13</td><td>110
230
400</td><td>1
3
6</td></tr>
<tr><td>3 - 4</td><td>790</td><td>2</td></tr>
<tr><td>6 - 7</td><td>760</td><td>2</td></tr>
</table>
"""

def test_payout_extraction():
    """払戻データ抽出のテスト"""
    soup = BeautifulSoup(html_sample, "html.parser")
    
    payout_data = {
        'win': {},
        'place': {},
        'quinella': {},
        'exacta': {},
        'wide': {},
        'trio': {},
        'trifecta': {}
    }
    
    # 払戻テーブルを取得
    table = soup.find("table", {"class": "pay_table_01"})
    rows = table.find_all("tr")
    
    # 馬券種別（テーブル1の場合）
    bet_types = ['単勝', '複勝', '枠連', '馬連']
    
    for row_idx, row in enumerate(rows):
        if row_idx < len(bet_types):
            cells = row.find_all("td")
            if len(cells) >= 2:
                bet_type = bet_types[row_idx]
                combination = cells[0].text.strip()
                payout_text = cells[1].text.strip()
                
                print(f"\n{bet_type}:")
                print(f"  組み合わせ: {repr(combination)}")
                print(f"  払戻: {repr(payout_text)}")
                
                if bet_type == '単勝':
                    payout_data['win'][combination] = int(payout_text)
                elif bet_type == '複勝':
                    # 改行で分割
                    combos = combination.split('\n')
                    pays = payout_text.split('\n')
                    print(f"  → 分割後: {combos} / {pays}")
                    for i, combo in enumerate(combos):
                        if i < len(pays) and combo.strip():
                            payout_data['place'][combo.strip()] = int(pays[i])
                elif bet_type == '馬連':
                    payout_data['quinella'][combination] = int(payout_text)
    
    print("\n=== 抽出結果 ===")
    import json
    print(json.dumps(payout_data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_payout_extraction()