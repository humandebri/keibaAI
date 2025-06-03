#!/usr/bin/env python3
"""
実際のオッズのみスクレイパー
データ生成は一切せず、実際にスクレイピングしたオッズのみを取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from typing import Dict, List, Optional


class RealOddsOnlyScraper:
    """実際のオッズのみを取得するスクレイパー"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        })
        self.base_url = "https://race.netkeiba.com"
    
    def scrape_real_data_only(self, race_id: str) -> pd.DataFrame:
        """実際のデータのみを取得（生成なし）"""
        print(f"🏇 実際のデータのみスクレイピング: {race_id}")
        
        # 1. 基本情報を取得
        basic_data = self._scrape_basic_data(race_id)
        if basic_data.empty:
            return pd.DataFrame()
        
        # 2. 実際のオッズを取得（生成なし）
        real_odds = self._scrape_real_odds_only(race_id)
        
        # 3. 実際のデータのみ統合
        final_data = self._merge_real_data_only(basic_data, real_odds)
        
        return final_data
    
    def _scrape_basic_data(self, race_id: str) -> pd.DataFrame:
        """基本データをスクレイピング"""
        url = f"{self.base_url}/race/shutuba.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(1.0, 2.0))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='Shutuba_Table')
            
            if not table:
                print("❌ Shutuba_Tableが見つかりません")
                return pd.DataFrame()
            
            print("✓ 基本データ取得中...")
            
            horses_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 8:
                    continue
                
                umaban_text = cells[1].get_text(strip=True)
                if not (umaban_text.isdigit() and 1 <= int(umaban_text) <= 18):
                    continue
                
                horse_data = self._extract_basic_data_only(cells, race_id)
                if horse_data:
                    horses_data.append(horse_data)
            
            df = pd.DataFrame(horses_data)
            print(f"✓ 基本データ取得完了: {len(df)}頭")
            return df
            
        except Exception as e:
            print(f"❌ 基本データエラー: {e}")
            return pd.DataFrame()
    
    def _extract_basic_data_only(self, cells: List, race_id: str) -> Optional[Dict]:
        """基本データのみ抽出"""
        try:
            data = {'race_id': race_id}
            
            # 枠番
            waku_text = cells[0].get_text(strip=True)
            data['枠'] = int(waku_text) if waku_text.isdigit() and 1 <= int(waku_text) <= 8 else None
            
            # 馬番
            umaban_text = cells[1].get_text(strip=True)
            if not (umaban_text.isdigit() and 1 <= int(umaban_text) <= 18):
                return None
            data['馬番'] = int(umaban_text)
            
            # 馬名
            horse_name = None
            if len(cells) > 3:
                horse_cell = cells[3]
                horse_link = horse_cell.find('a', href=lambda href: href and 'horse' in href)
                if horse_link:
                    horse_name = horse_link.get_text(strip=True)
                elif horse_cell.get_text(strip=True):
                    horse_name = horse_cell.get_text(strip=True)
            data['馬名'] = horse_name
            
            # 性齢
            sei_rei = cells[4].get_text(strip=True) if len(cells) > 4 else None
            data['性齢'] = sei_rei
            
            # 斤量
            kinryo = None
            if len(cells) > 5:
                kinryo_text = cells[5].get_text(strip=True)
                if re.match(r'^5[0-9]\.[05]$', kinryo_text):
                    kinryo = float(kinryo_text)
            data['斤量'] = kinryo
            
            # 騎手
            jockey = None
            if len(cells) > 6:
                jockey_cell = cells[6]
                jockey_link = jockey_cell.find('a')
                if jockey_link:
                    jockey = jockey_link.get_text(strip=True)
                else:
                    jockey = jockey_cell.get_text(strip=True)
            data['騎手'] = jockey
            
            # 厩舎
            trainer = None
            if len(cells) > 7:
                trainer_text = cells[7].get_text(strip=True)
                trainer = re.sub(r'^(栗東|美浦)', '', trainer_text)
            data['厩舎'] = trainer
            
            # 馬体重
            horse_weight = None
            if len(cells) > 8:
                weight_text = cells[8].get_text(strip=True)
                if re.match(r'\d{3,4}\([+-]?\d+\)', weight_text):
                    horse_weight = weight_text
            data['馬体重'] = horse_weight
            
            return data
            
        except Exception:
            return None
    
    def _scrape_real_odds_only(self, race_id: str) -> Dict[int, Dict]:
        """実際のオッズのみを取得（生成は一切なし）"""
        print("💰 実際のオッズページから確定オッズ取得中...")
        
        # オッズページを試行
        odds_urls = [
            f"{self.base_url}/odds/index.html?race_id={race_id}",
            f"{self.base_url}/race/result.html?race_id={race_id}",
        ]
        
        for url in odds_urls:
            print(f"📊 {url} を確認中...")
            
            try:
                time.sleep(random.uniform(1.5, 2.5))
                response = self.session.get(url, timeout=15)
                
                if response.status_code != 200:
                    print(f"   ❌ ステータス: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 複数のテーブルパターンを試行
                odds_data = {}
                
                # パターン1: オッズテーブル
                odds_tables = soup.find_all('table')
                print(f"   📊 テーブル数: {len(odds_tables)}")
                
                for i, table in enumerate(odds_tables):
                    table_odds = self._extract_real_odds_from_table(table, race_id)
                    if table_odds:
                        print(f"   ✓ テーブル{i+1}から実際のオッズ取得: {len(table_odds)}頭")
                        odds_data.update(table_odds)
                
                if odds_data:
                    print(f"✅ 実際のオッズ取得成功: {len(odds_data)}頭")
                    return odds_data
                else:
                    print(f"   ⚠️ このページにはオッズデータなし")
                
            except Exception as e:
                print(f"   ❌ エラー: {e}")
        
        print("❌ 実際のオッズは見つかりませんでした")
        return {}
    
    def _extract_real_odds_from_table(self, table, race_id: str) -> Dict[int, Dict]:
        """テーブルから実際のオッズのみを抽出"""
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
            has_umaban = any('馬番' in text for text in header_texts)
            
            # オッズ関連テーブルでない場合はスキップ
            if not (has_odds or has_popularity):
                return odds_data
            
            print(f"      オッズテーブル候補発見: {header_texts}")
            
            # 列位置を特定
            popularity_col = -1
            umaban_col = -1
            odds_col = -1
            horse_name_col = -1
            
            for i, text in enumerate(header_texts):
                if '人気' in text:
                    popularity_col = i
                elif '馬番' in text:
                    umaban_col = i
                elif '馬名' in text:
                    horse_name_col = i
                elif 'オッズ' in text or '単勝' in text:
                    odds_col = i
            
            print(f"      列位置: 人気={popularity_col}, 馬番={umaban_col}, オッズ={odds_col}")
            
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
                
                # オッズ（実際の値のみ、---.-や**は除外）
                odds = None
                if odds_col >= 0 and odds_col < len(cells):
                    odds_text = cells[odds_col].get_text(strip=True)
                    if (odds_text and 
                        odds_text not in ['---.-', '**', '--', '', '―', '－'] and
                        not odds_text.startswith('---')):
                        try:
                            # 数値パターンマッチ
                            if re.match(r'^\d+\.\d+$', odds_text):
                                odds_val = float(odds_text)
                                # 妥当なオッズ範囲チェック
                                if 1.0 <= odds_val <= 999.0:
                                    odds = odds_val
                        except:
                            pass
                
                # 実際のデータがある場合のみ保存
                if popularity and umaban and odds:
                    odds_data[umaban] = {
                        '人気': popularity,
                        'オッズ': odds
                    }
                    print(f"      ✓ 実データ: {umaban}番 {popularity}人気 {odds}倍")
        
        except Exception as e:
            print(f"      ❌ テーブル解析エラー: {e}")
        
        return odds_data
    
    def _merge_real_data_only(self, basic_data: pd.DataFrame, odds_data: Dict[int, Dict]) -> pd.DataFrame:
        """実際のデータのみを統合"""
        if basic_data.empty:
            return pd.DataFrame()
        
        final_data = basic_data.copy()
        
        # オッズと人気の列を追加（デフォルトはNone）
        final_data['オッズ'] = None
        final_data['人気'] = None
        
        # 実際のオッズデータがある場合のみ統合
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
        total_count = len(final_data)
        
        print(f"📊 実データ統合結果: 全{total_count}頭中、実オッズ{odds_count}頭、実人気{pop_count}頭")
        
        return final_data


def main():
    """実際のオッズのみスクレイパー実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='実際のオッズのみスクレイパー')
    parser.add_argument('race_id', type=str, help='レースID (例: 202305021211)')
    parser.add_argument('--output', type=str, default='real_odds_only.csv', help='出力CSVファイル')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    scraper = RealOddsOnlyScraper()
    race_data = scraper.scrape_real_data_only(args.race_id)
    
    if race_data.empty:
        print("❌ データ取得に失敗しました")
        return
    
    # CSV保存
    race_data.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n💾 実データ保存: {args.output}")
    
    # 結果表示
    print(f"\n📊 実取得データ: {len(race_data)}頭")
    print("\n🏇 実際のデータ:")
    
    # 人気順があれば人気順、なければ馬番順
    if 'オッズ' in race_data.columns and race_data['オッズ'].notna().any():
        display_data = race_data.sort_values('オッズ')
        print("（オッズ順）")
    else:
        display_data = race_data.sort_values('馬番')
        print("（馬番順）")
    
    for _, horse in display_data.iterrows():
        odds_str = f"{horse['オッズ']}倍" if pd.notna(horse['オッズ']) else "取得できず"
        pop_str = f"{horse['人気']}人気" if pd.notna(horse['人気']) else "取得できず"
        
        print(f"  {horse['枠']}枠{horse['馬番']:2d}番 {horse['馬名']:15s} "
              f"{horse['騎手']:8s} {horse['厩舎']:8s} {horse['馬体重'] or '不明':10s} "
              f"{odds_str:10s} {pop_str}")
    
    # 取得状況
    if 'オッズ' in race_data.columns:
        odds_count = race_data['オッズ'].notna().sum()
        if odds_count > 0:
            print(f"\n✅ 実際のオッズ取得成功: {odds_count}頭")
            print(f"   最低オッズ: {race_data['オッズ'].min():.1f}倍")
            print(f"   最高オッズ: {race_data['オッズ'].max():.1f}倍")
        else:
            print(f"\n⚠️ 実際のオッズは取得できませんでした")
            print(f"💡 このレースはオッズが未確定か、確定前の状態です")


if __name__ == "__main__":
    main()