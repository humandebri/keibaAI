#!/usr/bin/env python3
"""
netkeiba.com レース状況分析ツール
レースの状態（開催前/開催後）とオッズの有無を確認
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from datetime import datetime
from typing import Dict, List, Optional


class RaceStatusAnalyzer:
    """レースの状況とオッズの取得可能性を分析"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
        })
    
    def analyze_race(self, race_id: str) -> Dict:
        """レースの状況を詳細分析"""
        print(f"🔍 レース {race_id} の状況分析中...")
        
        # レースIDから日付を抽出
        race_info = self._parse_race_id(race_id)
        print(f"📅 レース日時: {race_info['date_str']}")
        print(f"🏟️ 競馬場: {race_info['place']}")
        print(f"🏁 レース番号: {race_info['race_num']}")
        
        # 出馬表ページを確認
        shutuba_status = self._check_shutuba_page(race_id)
        
        # オッズページを確認
        odds_status = self._check_odds_page(race_id)
        
        # 結果ページを確認
        result_status = self._check_result_page(race_id)
        
        # 総合判定
        analysis = {
            'race_id': race_id,
            'race_info': race_info,
            'shutuba_available': shutuba_status['available'],
            'shutuba_horse_count': shutuba_status['horse_count'],
            'odds_available': odds_status['available'],
            'odds_set': odds_status['odds_set'],
            'result_available': result_status['available'],
            'race_status': self._determine_race_status(shutuba_status, odds_status, result_status),
            'recommendation': self._get_recommendation(shutuba_status, odds_status, result_status)
        }
        
        self._print_analysis(analysis)
        return analysis
    
    def _parse_race_id(self, race_id: str) -> Dict:
        """レースIDから情報を抽出"""
        if len(race_id) != 12:
            return {'error': 'レースIDが12桁ではありません'}
        
        year = race_id[:4]
        place_code = race_id[4:6] 
        meeting = race_id[6:8]
        day = race_id[8:10]
        race_num = race_id[10:12]
        
        place_dict = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
        }
        
        place = place_dict.get(place_code, f"不明({place_code})")
        
        return {
            'year': year,
            'place_code': place_code,
            'place': place,
            'meeting': meeting,
            'day': day,
            'race_num': race_num,
            'date_str': f"{year}年 {place} {meeting}回 {day}日目 {race_num}R"
        }
    
    def _check_shutuba_page(self, race_id: str) -> Dict:
        """出馬表ページの状況を確認"""
        print("📋 出馬表ページ確認中...")
        
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(0.5, 1.0))
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='Shutuba_Table')
            
            if not table:
                return {'available': False, 'horse_count': 0}
            
            # 馬の数をカウント
            horse_count = 0
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 8:
                    if len(cells) > 1:
                        umaban_text = cells[1].get_text(strip=True)
                        if umaban_text.isdigit() and 1 <= int(umaban_text) <= 18:
                            horse_count += 1
            
            return {'available': True, 'horse_count': horse_count}
            
        except Exception as e:
            return {'available': False, 'horse_count': 0, 'error': str(e)}
    
    def _check_odds_page(self, race_id: str) -> Dict:
        """オッズページの状況を確認"""
        print("💰 オッズページ確認中...")
        
        url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(1.0, 2.0))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # オッズテーブルを探す
            odds_set = False
            tables = soup.find_all('table')
            
            for table in tables:
                header_row = table.find('tr')
                if header_row:
                    header_cells = header_row.find_all(['th', 'td'])
                    header_texts = [cell.get_text(strip=True) for cell in header_cells]
                    
                    if ('人気' in header_texts and ('単勝オッズ' in header_texts or 'オッズ' in ' '.join(header_texts))):
                        # データ行をチェック
                        rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
                        for row in rows[:5]:  # 最初の5行をチェック
                            cells = row.find_all(['td', 'th'])
                            for cell in cells:
                                cell_text = cell.get_text(strip=True)
                                # 実際のオッズ値があるかチェック
                                if (re.match(r'^\d+\.\d+$', cell_text) and 
                                    float(cell_text) > 1.0 and float(cell_text) < 999.0):
                                    odds_set = True
                                    break
                            if odds_set:
                                break
                        break
            
            return {'available': True, 'odds_set': odds_set}
            
        except Exception as e:
            return {'available': False, 'odds_set': False, 'error': str(e)}
    
    def _check_result_page(self, race_id: str) -> Dict:
        """結果ページの状況を確認"""
        print("🏆 結果ページ確認中...")
        
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        
        try:
            time.sleep(random.uniform(0.5, 1.0))
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # 結果テーブルがあるかチェック
                result_table = soup.find('table', class_='race_table_01')
                return {'available': result_table is not None}
            else:
                return {'available': False}
                
        except Exception:
            return {'available': False}
    
    def _determine_race_status(self, shutuba_status: Dict, odds_status: Dict, result_status: Dict) -> str:
        """レースの状態を判定"""
        if result_status['available']:
            return "終了済み"
        elif odds_status['available'] and odds_status['odds_set']:
            return "開催中/オッズ確定"
        elif odds_status['available'] and not odds_status['odds_set']:
            return "開催前/オッズ未確定"
        elif shutuba_status['available']:
            return "開催前/出馬表のみ"
        else:
            return "データなし"
    
    def _get_recommendation(self, shutuba_status: Dict, odds_status: Dict, result_status: Dict) -> str:
        """推奨アクションを提案"""
        if result_status['available']:
            return "✅ 結果データ取得可能（過去レースの分析に使用）"
        elif odds_status['available'] and odds_status['odds_set']:
            return "🎯 オッズデータ取得可能（リアルタイム予想に使用）"
        elif odds_status['available'] and not odds_status['odds_set']:
            return "⏰ オッズ未確定（レース当日まで待機またはデモデータで代用）"
        elif shutuba_status['available']:
            return "📋 基本データのみ取得可能（馬体重、騎手等の分析に使用）"
        else:
            return "❌ データ取得不可"
    
    def _print_analysis(self, analysis: Dict):
        """分析結果を表示"""
        print("\n" + "="*60)
        print("📊 レース分析結果")
        print("="*60)
        print(f"🏇 レース: {analysis['race_info']['date_str']}")
        print(f"📋 出馬表: {'✅ 利用可能' if analysis['shutuba_available'] else '❌ 利用不可'}")
        if analysis['shutuba_available']:
            print(f"   └ 出走馬数: {analysis['shutuba_horse_count']}頭")
        print(f"💰 オッズ: {'✅ 利用可能' if analysis['odds_available'] else '❌ 利用不可'}")
        if analysis['odds_available']:
            print(f"   └ オッズ確定: {'✅ 確定済み' if analysis['odds_set'] else '❌ 未確定'}")
        print(f"🏆 結果: {'✅ 利用可能' if analysis['result_available'] else '❌ 利用不可'}")
        print(f"📊 状態: {analysis['race_status']}")
        print(f"💡 推奨: {analysis['recommendation']}")
        print("="*60)


def main():
    """テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='レース状況分析ツール')
    parser.add_argument('race_id', type=str, help='レースID (例: 202406020311)')
    
    args = parser.parse_args()
    
    if not args.race_id.isdigit() or len(args.race_id) != 12:
        print("❌ レースIDは12桁の数字で入力してください")
        return
    
    analyzer = RaceStatusAnalyzer()
    analysis = analyzer.analyze_race(args.race_id)


if __name__ == "__main__":
    main()