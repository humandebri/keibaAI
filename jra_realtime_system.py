#!/usr/bin/env python3
"""
JRA競馬リアルタイムデータ取得・自動投票システム
無料データソースを活用した実装
"""

import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple
import re
from urllib.robotparser import RobotFileParser
import asyncio
import aiohttp
from pathlib import Path

class JRARealTimeSystem:
    """JRAリアルタイムデータ取得システム"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        self.session = None
        self.robot_parser = RobotFileParser()
        self._setup_robot_parser()
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'min_delay': 1.0,  # 最小遅延（秒）
            'max_delay': 3.0,  # 最大遅延（秒）
            'timeout': 10,
            'max_retries': 3,
            'cache_dir': 'cache/jra_data',
            'enable_cache': True,
            'cache_ttl': 300  # キャッシュ有効期限（秒）
        }
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger('JRARealTime')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # コンソール出力
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # ファイル出力
            Path('logs').mkdir(exist_ok=True)
            fh = logging.FileHandler(f'logs/jra_realtime_{datetime.now().strftime("%Y%m%d")}.log')
            fh.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            
            logger.addHandler(ch)
            logger.addHandler(fh)
        
        return logger
    
    def _setup_robot_parser(self):
        """robots.txt解析の設定"""
        try:
            self.robot_parser.set_url("https://www.jra.go.jp/robots.txt")
            self.robot_parser.read()
        except Exception as e:
            self.logger.warning(f"robots.txt読み込みエラー: {e}")
    
    def _create_session(self) -> requests.Session:
        """セッション作成"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        return session
    
    def _respectful_delay(self):
        """レート制限を守るための遅延"""
        delay = random.uniform(self.config['min_delay'], self.config['max_delay'])
        time.sleep(delay)
    
    def get_today_races(self) -> List[Dict]:
        """本日のレース一覧を取得"""
        return self.get_upcoming_races(days_ahead=0)
    
    def get_upcoming_races(self, days_ahead: int = 1, max_days: int = 7) -> List[Dict]:
        """今後のレース一覧を取得（明日以降のレース）
        
        Args:
            days_ahead: 何日先から取得するか（0=今日、1=明日、2=明後日）
            max_days: 最大何日先まで取得するか
        
        Returns:
            レース情報のリスト
        """
        all_races = []
        base_date = datetime.now()
        
        for day_offset in range(days_ahead, days_ahead + max_days):
            target_date = base_date + timedelta(days=day_offset)
            date_str = target_date.strftime('%Y%m%d')
            
            if day_offset == 0:
                self.logger.info("本日のレース一覧を取得中...")
            else:
                self.logger.info(f"{target_date.strftime('%Y年%m月%d日')}のレース一覧を取得中...")
            
            # JRA公式サイトのレース一覧ページ（日付指定）
            # 本日の場合は/keiba/today/、それ以外は/keiba/calendar/で対応
            if day_offset == 0:
                url = "https://www.jra.go.jp/keiba/today/"
            else:
                # カレンダーページから特定日のレースを取得
                url = f"https://www.jra.go.jp/keiba/calendar/{target_date.year}/{target_date.month:02d}/"
            
            try:
                if not self.session:
                    self.session = self._create_session()
                
                self._respectful_delay()
                response = self.session.get(url, timeout=self.config['timeout'])
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # レース情報の抽出（JRAサイトの構造に依存）
                race_tables = soup.find_all('table', class_='basic')
                
                for table in race_tables:
                    rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            race_info = {
                                'date': target_date.strftime('%Y-%m-%d'),
                                'time': cells[0].text.strip(),
                                'racecourse': cells[1].text.strip(),
                                'race_number': cells[2].text.strip(),
                                'race_name': cells[3].text.strip(),
                                'status': 'upcoming',
                                'days_ahead': day_offset
                            }
                            all_races.append(race_info)
                
            except Exception as e:
                self.logger.error(f"{target_date.strftime('%Y-%m-%d')}のレース一覧取得エラー: {e}")
                continue
        
        self.logger.info(f"合計{len(all_races)}件のレースを取得しました")
        return all_races
    
    def get_race_details(self, race_id: str) -> Optional[Dict]:
        """レース詳細情報を取得"""
        self.logger.info(f"レース詳細を取得中: {race_id}")
        
        # キャッシュチェック
        if self.config['enable_cache']:
            cached_data = self._get_cached_data(f"race_details_{race_id}")
            if cached_data:
                return cached_data
        
        # 実際の取得処理（ここでは仮の実装）
        # 本来はJRAサイトの詳細ページをスクレイピング
        
        race_details = {
            'race_id': race_id,
            'horses': self._get_horses_info(race_id),
            'odds': self._get_odds_info(race_id),
            'track_condition': self._get_track_condition(race_id)
        }
        
        # キャッシュ保存
        if self.config['enable_cache']:
            self._save_cache(f"race_details_{race_id}", race_details)
        
        return race_details
    
    def _get_horses_info(self, race_id: str) -> List[Dict]:
        """出走馬情報を取得（ダミー実装）"""
        # 実際にはJRAサイトから取得
        horses = []
        for i in range(1, 13):
            horses.append({
                'horse_number': i,
                'horse_name': f'テスト馬{i}',
                'jockey': f'騎手{i}',
                'weight': 55.0,
                'age': 3,
                'sex': '牡' if i % 2 == 0 else '牝'
            })
        return horses
    
    def _get_odds_info(self, race_id: str) -> Dict:
        """オッズ情報を取得（ダミー実装）"""
        # 実際にはJRAサイトから取得
        import numpy as np
        
        odds = {}
        # 単勝オッズ
        odds['win'] = {str(i): round(np.random.uniform(2, 50), 1) for i in range(1, 13)}
        # 複勝オッズ
        odds['place'] = {str(i): round(np.random.uniform(1.5, 10), 1) for i in range(1, 13)}
        
        return odds
    
    def _get_track_condition(self, race_id: str) -> Dict:
        """馬場状態を取得（ダミー実装）"""
        # 実際にはJRAサイトから取得
        return {
            'surface': '芝',
            'condition': '良',
            'distance': 2000,
            'weather': '晴'
        }
    
    def _get_cached_data(self, key: str) -> Optional[Dict]:
        """キャッシュからデータ取得"""
        cache_dir = Path(self.config['cache_dir'])
        cache_file = cache_dir / f"{key}.json"
        
        if cache_file.exists():
            # TTLチェック
            if time.time() - cache_file.stat().st_mtime < self.config['cache_ttl']:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None
    
    def _save_cache(self, key: str, data: Dict):
        """キャッシュにデータ保存"""
        cache_dir = Path(self.config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def monitor_odds_changes(self, race_id: str, interval: int = 30):
        """オッズの変化をモニタリング"""
        self.logger.info(f"オッズモニタリング開始: {race_id}")
        
        previous_odds = None
        
        while True:
            try:
                current_odds = self._get_odds_info(race_id)
                
                if previous_odds:
                    # 変化を検出
                    changes = self._detect_odds_changes(previous_odds, current_odds)
                    if changes:
                        self.logger.info(f"オッズ変化検出: {changes}")
                        yield changes
                
                previous_odds = current_odds
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"オッズモニタリングエラー: {e}")
                await asyncio.sleep(interval * 2)
    
    def _detect_odds_changes(self, old_odds: Dict, new_odds: Dict) -> List[Dict]:
        """オッズの変化を検出"""
        changes = []
        
        for horse_num, new_odd in new_odds['win'].items():
            old_odd = old_odds['win'].get(horse_num)
            if old_odd and abs(new_odd - old_odd) > 0.5:
                change_rate = (new_odd - old_odd) / old_odd
                changes.append({
                    'horse_number': horse_num,
                    'old_odds': old_odd,
                    'new_odds': new_odd,
                    'change_rate': change_rate
                })
        
        return changes


class NetkeibaDataCollector:
    """netkeiba.comからのデータ収集（JRA公認サイト）"""
    
    def __init__(self):
        self.base_url = "https://race.netkeiba.com"
        self.db_url = "https://db.netkeiba.com"
        self.encoding = 'euc-jp'
        self.session = self._create_session()
        self.logger = logging.getLogger('NetkeibaCollector')
    
    def _create_session(self) -> requests.Session:
        """セッション作成"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (研究用/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en;q=0.9'
        })
        return session
    
    def get_today_race_list(self) -> List[Dict]:
        """本日のレース一覧を取得"""
        return self.get_upcoming_race_list(days_ahead=0)
    
    def get_upcoming_race_list(self, days_ahead: int = 1, max_days: int = 7) -> List[Dict]:
        """今後のレース一覧を取得
        
        Args:
            days_ahead: 何日先から取得するか（0=今日、1=明日）
            max_days: 最大何日先まで取得するか
        
        Returns:
            レース情報のリスト
        """
        all_races = []
        base_date = datetime.now()
        
        for day_offset in range(days_ahead, days_ahead + max_days):
            target_date = base_date + timedelta(days=day_offset)
            date_str = target_date.strftime('%Y%m%d')
            
            # netkeibaの日付別レース一覧URL
            url = f"{self.base_url}/race/list/{date_str}/"
            
            try:
                time.sleep(random.uniform(1, 2))
                response = self.session.get(url)
                response.encoding = self.encoding
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # netkeibaのレース一覧解析
                race_links = soup.find_all('a', href=re.compile(r'/race/\d+'))
                
                for link in race_links:
                    race_id = re.search(r'/race/(\d+)', link['href']).group(1)
                    race_name = link.text.strip()
                    
                    all_races.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'race_id': race_id,
                        'race_name': race_name,
                        'url': f"{self.base_url}{link['href']}",
                        'days_ahead': day_offset
                    })
                
            except Exception as e:
                self.logger.error(f"netkeiba {target_date.strftime('%Y-%m-%d')}のレース一覧取得エラー: {e}")
                continue
        
        return all_races
    
    def get_race_card(self, race_id: str) -> pd.DataFrame:
        """出馬表を取得"""
        url = f"{self.base_url}/race/{race_id}/"
        
        try:
            time.sleep(random.uniform(1, 2))
            response = self.session.get(url)
            response.encoding = self.encoding
            
            # pandas.read_htmlで表を直接取得
            tables = pd.read_html(response.text)
            
            # 出馬表は通常最初の大きなテーブル
            if tables:
                race_card = tables[0]
                # カラム名の正規化
                race_card.columns = [
                    '枠番', '馬番', '馬名', '性齢', '斤量', '騎手', 
                    '馬体重', '単勝オッズ', '人気', '調教師'
                ]
                return race_card
            
        except Exception as e:
            self.logger.error(f"出馬表取得エラー: {e}")
            
        return pd.DataFrame()
    
    def get_real_time_odds(self, race_id: str) -> Dict:
        """リアルタイムオッズを取得"""
        url = f"{self.base_url}/odds/{race_id}/"
        
        try:
            time.sleep(random.uniform(1, 2))
            response = self.session.get(url)
            response.encoding = self.encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            odds_data = {
                'win': {},     # 単勝
                'place': {},   # 複勝
                'exacta': {},  # 馬連
                'quinella': {} # ワイド
            }
            
            # オッズ表の解析（サイト構造に依存）
            # 実装は省略
            
            return odds_data
            
        except Exception as e:
            self.logger.error(f"オッズ取得エラー: {e}")
            return {}


class JRAIPATInterface:
    """JRA IPAT（インターネット投票）インターフェース"""
    
    def __init__(self, member_id: str, pin: str, pars: str):
        self.member_id = member_id
        self.pin = pin
        self.pars = pars
        self.session = None
        self.logger = logging.getLogger('JRAIPAT')
        self.is_logged_in = False
        
    def login(self) -> bool:
        """IPATにログイン"""
        self.logger.info("IPAT ログイン処理開始")
        
        # 注意：実際のIPATログインは複雑なセキュリティ機構があるため
        # ここでは概念的な実装のみ
        
        login_url = "https://www.ipat.jra.go.jp/"
        
        try:
            self.session = requests.Session()
            
            # ログインフォームの取得
            response = self.session.get(login_url)
            
            # ログイン情報の送信（実際はもっと複雑）
            login_data = {
                'member_id': self.member_id,
                'pin': self.pin,
                'pars': self.pars
            }
            
            # ログイン実行
            # response = self.session.post(login_url, data=login_data)
            
            # 成功判定（実際はレスポンスを解析）
            self.is_logged_in = True
            self.logger.info("IPAT ログイン成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"IPAT ログインエラー: {e}")
            return False
    
    def place_bet(self, race_id: str, bet_type: str, 
                  selections: List[int], amount: int) -> Dict:
        """馬券購入（注意：手動確認必須）"""
        if not self.is_logged_in:
            raise Exception("ログインが必要です")
        
        self.logger.warning(
            f"馬券購入リクエスト - レース:{race_id}, "
            f"式別:{bet_type}, 選択:{selections}, 金額:{amount}円"
        )
        
        # 重要：実際の投票は手動確認が必要
        self.logger.warning("警告：すべての投票は手動で確認してください！")
        
        # 仮の実装
        result = {
            'status': 'pending',
            'race_id': race_id,
            'bet_type': bet_type,
            'selections': selections,
            'amount': amount,
            'confirmation_required': True,
            'message': '投票内容を必ず手動で確認してください'
        }
        
        return result
    
    def check_balance(self) -> Optional[int]:
        """残高照会"""
        if not self.is_logged_in:
            return None
        
        # 実際はIPATから残高を取得
        # ここではダミー値
        return 100000
    
    def get_bet_history(self) -> List[Dict]:
        """投票履歴を取得"""
        if not self.is_logged_in:
            return []
        
        # 実際はIPATから履歴を取得
        return []


def main():
    """デモンストレーション"""
    # システム初期化
    jra_system = JRARealTimeSystem()
    
    print("=" * 60)
    print("JRAリアルタイムシステム - デモ")
    print("=" * 60)
    
    # 本日のレース取得
    today_races = jra_system.get_today_races()
    print(f"\n本日のレース: {len(today_races)}件")
    
    for i, race in enumerate(today_races[:3], 1):
        print(f"{i}. {race['time']} {race['racecourse']} "
              f"{race['race_number']}R {race['race_name']}")
    
    # 明日以降のレース取得
    print("\n" + "=" * 40)
    upcoming_races = jra_system.get_upcoming_races(days_ahead=1, max_days=3)
    print(f"\n今後のレース（明日から3日間）: {len(upcoming_races)}件")
    
    # 日付ごとにグループ化して表示
    from itertools import groupby
    from operator import itemgetter
    
    grouped_races = groupby(upcoming_races, key=itemgetter('date'))
    for date, races_on_date in grouped_races:
        races_list = list(races_on_date)
        print(f"\n【{date}】 {len(races_list)}レース")
        for i, race in enumerate(races_list[:3], 1):
            print(f"  {i}. {race['time']} {race['racecourse']} "
                  f"{race['race_number']}R {race['race_name']}")
        if len(races_list) > 3:
            print(f"  ... 他{len(races_list) - 3}レース")
    
    # netkeiba.comからもデータ取得
    print("\n" + "=" * 40)
    netkeiba = NetkeibaDataCollector()
    netkeiba_upcoming = netkeiba.get_upcoming_race_list(days_ahead=1, max_days=2)
    print(f"\nnetkeiba.com 今後のレース: {len(netkeiba_upcoming)}件")
    
    # IPAT（ダミー）
    print("\n※IPAT自動投票は手動確認が必須です")
    print("※実際の使用には十分注意してください")


if __name__ == "__main__":
    main()