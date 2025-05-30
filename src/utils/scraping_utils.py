"""
競馬データスクレイピング用ユーティリティ関数
"""
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RaceScraper:
    """競馬データスクレイピング用クラス"""
    
    def __init__(self, user_agents: Optional[List[str]] = None):
        """
        初期化
        
        Args:
            user_agents: 使用するユーザーエージェントのリスト
        """
        self.user_agents = user_agents or [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
        ]
        self.failed_urls = set()
        
    def fetch_with_retry(self, url: str, max_retries: int = 3, 
                        timeout: int = 10) -> Optional[bytes]:
        """
        リトライ機能付きHTTPリクエスト
        
        Args:
            url: 取得するURL
            max_retries: 最大リトライ回数
            timeout: タイムアウト秒数
            
        Returns:
            レスポンスのコンテンツ、失敗時はNone
        """
        if url in self.failed_urls:
            return None
            
        for attempt in range(max_retries):
            try:
                headers = {'User-Agent': random.choice(self.user_agents)}
                response = requests.get(url, headers=headers, timeout=timeout)
                
                if response.status_code == 404:
                    self.failed_urls.add(url)
                    logger.warning(f"404 Not Found: {url}")
                    return None
                    
                response.raise_for_status()
                return response.content
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10) + random.random()
                    time.sleep(wait_time)
                    
        self.failed_urls.add(url)
        return None


def parse_weight(weight_str: str) -> Tuple[int, int]:
    """
    馬体重文字列をパース
    
    Args:
        weight_str: "480(+2)" のような馬体重文字列
        
    Returns:
        (体重, 体重差分) のタプル
    """
    try:
        if '(' in weight_str:
            weight = int(weight_str.split('(')[0])
            diff = int(weight_str.split('(')[1].replace(')', ''))
            return weight, diff
    except (ValueError, IndexError):
        pass
    return 0, 0


def parse_race_details(soup: BeautifulSoup) -> Dict[str, str]:
    """
    レース詳細情報を抽出
    
    Args:
        soup: BeautifulSoupオブジェクト
        
    Returns:
        レース詳細情報の辞書
    """
    details = {
        'surface': '',     # 芝/ダート
        'distance': '',    # 距離
        'direction': '',   # 回り
        'condition': '',   # 馬場状態
        'weather': '',     # 天気
        'date': '',        # 日付
        'class': '',       # クラス
        'title': ''        # レース名
    }
    
    try:
        # レースデータの抽出
        race_data_div = soup.find("div", class_="racedata")
        if race_data_div:
            items = race_data_div.text.strip().split("/")
            if len(items) >= 3:
                # 芝/ダート、回り、距離の解析
                track_info = items[0].strip()
                if track_info:
                    details['surface'] = track_info[0]
                    if len(track_info) > 1:
                        details['direction'] = track_info[1]
                    details['distance'] = ''.join(filter(str.isdigit, track_info))
                
                # 天気
                if len(items) > 1 and ":" in items[1]:
                    details['weather'] = items[1].split(":")[-1].strip()
                    
                # 馬場状態
                if len(items) > 2 and ":" in items[2]:
                    details['condition'] = items[2].split(":")[-1].strip()
        
        # 日付とクラス情報
        smalltxt = soup.find("p", class_="smalltxt")
        if smalltxt:
            info_parts = smalltxt.text.strip().split()
            if info_parts:
                details['date'] = info_parts[0]
            if len(info_parts) > 2:
                details['class'] = info_parts[2].split()[0]
        
        # レース名
        h1_tags = soup.find_all("h1")
        if len(h1_tags) > 1:
            details['title'] = h1_tags[1].text.strip()
            
    except Exception as e:
        logger.error(f"Failed to parse race details: {e}")
        
    return details


def generate_race_ids(year: int, place_dict: Dict[str, str]) -> List[str]:
    """
    指定年のレースIDを生成
    
    Args:
        year: 年
        place_dict: 場所コードと名前の辞書
        
    Returns:
        レースIDのリスト
    """
    race_ids = []
    for place_code in place_dict:
        for round_num in range(1, 7):      # 開催回数
            for day in range(1, 10):       # 開催日数
                for race in range(1, 13):  # レース番号
                    race_id = f"{year}{place_code}{round_num:02d}{day:02d}{race:02d}"
                    race_ids.append(race_id)
    return race_ids