#!/usr/bin/env python3
"""
拡張版競馬データスクレイピングモジュール
複勝・馬連・馬単・三連複・三連単のオッズも取得
日付情報の修正も含む
"""

import requests
from bs4 import BeautifulSoup
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import time
import os
import random
import argparse
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import re


class EnhancedRaceScraper:
    """拡張版: netkeiba.comから競馬データとオッズを取得するクラス"""
    
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
    
    def __init__(self, output_dir: str = "data_enhanced", max_workers: int = 3):
        """
        Args:
            output_dir: データ保存先ディレクトリ
            max_workers: 並列処理のワーカー数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_data(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """URLからデータを取得"""
        retries = 0
        while retries < max_retries:
            try:
                headers = {'User-Agent': random.choice(self.USER_AGENTS)}
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                return r.content
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                retries += 1
                wait_time = random.uniform(0.1, 0.5)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        return None
    
    def parse_race_date(self, soup) -> str:
        """レース日付を正確に取得"""
        try:
            # 複数の方法で日付を探す
            
            # 方法1: race_infoから
            race_info = soup.find("div", {"class": "race_info"})
            if race_info:
                date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', race_info.text)
                if date_match:
                    year, month, day = date_match.groups()
                    return f"{year}/{month.zfill(2)}/{day.zfill(2)}"
            
            # 方法2: data_introから
            data_intro = soup.find("div", {"class": "data_intro"})
            if data_intro:
                date_text = data_intro.find("p", {"class": "smalltxt"})
                if date_text:
                    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_text.text)
                    if date_match:
                        year, month, day = date_match.groups()
                        return f"{year}/{month.zfill(2)}/{day.zfill(2)}"
            
            # 方法3: race_idから推定
            # race_idの形式: YYYYppkkndrr (年4桁+場所2桁+開催回2桁+日数2桁+レース番号2桁)
            race_title = soup.find("h1", {"class": "race_name"})
            if race_title:
                race_id_match = re.search(r'race_id=(\d+)', str(soup))
                if race_id_match:
                    race_id = race_id_match.group(1)
                    year = race_id[:4]
                    # この方法では月日は正確に取得できないため、他の情報と組み合わせる必要がある
                    return f"{year}/XX/XX"  # 暫定値
            
        except Exception as e:
            print(f"Error parsing date: {e}")
        
        return ""
    
    def fetch_payout_data(self, race_id: str, soup=None) -> Dict[str, Any]:
        """払戻金データを取得（既存のsoupからも取得可能）"""
        payout_data = {
            'win': {},      # 単勝
            'place': {},    # 複勝
            'quinella': {}, # 馬連
            'exacta': {},   # 馬単
            'wide': {},     # ワイド
            'trio': {},     # 三連複
            'trifecta': {}  # 三連単
        }
        
        # soupが渡されていない場合は取得しない（レースページ内で取得）
        if soup is None:
            return payout_data
        
        # 払戻テーブルを探す（db.netkeiba.comの場合）
        payout_tables = soup.find_all("table", {"class": "pay_table_01"})
        
        # デバッグ: テーブルが見つかったか確認
        if not payout_tables:
            # 別の方法でも探す
            all_tables = soup.find_all("table")
            for table in all_tables:
                if any(word in str(table) for word in ["単勝", "複勝", "払戻"]):
                    payout_tables = [table]
                    break
        
        for table in payout_tables:
            rows = table.find_all("tr")
            for row in rows:
                try:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        # db.netkeiba.comの形式: 馬券種 | 番号 | 払戻金 | 人気
                        bet_type = cells[0].text.strip()
                        combination = cells[1].text.strip()
                        payout_text = cells[2].text.strip()
                        
                        # 金額を抽出（カンマを除去して数値に変換）
                        payout = int(payout_text.replace(",", "").replace("円", ""))
                        
                        if "単勝" in bet_type:
                            payout_data['win'][combination] = payout
                        elif "複勝" in bet_type:
                            # 複勝は複数の馬番がある場合がある（改行で区切られている）
                            combinations = combination.split('\n')
                            payouts = payout_text.split('\n')
                            for i, combo in enumerate(combinations):
                                if i < len(payouts) and combo.strip():
                                    try:
                                        payout_value = int(payouts[i].replace(",", "").replace("円", ""))
                                        payout_data['place'][combo.strip()] = payout_value
                                    except:
                                        pass
                        elif "枠連" in bet_type:
                            # 枠連は今回は扱わない
                            pass
                        elif "馬連" in bet_type:
                            payout_data['quinella'][combination] = payout
                        elif "ワイド" in bet_type:
                            # ワイドも複数の組み合わせがある
                            combinations = combination.split('\n')
                            payouts = payout_text.split('\n')
                            for i, combo in enumerate(combinations):
                                if i < len(payouts) and combo.strip():
                                    try:
                                        payout_value = int(payouts[i].replace(",", "").replace("円", ""))
                                        payout_data['wide'][combo.strip()] = payout_value
                                    except:
                                        pass
                        elif "馬単" in bet_type:
                            payout_data['exacta'][combination] = payout
                        elif "三連複" in bet_type:
                            payout_data['trio'][combination] = payout
                        elif "三連単" in bet_type:
                            payout_data['trifecta'][combination] = payout
                            
                except Exception as e:
                    continue
        
        return payout_data
    
    def fetch_blood_data(self, horse_id: str) -> Dict[str, str]:
        """血統情報を取得"""
        blood_data = {
            'father': '',      # 父
            'mother': '',      # 母
            'father_father': '', # 父父
            'father_mother': '', # 父母
            'mother_father': '', # 母父
            'mother_mother': '', # 母母
        }
        
        # 馬の詳細ページから血統情報を取得
        url = f"https://db.netkeiba.com/horse/{horse_id}"
        content = self.fetch_data(url)
        
        if not content:
            return blood_data
        
        soup = BeautifulSoup(content, "html.parser")
        
        # 血統テーブルを探す
        blood_table = soup.find("table", {"class": "blood_table"})
        
        if blood_table:
            # 血統情報の抽出ロジック
            # 実際のHTML構造に合わせて実装
            pass
        
        return blood_data
    
    def parse_race_data_enhanced(self, race_id: str, content: bytes, place_code: str, place: str) -> List[Dict[str, Any]]:
        """拡張版: HTMLコンテンツからレースデータを抽出"""
        if content is None:
            return []
            
        soup = BeautifulSoup(content, "html.parser", from_encoding="euc-jp")
        main_table = soup.find("table", {"class": "race_table_01 nk_tb_common"})
        
        if not main_table:
            print(f'No data found for race_id: {race_id}')
            return []
        
        race_data = []
        
        # レース詳細情報の取得
        try:
            race_info = self._extract_race_info(soup)
        except Exception as e:
            print(f"Error extracting race info for {race_id}: {e}")
            return []
        
        # 日付の取得（修正版）
        date = self.parse_race_date(soup)
        
        # レースタイトル
        title_elem = soup.find("h1", {"class": "race_name"})
        title = title_elem.text.strip() if title_elem else ""
        
        # 払戻データの取得（レースが終了している場合）
        payout_data = self.fetch_payout_data(race_id, soup)
        
        # 各馬のデータを抽出
        for row in main_table.find_all("tr")[1:]:  # ヘッダ行をスキップ
            cols = row.find_all("td")
            if len(cols) < 21:
                continue
                
            horse_data = self._extract_horse_data_enhanced(
                cols, race_id, race_info, title, date,
                place_code, place, payout_data
            )
            race_data.append(horse_data)
        
        return race_data
    
    def _extract_race_info(self, soup) -> Dict[str, str]:
        """レース情報を抽出（改良版）"""
        race_info = {"sur": "", "rou": "", "dis": "", "con": "", "wed": ""}
        
        # race_infoクラスから情報を取得
        race_info_elem = soup.find("div", {"class": "race_info"})
        if race_info_elem:
            info_text = race_info_elem.text
            
            # 距離
            dist_match = re.search(r'(\d+)m', info_text)
            if dist_match:
                race_info["dis"] = dist_match.group(1)
            
            # 芝・ダート
            if "芝" in info_text:
                race_info["sur"] = "芝"
            elif "ダ" in info_text or "ダート" in info_text:
                race_info["sur"] = "ダ"
            
            # 回り
            if "右" in info_text:
                race_info["rou"] = "右"
            elif "左" in info_text:
                race_info["rou"] = "左"
            
            # 馬場状態
            conditions = ["良", "稍重", "重", "不良"]
            for cond in conditions:
                if cond in info_text:
                    race_info["con"] = cond
                    break
            
            # 天気
            weathers = ["晴", "曇", "雨", "雪", "小雨"]
            for weather in weathers:
                if weather in info_text:
                    race_info["wed"] = weather
                    break
        
        return race_info
    
    def _extract_horse_data_enhanced(self, cols: List, race_id: str, race_info: Dict[str, str],
                                   title: str, date: str, place_code: str, place: str,
                                   payout_data: Dict[str, Any]) -> Dict[str, Any]:
        """拡張版: 馬のデータを抽出（辞書形式）"""
        # 体重処理
        var = cols[14].text.strip()
        try:
            weight = int(var.split("(")[0])
            weight_dif = int(var.split("(")[1].replace(")", ""))
        except (ValueError, IndexError):
            weight = weight_dif = 0
        
        # 馬番
        horse_num = cols[2].text.strip()
        
        # 調教師名
        trainer_name = cols[18].find('a').text.strip() if cols[18].find('a') else ''
        
        # 基本データ
        horse_data = {
            'race_id': race_id,
            '馬': cols[3].text.strip(),
            '騎手': cols[6].text.strip(),
            '馬番': horse_num,
            '調教師': trainer_name,
            '走破時間': cols[7].text.strip() if len(cols) > 7 else '',
            'オッズ': cols[12].text.strip(),
            '通過順': cols[10].text.strip() if len(cols) > 10 else '',
            '着順': cols[0].text.strip(),
            '体重': weight,
            '体重変化': weight_dif,
            '性': cols[4].text.strip()[0] if cols[4].text.strip() else '',
            '齢': cols[4].text.strip()[1] if len(cols[4].text.strip()) > 1 else '',
            '斤量': cols[5].text.strip(),
            '賞金': cols[20].text.strip(),
            '上がり': cols[11].text.strip() if len(cols) > 11 else '',
            '人気': cols[13].text.strip() if len(cols) > 13 else '',
            'レース名': title,
            '日付': date,
            'クラス': self._extract_class(title),
            '芝・ダート': race_info["sur"],
            '距離': race_info["dis"],
            '回り': race_info["rou"],
            '馬場': race_info["con"],
            '天気': race_info["wed"],
            '場id': place_code,
            '場名': place,
        }
        
        # 枠番を計算（馬番から）
        horse_data['枠番'] = ((int(horse_num) - 1) // 2) + 1 if horse_num.isdigit() else 0
        
        # 払戻データをレース単位で追加（個別の馬には紐付けない）
        horse_data['払戻_単勝'] = payout_data.get('win', {})
        horse_data['払戻_複勝'] = payout_data.get('place', {})
        horse_data['払戻_馬連'] = payout_data.get('quinella', {})
        horse_data['払戻_馬単'] = payout_data.get('exacta', {})
        horse_data['払戻_ワイド'] = payout_data.get('wide', {})
        horse_data['払戻_三連複'] = payout_data.get('trio', {})
        horse_data['払戻_三連単'] = payout_data.get('trifecta', {})
        
        return horse_data
    
    def _extract_class(self, title: str) -> str:
        """レースタイトルからクラスを抽出"""
        if "G1" in title or "GⅠ" in title:
            return "G1"
        elif "G2" in title or "GⅡ" in title:
            return "G2"
        elif "G3" in title or "GⅢ" in title:
            return "G3"
        elif "オープン" in title or "OP" in title:
            return "オープン"
        elif "3勝" in title or "1600万" in title:
            return "3勝"
        elif "2勝" in title or "1000万" in title:
            return "2勝"
        elif "1勝" in title or "500万" in title:
            return "1勝"
        elif "新馬" in title:
            return "新馬"
        elif "未勝利" in title:
            return "未勝利"
        else:
            return "その他"
    
    def process_race_enhanced(self, url_race_id_tuple: Tuple[str, str, str, str]) -> List[Dict[str, Any]]:
        """レースデータを処理（拡張版）"""
        url, race_id, place_code, place = url_race_id_tuple
        content = self.fetch_data(url)
        return self.parse_race_data_enhanced(race_id, content, place_code, place)
    
    def scrape_year(self, year: int) -> pd.DataFrame:
        """指定年のデータをスクレイピング（拡張版）"""
        race_data_all = []
        urls_data = []
        
        # URLリストの生成
        for place_code, place in self.PLACE_DICT.items():
            for kai in range(1, 8):  # 開催回数
                for nichi in range(1, 14):  # 開催日数
                    race_id_base = f"{year}{place_code}{kai:02d}{nichi:02d}"
                    for race_num in range(1, 13):  # レース数
                        race_id = f"{race_id_base}{race_num:02d}"
                        url = f"https://db.netkeiba.com/race/{race_id}"
                        urls_data.append((url, race_id, place_code, place))
        
        # 並列処理でデータ取得
        print(f"Year {year}: Processing {len(urls_data)} potential races...")
        with tqdm(total=len(urls_data), desc=f"Year {year}") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.process_race_enhanced, data): data for data in urls_data}
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        result = future.result()
                        if result:
                            race_data_all.extend(result)
                    except Exception as e:
                        print(f"Error processing race: {e}")
                    pbar.update(1)
                    
                    # レート制限対策
                    if pbar.n % 100 == 0:
                        time.sleep(1)
        
        # DataFrameに変換
        df = pd.DataFrame(race_data_all)
        
        # 出走頭数を追加
        if not df.empty and 'race_id' in df.columns:
            headcount_series = df.groupby('race_id')['race_id'].transform('count')
            race_id_index = df.columns.get_loc('race_id') + 1
            df.insert(race_id_index, '出走頭数', headcount_series)
        
        return df
    
    def save_data(self, df: pd.DataFrame, year: int, format: str = 'xlsx') -> str:
        """データを保存"""
        output_path = os.path.join(self.output_dir, f'{year}_enhanced.{format}')
        
        if format == 'xlsx':
            df.to_excel(output_path, index=False)
        else:
            # CSVの場合
            df.to_csv(output_path, index=False, encoding="utf-8")
        
        print(f"{year}年のデータを保存しました: {output_path}")
        print(f"  総レース数: {df['race_id'].nunique() if not df.empty else 0}")
        print(f"  総データ数: {len(df)}")
        
        return output_path
    
    def scrape_years(self, start_year: int, end_year: int, format: str = 'xlsx') -> List[str]:
        """複数年のデータをスクレイピング"""
        saved_files = []
        total_years = end_year - start_year + 1
        
        with tqdm(total=total_years, desc="Total Progress") as pbar_total:
            for year in range(start_year, end_year + 1):
                print(f"\n{'='*50}")
                print(f"Processing year {year}")
                print(f"{'='*50}")
                
                df = self.scrape_year(year)
                if not df.empty:
                    saved_path = self.save_data(df, year, format)
                    saved_files.append(saved_path)
                else:
                    print(f"Warning: No data scraped for year {year}")
                pbar_total.update(1)
        
        print("\nスクレイピング完了")
        print(f"保存されたファイル: {saved_files}")
        return saved_files


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='拡張版競馬データスクレイピング')
    parser.add_argument('--start', type=int, default=2024, help='開始年')
    parser.add_argument('--end', type=int, default=2024, help='終了年')
    parser.add_argument('--output', type=str, default='data_enhanced', help='出力ディレクトリ')
    parser.add_argument('--workers', type=int, default=3, help='並列処理のワーカー数')
    parser.add_argument('--format', type=str, default='xlsx', choices=['xlsx', 'csv'], help='出力形式')
    
    args = parser.parse_args()
    
    print("拡張版スクレイピングを開始します...")
    print(f"期間: {args.start}年 - {args.end}年")
    print(f"出力先: {args.output}")
    print(f"並列数: {args.workers}")
    
    scraper = EnhancedRaceScraper(output_dir=args.output, max_workers=args.workers)
    scraper.scrape_years(args.start, args.end, args.format)


if __name__ == "__main__":
    main()