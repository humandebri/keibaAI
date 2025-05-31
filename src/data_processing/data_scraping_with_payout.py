#!/usr/bin/env python3
"""
競馬データスクレイピングモジュール（払戻データ追加版）
元のdata_scraping.pyに払戻データ取得機能を追加
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
import re


class RaceScraperWithPayout:
    """netkeiba.comから競馬データと払戻データを取得するクラス"""
    
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
    
    def __init__(self, output_dir: str = "data", max_workers: int = 3):
        """
        Args:
            output_dir: データ保存先ディレクトリ
            max_workers: 並列処理のワーカー数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_race_data(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """URLからレースデータを取得"""
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
                wait_time = random.uniform(2, 5)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        return None
    
    def extract_payout_data(self, soup) -> Dict[str, Any]:
        """払戻データを抽出"""
        payout_data = {
            'win': {},      # 単勝
            'place': {},    # 複勝
            'quinella': {}, # 馬連
            'exacta': {},   # 馬単
            'wide': {},     # ワイド
            'trio': {},     # 三連複
            'trifecta': {}  # 三連単
        }
        
        # 払戻テーブルを探す
        payout_tables = soup.find_all("table", {"class": "pay_table_01"})
        
        # 馬券種別のマッピング（行の位置で判断）
        bet_types_table1 = ['単勝', '複勝', '枠連', '馬連']
        bet_types_table2 = ['ワイド', '馬単', '三連複', '三連単']
        
        for table_idx, table in enumerate(payout_tables):
            rows = table.find_all("tr")
            bet_types = bet_types_table1 if table_idx == 0 else bet_types_table2
            
            for row_idx, row in enumerate(rows):
                try:
                    cells = row.find_all("td")
                    if len(cells) >= 3 and row_idx < len(bet_types):
                        # 各セルのテキストを取得
                        bet_type = bet_types[row_idx]
                        combination = cells[0].text.strip()
                        payout_text = cells[1].text.strip()
                        
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
    
    def parse_race_data(self, race_id: str, content: bytes, place_code: str, place: str) -> List[List[Any]]:
        """HTMLコンテンツからレースデータを抽出（払戻データ付き）"""
        if content is None:
            return []
            
        soup = BeautifulSoup(content, "html.parser", from_encoding="euc-jp")
        soup_span = soup.find_all("span")
        main_table = soup.find("table", {"class": "race_table_01 nk_tb_common"})
        
        if not main_table:
            print(f'No data found for race_id: {race_id}')
            return []
        
        race_data = []
        
        # レース詳細情報の取得
        try:
            race_info = self._extract_race_info(soup_span)
        except Exception as e:
            print(f"Error extracting race info for {race_id}: {e}")
            return []
        
        # レースタイトルと日付情報
        soup_smalltxt = soup.find_all("p", class_="smalltxt")
        if soup_smalltxt:
            detail_text = str(soup_smalltxt).split(">")[1]
            date = detail_text.split(" ")[0]
            detail = detail_text.split(" ")[1]
            clas = detail_text.split(" ")[2].replace(u'\xa0', u' ').split(" ")[0]
        else:
            date = detail = clas = ""
        
        title = str(soup.find_all("h1")[1]).split(">")[1].split("<")[0] if len(soup.find_all("h1")) > 1 else ""
        
        # 払戻データを取得
        payout_data = self.extract_payout_data(soup)
        
        # 払戻データをJSON文字列に変換（Excelに保存するため）
        import json
        payout_json = json.dumps(payout_data, ensure_ascii=False)
        
        # 各馬のデータを抽出
        for row in main_table.find_all("tr")[1:]:  # ヘッダ行をスキップ
            cols = row.find_all("td")
            if len(cols) < 21:
                continue
                
            horse_data = self._extract_horse_data(cols, race_id, race_info, 
                                                title, date, detail, clas, 
                                                place_code, place)
            
            # 払戻データを追加（全ての馬に同じデータを付与）
            horse_data.append(payout_json)
            
            # 枠番を追加（馬番から計算）
            horse_num = cols[2].text.strip()
            if horse_num.isdigit():
                waku_ban = ((int(horse_num) - 1) // 2) + 1
            else:
                waku_ban = 0
            horse_data.append(waku_ban)
            
            race_data.append(horse_data)
        
        return race_data
    
    def _extract_race_info(self, soup_span: List) -> Dict[str, str]:
        """レース情報を抽出"""
        for idx in [8, 7, 6]:
            try:
                var = soup_span[idx]
                var_str = str(var)
                sur = var_str.split("/")[0].split(">")[1][0]
                rou = var_str.split("/")[0].split(">")[1][1]
                dis = var_str.split("/")[0].split(">")[1].split("m")[0][-4:]
                con = var_str.split("/")[2].split(":")[1][1]
                wed = var_str.split("/")[1].split(":")[1][1]
                return {"sur": sur, "rou": rou, "dis": dis, "con": con, "wed": wed}
            except IndexError:
                continue
        raise ValueError("Could not extract race info")
    
    def _extract_horse_data(self, cols: List, race_id: str, race_info: Dict[str, str],
                          title: str, date: str, detail: str, clas: str,
                          place_code: str, place: str) -> List[Any]:
        """馬のデータを抽出"""
        # 体重処理
        var = cols[14].text.strip()
        try:
            weight = int(var.split("(")[0])
            weight_dif = int(var.split("(")[1].replace(")", ""))
        except (ValueError, IndexError):
            weight = weight_dif = 0
        
        # 調教師名
        trainer_name = cols[18].find('a').text.strip() if cols[18].find('a') else ''
        
        return [
            race_id,
            cols[3].text.strip(),  # 馬の名前
            cols[6].text.strip(),  # 騎手の名前
            cols[2].text.strip(),  # 馬番
            trainer_name,  # 調教師
            cols[7].text.strip() if len(cols) > 7 else '',  # 走破時間
            cols[12].text.strip(),  # オッズ
            cols[10].text.strip() if len(cols) > 10 else '',  # 通過順
            cols[0].text.strip(),  # 着順
            weight,  # 体重
            weight_dif,  # 体重変化
            cols[4].text.strip()[0] if cols[4].text.strip() else '',  # 性
            cols[4].text.strip()[1] if len(cols[4].text.strip()) > 1 else '',  # 齢
            cols[5].text.strip(),  # 斤量
            cols[20].text.strip(),  # 賞金
            cols[11].text.strip() if len(cols) > 11 else '',  # 上がり
            cols[13].text.strip() if len(cols) > 13 else '',  # 人気
            title,  # レース名
            date,  # 日付
            detail,
            clas,  # クラス
            race_info["sur"],  # 芝・ダート
            race_info["dis"],  # 距離
            race_info["rou"],  # 回り
            race_info["con"],  # 馬場状態
            race_info["wed"],  # 天気
            place_code,  # 場id
            place,  # 場名
        ]
    
    def process_race(self, url_race_id_tuple: Tuple[str, str, str, str]) -> List[List[Any]]:
        """レースデータを処理"""
        url, race_id, place_code, place = url_race_id_tuple
        content = self.fetch_race_data(url)
        return self.parse_race_data(race_id, content, place_code, place)
    
    def scrape_year(self, year: int) -> pd.DataFrame:
        """指定年のデータをスクレイピング"""
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
        with tqdm(total=len(urls_data), desc=f"Year {year}") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {executor.submit(self.process_race, data): data for data in urls_data}
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future.result()
                    race_data_all.extend(result)
                    pbar.update(1)
        
        # DataFrameに変換
        df = pd.DataFrame(race_data_all, columns=[
            'race_id', '馬', '騎手', '馬番', '調教師', '走破時間', 'オッズ', '通過順', 
            '着順', '体重', '体重変化', '性', '齢', '斤量', '賞金', '上がり', '人気', 
            'レース名', '日付', '開催', 'クラス', '芝・ダート', '距離', '回り', '馬場', 
            '天気', '場id', '場名', '払戻データ', '枠番'
        ])
        
        # 出走頭数を追加
        if not df.empty:
            headcount_series = df.groupby('race_id')['race_id'].transform('count')
            race_id_index = df.columns.get_loc('race_id') + 1
            df.insert(race_id_index, '出走頭数', headcount_series)
        
        return df
    
    def save_data(self, df: pd.DataFrame, year: int, format: str = 'xlsx') -> str:
        """データを保存"""
        output_path = os.path.join(self.output_dir, f'{year}_with_payout.{format}')
        
        if format == 'xlsx':
            df.to_excel(output_path, index=False)
        else:
            # CSVの場合はSHIFT-JISでエンコーディング
            df = df.apply(lambda col: col.map(lambda x: x if isinstance(x, str) else str(x)).fillna(''))
            df.to_csv(output_path, index=False, encoding="SHIFT-JIS", errors="replace")
        
        print(f"{year}年のデータを保存しました: {output_path}")
        return output_path
    
    def scrape_years(self, start_year: int, end_year: int, format: str = 'xlsx') -> List[str]:
        """複数年のデータをスクレイピング"""
        saved_files = []
        total_years = end_year - start_year + 1
        
        with tqdm(total=total_years, desc="Total Progress") as pbar_total:
            for year in range(start_year, end_year + 1):
                df = self.scrape_year(year)
                if not df.empty:
                    saved_path = self.save_data(df, year, format)
                    saved_files.append(saved_path)
                else:
                    print(f"Warning: No data scraped for year {year}")
                pbar_total.update(1)
        
        print("スクレイピング完了")
        return saved_files


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='競馬データスクレイピング（払戻データ付き）')
    parser.add_argument('--start', type=int, default=2024, help='開始年')
    parser.add_argument('--end', type=int, default=2024, help='終了年')
    parser.add_argument('--output', type=str, default='data_with_payout', help='出力ディレクトリ')
    parser.add_argument('--workers', type=int, default=3, help='並列処理のワーカー数')
    parser.add_argument('--format', type=str, default='xlsx', choices=['xlsx', 'csv'], help='出力形式')
    
    args = parser.parse_args()
    
    print("払戻データ付きスクレイピングを開始します...")
    print(f"期間: {args.start}年 - {args.end}年")
    print(f"出力先: {args.output}")
    
    scraper = RaceScraperWithPayout(output_dir=args.output, max_workers=args.workers)
    scraper.scrape_years(args.start, args.end, args.format)


if __name__ == "__main__":
    main()