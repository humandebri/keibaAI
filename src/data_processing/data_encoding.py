#!/usr/bin/env python3
"""
競馬データエンコーディングモジュール
スクレイピングしたデータを機械学習用に前処理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from typing import List, Tuple, Dict, Any
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')


class RaceDataEncoder:
    """競馬データのエンコーディングクラス"""
    
    def __init__(self, config_dir: str = "config", encoded_dir: str = "encoded"):
        """
        Args:
            config_dir: 設定ファイル保存ディレクトリ
            encoded_dir: エンコード済みデータ保存ディレクトリ
        """
        self.config_dir = config_dir
        self.encoded_dir = encoded_dir
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(encoded_dir, exist_ok=True)
        
        # マッピング定義
        self.class_mappings = {
            '障害': 0, 'G1': 10, 'G2': 9, 'G3': 8, '(L)': 7, 
            'オープン': 7, 'OP': 7, '3勝': 6, '1600': 6, 
            '2勝': 5, '1000': 5, '1勝': 4, '500': 4, 
            '新馬': 3, '未勝利': 1
        }
        
        self.categorical_mappings = {
            '性': {'牡': 0, '牝': 1, 'セ': 2},
            '芝・ダート': {'芝': 0, 'ダ': 1, '障': 2},
            '回り': {'右': 0, '左': 1, '芝': 2, '直': 2},
            '馬場': {'良': 0, '稍': 1, '重': 2, '不': 3},
            '天気': {'晴': 0, '曇': 1, '小': 2, '雨': 3, '雪': 4}
        }
    
    def load_data(self, year_start: int, year_end: int, data_dir: str = "data") -> pd.DataFrame:
        """指定年のデータを読み込み"""
        dfs = []
        
        print("ファイル取得：開始")
        for year in range(year_start, year_end + 1):
            file_path = os.path.join(data_dir, f"{year}.xlsx")
            if os.path.exists(file_path):
                print(f"読み込み中: {file_path}")
                df = pd.read_excel(file_path, header=0)
                
                # 日付処理
                if '日付' in df.columns:
                    df['日付'] = self._parse_date(df['日付'])
                
                # 着順処理
                df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                df = df.dropna(subset=['着順'])
                df['着順'] = df['着順'].astype(int)
                
                # 賞金処理
                if '賞金' in df.columns:
                    df['賞金'] = df['賞金'].astype(str).str.replace(',', '')
                    df['賞金'] = pd.to_numeric(df['賞金'], errors='coerce').fillna(0)
                
                dfs.append(df)
                print(f"  → {len(df)}行のデータを読み込みました")
            else:
                print(f"警告: {file_path} が見つかりません")
        
        if not dfs:
            raise ValueError("データファイルが見つかりません")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"\n合計 {len(df_combined)} 行のデータを読み込みました")
        print("ファイル取得：完了")
        
        return df_combined
    
    def _parse_date(self, date_series: pd.Series) -> pd.Series:
        """日付のパース"""
        if pd.api.types.is_datetime64_any_dtype(date_series):
            return date_series
        
        # 複数の日付フォーマットを試す
        date_formats = ['%Y年%m月%d日', '%Y/%m/%d', '%Y-%m-%d']
        for date_format in date_formats:
            try:
                return pd.to_datetime(date_series, format=date_format)
            except:
                continue
        
        # 自動推定
        return pd.to_datetime(date_series, errors='coerce')
    
    def _class_mapping(self, row: str) -> int:
        """クラス名を数値にマッピング"""
        for key, value in self.class_mappings.items():
            if key in str(row):
                return value
        return 0
    
    def _standardize_times(self, df: pd.DataFrame, col_name: str) -> Tuple[float, float, float, float]:
        """走破時間を標準化"""
        # 時間を秒に変換
        time_parts = df[col_name].str.split(':', expand=True)
        seconds = (time_parts[0].astype(float) * 60 + 
                  time_parts[1].str.split('.', expand=True)[0].astype(float) + 
                  time_parts[1].str.split('.', expand=True)[1].astype(float) / 10)
        seconds = seconds.ffill()
        
        # 1回目の標準化
        mean_seconds = seconds.mean()
        std_seconds = seconds.std()
        df[col_name] = -((seconds - mean_seconds) / std_seconds)
        
        # 外れ値処理
        df[col_name] = df[col_name].apply(lambda x: -3 if x < -3 else (2 if x > 2.5 else x))
        
        # 2回目の標準化
        mean_seconds_2 = df[col_name].mean()
        std_seconds_2 = df[col_name].std()
        df[col_name] = (df[col_name] - mean_seconds_2) / std_seconds_2
        
        return mean_seconds, std_seconds, mean_seconds_2, std_seconds_2
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの変換処理"""
        print("データ変換：開始")
        
        # NaN行の削除
        df = df.dropna(subset=['走破時間'])
        
        # 走破時間の標準化
        mean_s, std_s, mean_s2, std_s2 = self._standardize_times(df, '走破時間')
        print(f'1回目平均: {mean_s}')
        print(f'2回目平均: {mean_s2}')
        print(f'1回目標準偏差: {std_s}')
        print(f'2回目標準偏差: {std_s2}')
        
        # 標準化情報を保存
        time_df = pd.DataFrame({
            'Mean': [mean_s, mean_s2],
            'Standard Deviation': [std_s, std_s2]
        }, index=['First Time', 'Second Time'])
        time_df.to_excel(os.path.join(self.config_dir, 'standard_deviation.xlsx'))
        
        # 通過順の平均計算
        if '通過順' in df.columns:
            pas = df['通過順'].str.split('-', expand=True)
            df['通過順'] = pas.astype(float).mean(axis=1)
        
        # カテゴリカル変数のマッピング
        for column, mapping in self.categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
        
        # クラス変換
        if 'クラス' in df.columns:
            df['クラス'] = df['クラス'].apply(self._class_mapping)
        
        print("データ変換：完了")
        return df
    
    def add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去走データの特徴量を追加"""
        print("近5走取得：開始")
        
        # データをソート
        df.sort_values(by=['馬', '日付'], ascending=[True, False], inplace=True)
        
        features = ['馬番', '騎手', '斤量', 'オッズ', '体重', '体重変化', 
                   '上がり', '通過順', '着順', '距離', 'クラス', '走破時間', 
                   '芝・ダート', '天気', '馬場']
        
        # 過去5走の情報を取得
        shifts = {}
        for i in range(1, 6):
            shifts[f'日付{i}'] = df.groupby('馬')['日付'].shift(-i)
            for feature in features:
                if feature in df.columns:
                    shifts[f'{feature}{i}'] = df.groupby('馬')[feature].shift(-i).ffill()
        
        # 新しい列を追加
        df = pd.concat([df, pd.DataFrame(shifts)], axis=1)
        
        # 最新データのみ取得
        df = df.groupby(['race_id', '馬'], as_index=False).last()
        df.sort_values(by='race_id', ascending=False, inplace=True)
        
        print("近5走取得：終了")
        return df
    
    def engineer_features(self, df: pd.DataFrame, year_start: int) -> pd.DataFrame:
        """特徴量エンジニアリング"""
        print("日付変換と特徴量エンジニアリング：開始")
        
        df.replace('---', np.nan, inplace=True)
        
        # 距離差と日付差
        if '距離' in df.columns and '距離1' in df.columns:
            df['距離差'] = df['距離'] - df['距離1']
        
        if '日付' in df.columns and '日付1' in df.columns:
            df['日付差'] = (df['日付'] - df['日付1']).dt.days
        
        for i in range(1, 5):
            if f'距離{i}' in df.columns and f'距離{i+1}' in df.columns:
                df[f'距離差{i}'] = df[f'距離{i}'] - df[f'距離{i+1}']
            if f'日付{i}' in df.columns and f'日付{i+1}' in df.columns:
                df[f'日付差{i}'] = (df[f'日付{i}'] - df[f'日付{i+1}']).dt.days
        
        # 斤量関連
        kinryo_columns = ['斤量', '斤量1', '斤量2', '斤量3', '斤量4', '斤量5']
        existing_kinryo_cols = [col for col in kinryo_columns if col in df.columns]
        df[existing_kinryo_cols] = df[existing_kinryo_cols].apply(pd.to_numeric, errors='coerce')
        df['平均斤量'] = df[existing_kinryo_cols].mean(axis=1)
        
        # 騎手の勝率
        if '騎手' in df.columns and '着順' in df.columns:
            jockey_win_rate = df.groupby('騎手')['着順'].apply(
                lambda x: (x == 1).sum() / x.count()
            ).reset_index()
            jockey_win_rate.columns = ['騎手', '騎手の勝率']
            
            os.makedirs('calc_rate', exist_ok=True)
            jockey_win_rate.to_excel(
                os.path.join('calc_rate', 'jockey_win_rate.xlsx'), 
                index=False
            )
            df = pd.merge(df, jockey_win_rate, on='騎手', how='left')
        
        # 出走頭数
        if 'race_id' in df.columns:
            df['出走頭数'] = df.groupby('race_id')['race_id'].transform('count')
        
        for i in range(1, 6):
            df[f'出走頭数{i}'] = df.groupby('馬')['出走頭数'].shift(i).fillna(0)
        
        # スピード計算
        speed_cols = []
        for i in range(1, 6):
            if f'距離{i}' in df.columns and f'走破時間{i}' in df.columns:
                df[f'スピード{i}'] = df[f'距離{i}'] / df[f'走破時間{i}']
                speed_cols.append(f'スピード{i}')
        
        if speed_cols:
            df['平均スピード'] = df[speed_cols].mean(axis=1, skipna=True)
            df.drop(columns=speed_cols, inplace=True)
        
        # 賞金合計
        if '賞金' in df.columns:
            for i in range(1, 6):
                df[f'賞金{i}'] = df.groupby('馬')['賞金'].shift(i)
            df['過去5走の合計賞金'] = df[[f'賞金{i}' for i in range(1, 6)]].sum(axis=1)
            df.drop(columns=[f'賞金{i}' for i in range(1, 6)] + ['賞金'], inplace=True)
        
        print("日付変換と特徴量エンジニアリング：完了")
        return df
    
    def finalize_encoding(self, df: pd.DataFrame, year_start: int) -> pd.DataFrame:
        """最終的なエンコーディング処理"""
        print("最終処理：開始")
        
        df.sort_values(by='race_id', ascending=False, inplace=True)
        
        # 季節特徴量を追加
        date_columns = ['日付1', '日付2', '日付3', '日付4', '日付5']
        existing_date_cols = [col for col in date_columns if col in df.columns]
        if existing_date_cols:
            self._add_seasonal_features(df, existing_date_cols)
        
        # 日付を数値に変換
        date_columns = ['日付', '日付1', '日付2', '日付3', '日付4', '日付5']
        for col in date_columns:
            if col in df.columns and not df[col].isna().all():
                year = df[col].dt.year
                month = df[col].dt.month
                day = df[col].dt.day
                df[col] = (year - year_start) * 365 + month * 30 + day
        
        # 騎手の乗り替わり
        if '騎手' in df.columns:
            df['騎手の乗り替わり'] = df.groupby('馬')['騎手'].transform(
                lambda x: (x != x.shift()).astype(int)
            )
        
        # カテゴリカル変数のラベルエンコーディング
        categorical_features = ['馬', '騎手', '調教師', 'レース名', '開催', '場名', 
                              '騎手1', '騎手2', '騎手3', '騎手4', '騎手5']
        
        for i, feature in enumerate(categorical_features):
            if feature in df.columns:
                print(f"\rProcessing feature {i+1}/{len(categorical_features)}: {feature}", end="")
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
        
        print("\n最終処理：完了")
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame, date_columns: List[str]):
        """季節特徴量を追加"""
        for date_col in date_columns:
            if not np.issubdtype(df[date_col].dtype, np.datetime64):
                df[date_col] = pd.to_datetime(df[date_col])
            df[f'{date_col}_sin'] = np.sin((df[date_col].dt.month - 1) * (2 * np.pi / 12))
            df[f'{date_col}_cos'] = np.cos((df[date_col].dt.month - 1) * (2 * np.pi / 12))
    
    def encode_data(self, year_start: int, year_end: int, data_dir: str = "data") -> str:
        """データのエンコーディングを実行"""
        # データ読み込み
        df = self.load_data(year_start, year_end, data_dir)
        
        # 各種変換処理
        df = self.transform_data(df)
        df = self.add_historical_features(df)
        df = self.engineer_features(df, year_start)
        df = self.finalize_encoding(df, year_start)
        
        # 保存
        output_path = os.path.join(self.encoded_dir, f'{year_start}_{year_end}encoded_data.csv')
        df.to_csv(output_path, index=False)
        
        print(f"ファイル出力：完了")
        print(f"出力ファイル: {output_path}")
        print(f"データ件数: {len(df)}行")
        print(f"カラム数: {len(df.columns)}列")
        
        return output_path


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='競馬データエンコーディング')
    parser.add_argument('--start', type=int, default=2014, help='開始年')
    parser.add_argument('--end', type=int, default=2025, help='終了年')
    parser.add_argument('--data_dir', type=str, default='data', help='データディレクトリ')
    parser.add_argument('--encoded_dir', type=str, default='encoded', help='エンコード済みデータ出力ディレクトリ')
    parser.add_argument('--config_dir', type=str, default='config', help='設定ファイル出力ディレクトリ')
    
    args = parser.parse_args()
    
    encoder = RaceDataEncoder(config_dir=args.config_dir, encoded_dir=args.encoded_dir)
    encoder.encode_data(args.start, args.end, args.data_dir)


if __name__ == "__main__":
    main()