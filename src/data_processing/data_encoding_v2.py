#!/usr/bin/env python3
"""
競馬データエンコーディングモジュール（払戻データ対応版）
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
import json

warnings.filterwarnings('ignore')


class RaceDataEncoderV2:
    """競馬データのエンコーディングクラス（払戻データ対応版）"""
    
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
    
    def load_data(self, year_start: int, year_end: int, data_dir: str = "data_with_payout") -> pd.DataFrame:
        """指定年のデータを読み込み（払戻データ対応）"""
        dfs = []
        
        print("ファイル取得：開始")
        for year in range(year_start, year_end + 1):
            # 複数のファイル名パターンを試す
            file_patterns = [
                f"{year}_with_payout.xlsx",
                f"{year}.xlsx",
                f"{year}_enhanced.xlsx"
            ]
            
            file_loaded = False
            for file_name in file_patterns:
                file_path = os.path.join(data_dir, file_name)
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
                    
                    # 枠番が文字列の場合は数値に変換
                    if '枠番' in df.columns:
                        df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(0).astype(int)
                    
                    dfs.append(df)
                    print(f"  → {len(df)}行のデータを読み込みました")
                    file_loaded = True
                    break
            
            if not file_loaded:
                print(f"警告: {year}年のデータファイルが見つかりません")
        
        if not dfs:
            raise ValueError("データファイルが見つかりません")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"\n合計 {len(df_combined)} 行のデータを読み込みました")
        print("ファイル取得：完了")
        
        return df_combined
    
    def process_payout_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """払戻データを処理して特徴量を作成"""
        print("払戻データ処理：開始")
        
        # 払戻データがJSON形式で保存されている場合
        if '払戻データ' in df.columns:
            # 各レースの払戻情報から統計を作成
            df['単勝最高配当'] = 0.0
            df['複勝最低配当'] = 0.0
            df['複勝最高配当'] = 0.0
            df['三連単配当'] = 0.0
            
            for idx, row in df.iterrows():
                try:
                    payout_data = json.loads(row['払戻データ'])
                    
                    # 単勝の最高配当
                    if payout_data.get('win'):
                        df.at[idx, '単勝最高配当'] = max(payout_data['win'].values())
                    
                    # 複勝の最低・最高配当
                    if payout_data.get('place'):
                        place_values = list(payout_data['place'].values())
                        if place_values:
                            df.at[idx, '複勝最低配当'] = min(place_values)
                            df.at[idx, '複勝最高配当'] = max(place_values)
                    
                    # 三連単の配当
                    if payout_data.get('trifecta'):
                        trifecta_values = list(payout_data['trifecta'].values())
                        if trifecta_values:
                            df.at[idx, '三連単配当'] = max(trifecta_values)
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    # エラーの場合はデフォルト値のまま
                    pass
            
            # 元の払戻データ列は削除（JSON文字列は機械学習に使えないため）
            df = df.drop(columns=['払戻データ'])
        
        # 払戻関連の個別カラムがある場合の処理
        payout_columns = [col for col in df.columns if '払戻_' in col]
        if payout_columns:
            print(f"  払戻カラム削除: {payout_columns}")
            df = df.drop(columns=payout_columns)
        
        print("払戻データ処理：完了")
        return df
    
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
        """過去走データの特徴量を追加（枠番も含む）"""
        print("近5走取得：開始")
        
        # データをソート
        df.sort_values(by=['馬', '日付'], ascending=[True, False], inplace=True)
        
        # 枠番を特徴量に追加
        features = ['馬番', '枠番', '騎手', '斤量', 'オッズ', '体重', '体重変化', 
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
        """特徴量エンジニアリング（払戻データから派生特徴量も作成）"""
        print("日付変換と特徴量エンジニアリング：開始")
        
        df.replace('---', np.nan, inplace=True)
        
        # 数値列の型変換（エラー回避のため）
        numeric_columns = ['オッズ', '上がり', '体重', '体重変化', '斤量', '距離', '通過順']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        
        # 枠番の統計（過去5走の枠番平均）
        waku_columns = ['枠番', '枠番1', '枠番2', '枠番3', '枠番4', '枠番5']
        existing_waku_cols = [col for col in waku_columns if col in df.columns]
        if existing_waku_cols:
            df['平均枠番'] = df[existing_waku_cols].mean(axis=1)
        
        # 斤量関連
        kinryo_columns = ['斤量', '斤量1', '斤量2', '斤量3', '斤量4', '斤量5']
        existing_kinryo_cols = [col for col in kinryo_columns if col in df.columns]
        df[existing_kinryo_cols] = df[existing_kinryo_cols].apply(pd.to_numeric, errors='coerce')
        df['平均斤量'] = df[existing_kinryo_cols].mean(axis=1)
        
        # 騎手統計（拡張版）- ラベルエンコーディング前に追加
        if '騎手' in df.columns and '着順' in df.columns:
            print("  騎手統計を計算中...")
            
            # 基本統計
            jockey_stats = df.groupby('騎手').agg({
                '着順': [
                    lambda x: (x == 1).sum() / len(x),  # 勝率
                    lambda x: (x <= 3).sum() / len(x),  # 複勝率
                    'count',  # 騎乗数
                    'mean',   # 平均着順
                    'min'     # 最高着順
                ]
            }).round(3)
            jockey_stats.columns = ['騎手の勝率', '騎手の複勝率', '騎手の騎乗数', '騎手の平均着順', '騎手の最高着順']
            
            # オッズ情報があればROIも計算
            if 'オッズ' in df.columns:
                jockey_roi = df[df['着順'] == 1].groupby('騎手')['オッズ'].mean()
                jockey_stats['騎手のROI'] = jockey_stats['騎手の勝率'] * jockey_roi
                jockey_stats['騎手のROI'] = jockey_stats['騎手のROI'].fillna(1.0)
            else:
                jockey_stats['騎手のROI'] = 1.0
            
            # 騎乗数を対数変換
            jockey_stats['騎手の騎乗数'] = np.log1p(jockey_stats['騎手の騎乗数'])
            
            # データフレームに結合
            jockey_stats.reset_index(inplace=True)
            df = pd.merge(df, jockey_stats, on='騎手', how='left')
            
            # 時系列統計（30日、60日）
            if '日付' in df.columns:
                latest_date = df['日付'].max()
                
                for window_days in [30, 60]:
                    cutoff_date = latest_date - pd.Timedelta(days=window_days)
                    recent_df = df[df['日付'] >= cutoff_date]
                    
                    recent_stats = recent_df.groupby('騎手')['着順'].agg([
                        lambda x: (x == 1).sum() / len(x),  # 勝率
                        lambda x: (x <= 3).sum() / len(x)   # 複勝率
                    ]).round(3)
                    recent_stats.columns = [f'騎手の勝率_{window_days}日', f'騎手の複勝率_{window_days}日']
                    recent_stats.reset_index(inplace=True)
                    
                    df = pd.merge(df, recent_stats, on='騎手', how='left')
                    # 欠損値はデフォルト値で埋める
                    df[f'騎手の勝率_{window_days}日'] = df[f'騎手の勝率_{window_days}日'].fillna(0.08)
                    df[f'騎手の複勝率_{window_days}日'] = df[f'騎手の複勝率_{window_days}日'].fillna(0.25)
                
                # 連続不勝・最後の勝利からの日数
                streak_stats = {}
                for jockey in df['騎手'].unique():
                    jockey_df = df[df['騎手'] == jockey].sort_values('日付', ascending=False)
                    
                    # 連続不勝
                    cold_streak = 0
                    for _, row in jockey_df.iterrows():
                        if row['着順'] == 1:
                            break
                        cold_streak += 1
                    
                    # 最後の勝利からの日数
                    win_dates = jockey_df[jockey_df['着順'] == 1]['日付']
                    if len(win_dates) > 0:
                        last_win_days = (latest_date - win_dates.iloc[0]).days
                    else:
                        last_win_days = 365
                    
                    streak_stats[jockey] = {
                        '騎手の連続不勝': cold_streak,
                        '騎手の最後勝利日数': np.exp(-last_win_days / 30)  # 指数減衰
                    }
                
                streak_df = pd.DataFrame.from_dict(streak_stats, orient='index').reset_index()
                streak_df.columns = ['騎手', '騎手の連続不勝', '騎手の最後勝利日数']
                df = pd.merge(df, streak_df, on='騎手', how='left')
                df['騎手の連続不勝'] = df['騎手の連続不勝'].fillna(0)
                df['騎手の最後勝利日数'] = df['騎手の最後勝利日数'].fillna(np.exp(-30/30))
            
            # コンテキスト統計（芝/ダート、距離）
            if '芝・ダート' in df.columns:
                for surface in ['芝', 'ダ']:
                    surface_df = df[df['芝・ダート'] == surface]
                    surface_stats = surface_df.groupby('騎手')['着順'].apply(
                        lambda x: (x == 1).sum() / len(x)
                    ).reset_index()
                    surface_stats.columns = ['騎手', f'騎手の勝率_{surface}']
                    df = pd.merge(df, surface_stats, on='騎手', how='left')
                    df[f'騎手の勝率_{surface}'] = df[f'騎手の勝率_{surface}'].fillna(0.08)
            
            if '距離' in df.columns:
                # 距離カテゴリ別
                df['距離カテゴリ'] = pd.cut(df['距離'], 
                                        bins=[0, 1400, 1800, 2200, 4000], 
                                        labels=['短距離', '中距離', '中長距離', '長距離'])
                
                for dist_cat in ['短距離', '中距離', '長距離']:
                    dist_df = df[df['距離カテゴリ'].astype(str) == dist_cat]
                    if len(dist_df) > 0:
                        dist_stats = dist_df.groupby('騎手')['着順'].apply(
                            lambda x: (x == 1).sum() / len(x)
                        ).reset_index()
                        dist_stats.columns = ['騎手', f'騎手の勝率_{dist_cat}']
                        df = pd.merge(df, dist_stats, on='騎手', how='left')
                        df[f'騎手の勝率_{dist_cat}'] = df[f'騎手の勝率_{dist_cat}'].fillna(0.08)
                
                # 距離カテゴリ列は削除
                df = df.drop(columns=['距離カテゴリ'])
            
            # 騎手×調教師の相性
            if '調教師' in df.columns:
                synergy_stats = df.groupby(['騎手', '調教師'])['着順'].agg([
                    lambda x: (x == 1).sum() / len(x) if len(x) >= 3 else np.nan
                ]).reset_index()
                synergy_stats.columns = ['騎手', '調教師', '騎手調教師相性']
                df = pd.merge(df, synergy_stats, on=['騎手', '調教師'], how='left')
                df['騎手調教師相性'] = df['騎手調教師相性'].fillna(0.08)
            
            # 騎手統計の保存（デバッグ用）
            os.makedirs('calc_rate', exist_ok=True)
            jockey_stats.to_excel(
                os.path.join('calc_rate', 'jockey_stats_extended.xlsx'), 
                index=False
            )
        
        # 調教師統計
        if '調教師' in df.columns and '着順' in df.columns:
            trainer_stats = df.groupby('調教師').agg({
                '着順': [
                    lambda x: (x == 1).sum() / len(x),  # 勝率
                    lambda x: (x <= 3).sum() / len(x)   # 複勝率
                ]
            }).round(3)
            trainer_stats.columns = ['調教師の勝率', '調教師の複勝率']
            trainer_stats.reset_index(inplace=True)
            df = pd.merge(df, trainer_stats, on='調教師', how='left')
            df['調教師の勝率'] = df['調教師の勝率'].fillna(0.08)
            df['調教師の複勝率'] = df['調教師の複勝率'].fillna(0.25)
        
        # 馬の過去成績統計（中間日数など）
        if '馬' in df.columns:
            print("  馬の過去成績統計を計算中...")
            
            # 過去平均着順
            horse_stats = df.groupby('馬')['着順'].agg(['mean', 'min', 'count']).reset_index()
            horse_stats.columns = ['馬', '過去平均着順', '過去最高着順', '過去レース数']
            
            # 勝利・複勝経験
            win_stats = df.groupby('馬')['着順'].apply(lambda x: (x == 1).sum()).reset_index()
            win_stats.columns = ['馬', '勝利経験']
            horse_stats = pd.merge(horse_stats, win_stats, on='馬')
            
            place_stats = df.groupby('馬')['着順'].apply(lambda x: (x <= 3).sum()).reset_index()
            place_stats.columns = ['馬', '複勝経験']
            horse_stats = pd.merge(horse_stats, place_stats, on='馬')
            
            df = pd.merge(df, horse_stats, on='馬', how='left')
            
            # 中間日数（前走からの日数）
            if '日付' in df.columns and '日付1' in df.columns:
                df['前走からの日数'] = (df['日付'] - df['日付1']).dt.days
                df['前走からの日数'] = df['前走からの日数'].fillna(180)  # デフォルト値
                
                # 放牧区分
                df['放牧区分'] = pd.cut(df['前走からの日数'], 
                                     bins=[-1, 14, 28, 56, 84, 365, 9999],
                                     labels=[0, 1, 2, 3, 4, 5])
                df['放牧区分'] = df['放牧区分'].astype(int)
                
                # 平均中間日数
                interval_cols = []
                for i in range(1, 4):
                    if f'日付差{i}' in df.columns:
                        interval_cols.append(f'日付差{i}')
                        df[f'中間日数{i}'] = df[f'日付差{i}']
                
                if interval_cols:
                    df['平均中間日数'] = df[interval_cols].mean(axis=1)
                    df['中間日数標準偏差'] = df[interval_cols].std(axis=1).fillna(0)
                else:
                    df['平均中間日数'] = 30
                    df['中間日数標準偏差'] = 0
        
        # 出走頭数（既に含まれている場合はスキップ）
        if 'race_id' in df.columns and '出走頭数' not in df.columns:
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
        
        # 払戻データから作成した特徴量の処理
        if '単勝最高配当' in df.columns:
            # レースの配当レベル（高配当レースかどうか）
            df['高配当レース'] = (df['単勝最高配当'] > 1000).astype(int)
            df['三連単高配当'] = (df['三連単配当'] > 10000).astype(int)
        
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
    
    def encode_data(self, year_start: int, year_end: int, data_dir: str = "data_with_payout") -> str:
        """データのエンコーディングを実行（払戻データ対応版）"""
        # データ読み込み
        df = self.load_data(year_start, year_end, data_dir)
        
        # 払戻データの処理
        df = self.process_payout_data(df)
        
        # 各種変換処理
        df = self.transform_data(df)
        df = self.add_historical_features(df)
        df = self.engineer_features(df, year_start)
        df = self.finalize_encoding(df, year_start)
        
        # 保存
        output_path = os.path.join(self.encoded_dir, f'{year_start}_{year_end}encoded_data_v2.csv')
        df.to_csv(output_path, index=False)
        
        print(f"ファイル出力：完了")
        print(f"出力ファイル: {output_path}")
        print(f"データ件数: {len(df)}行")
        print(f"カラム数: {len(df.columns)}列")
        
        # 新しい特徴量の確認
        new_features = ['枠番', '平均枠番', '単勝最高配当', '複勝最低配当', 
                       '複勝最高配当', '三連単配当', '高配当レース', '三連単高配当']
        existing_new_features = [f for f in new_features if f in df.columns]
        if existing_new_features:
            print(f"\n新しい特徴量: {existing_new_features}")
        
        return output_path


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='競馬データエンコーディング（払戻データ対応版）')
    parser.add_argument('--start', type=int, default=2014, help='開始年')
    parser.add_argument('--end', type=int, default=2025, help='終了年')
    parser.add_argument('--data_dir', type=str, default='data_with_payout', help='データディレクトリ')
    parser.add_argument('--encoded_dir', type=str, default='encoded', help='エンコード済みデータ出力ディレクトリ')
    parser.add_argument('--config_dir', type=str, default='config', help='設定ファイル出力ディレクトリ')
    
    args = parser.parse_args()
    
    encoder = RaceDataEncoderV2(config_dir=args.config_dir, encoded_dir=args.encoded_dir)
    encoder.encode_data(args.start, args.end, args.data_dir)


if __name__ == "__main__":
    main()