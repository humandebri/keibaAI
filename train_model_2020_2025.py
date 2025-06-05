#!/usr/bin/env python3
"""
2020-2025年のデータを使用してモデルを訓練するスクリプト
詳細な特徴量（騎手統計、中間日数等）を使用した改良版
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import warnings
from datetime import datetime
from pathlib import Path
import pickle
from typing import Dict, Any, List

warnings.filterwarnings('ignore')

# 日本語フォント設定
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


class HorseDatabase:
    """馬の過去成績データベース（clean_full_data_ml_system.pyから移植）"""
    
    def __init__(self):
        self.db_file = "cache/horse_database.pkl"
        self.horse_data = {}
        self.jockey_stats = {}
        self.jockey_context_stats = {}
        self.jockey_time_stats = {}
        self.jockey_synergy_stats = {}
        self.trainer_stats = {}
        
    def build_database(self, years: List[int] = [2020, 2021, 2022, 2023, 2024, 2025]):
        """データベース構築（キャッシュ付き）"""
        if os.path.exists(self.db_file):
            print("   📂 キャッシュからデータベース読み込み中...")
            try:
                with open(self.db_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('years', []) == years and cache_data.get('version', 1) >= 2:
                        self.horse_data = cache_data['horse_data']
                        self.jockey_stats = cache_data['jockey_stats']
                        self.jockey_context_stats = cache_data.get('jockey_context_stats', {})
                        self.jockey_time_stats = cache_data.get('jockey_time_stats', {})
                        self.jockey_synergy_stats = cache_data.get('jockey_synergy_stats', {})
                        self.trainer_stats = cache_data['trainer_stats']
                        print(f"   ✅ {len(self.horse_data)}頭のデータを読み込み")
                        return
            except Exception as e:
                print(f"   ⚠️ キャッシュ読み込みエラー: {e}")
        
        print("   🔨 データベース構築中...")
        all_data = []
        
        for year in years:
            file_patterns = [
                (f'data_with_payout/{year}_with_payout.xlsx', 'payout'),
                (f'data/{year}.xlsx', 'regular')
            ]
            
            for file_path, file_type in file_patterns:
                if os.path.exists(file_path):
                    try:
                        print(f"   読み込み中: {file_path}")
                        df = pd.read_excel(file_path)
                        all_data.append(df)
                        print(f"   ✅ {year}年: {len(df)}件 ({file_type})")
                        break
                    except Exception as e:
                        print(f"   ⚠️ {year}年読み込みエラー: {e}")
        
        if not all_data:
            print("   ⚠️ データが見つかりません")
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if '着順' in combined_df.columns:
            combined_df['着順'] = pd.to_numeric(combined_df['着順'], errors='coerce')
            combined_df = combined_df.dropna(subset=['着順'])
        
        if '日付' in combined_df.columns:
            combined_df['日付'] = pd.to_datetime(combined_df['日付'], errors='coerce')
        
        self._calculate_jockey_stats(combined_df)
        self._calculate_jockey_context_stats(combined_df)
        self._calculate_jockey_time_stats(combined_df)
        self._calculate_jockey_synergy_stats(combined_df)
        
        if '調教師' in combined_df.columns and '着順' in combined_df.columns:
            trainer_group = combined_df.groupby('調教師')['着順']
            self.trainer_stats = {
                trainer: {
                    'win_rate': (group == 1).sum() / len(group) if len(group) > 0 else 0.08,
                    'place_rate': (group <= 3).sum() / len(group) if len(group) > 0 else 0.25,
                    'count': len(group)
                }
                for trainer, group in trainer_group
            }
        
        print("   🐴 馬ごとの詳細成績を集計中...")
        for horse_name in combined_df['馬'].unique():
            if pd.notna(horse_name):
                horse_races = combined_df[combined_df['馬'] == horse_name].copy()
                if '日付' in horse_races.columns:
                    horse_races['日付'] = pd.to_datetime(horse_races['日付'], errors='coerce')
                    horse_races = horse_races.sort_values('日付', ascending=False).head(20)
                
                recent_dates = horse_races['日付'].dropna().tolist()
                days_between_races = []
                if len(recent_dates) >= 2:
                    for i in range(len(recent_dates) - 1):
                        days_diff = (recent_dates[i] - recent_dates[i+1]).days
                        days_between_races.append(days_diff)
                
                self.horse_data[horse_name] = {
                    'recent_positions': horse_races['着順'].tolist()[:10],
                    'recent_dates': recent_dates[:10],
                    'days_between_races': days_between_races[:9],
                    'avg_position': horse_races['着順'].mean(),
                    'best_position': horse_races['着順'].min(),
                    'win_count': (horse_races['着順'] == 1).sum(),
                    'place_count': (horse_races['着順'] <= 3).sum(),
                    'race_count': len(horse_races)
                }
        
        os.makedirs('cache', exist_ok=True)
        cache_data = {
            'horse_data': self.horse_data,
            'jockey_stats': self.jockey_stats,
            'jockey_context_stats': self.jockey_context_stats,
            'jockey_time_stats': self.jockey_time_stats,
            'jockey_synergy_stats': self.jockey_synergy_stats,
            'trainer_stats': self.trainer_stats,
            'years': years,
            'version': 2
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"   ✅ データベース構築完了: {len(self.horse_data)}頭")
    
    def get_horse_features(self, horse_name: str, current_date=None) -> Dict[str, Any]:
        """馬の特徴量を取得"""
        if horse_name not in self.horse_data:
            return {}
        
        data = self.horse_data[horse_name]
        features = {
            '過去平均着順': data['avg_position'],
            '過去最高着順': data['best_position'],
            '勝利経験': data['win_count'],
            '複勝経験': data['place_count'],
            '過去レース数': data['race_count']
        }
        
        for i, pos in enumerate(data['recent_positions'][:5], 1):
            features[f'前{i}走着順'] = pos
        
        if current_date and data.get('recent_dates'):
            recent_dates = data['recent_dates']
            if recent_dates:
                last_race_date = recent_dates[0]
                if pd.notna(last_race_date):
                    days_since_last = (current_date - last_race_date).days
                    features['前走からの日数'] = days_since_last
                    
                    if days_since_last <= 14:
                        features['放牧区分'] = 0
                    elif days_since_last <= 28:
                        features['放牧区分'] = 1
                    elif days_since_last <= 56:
                        features['放牧区分'] = 2
                    elif days_since_last <= 84:
                        features['放牧区分'] = 3
                    else:
                        features['放牧区分'] = 4
                else:
                    features['前走からの日数'] = 60
                    features['放牧区分'] = 3
            else:
                features['前走からの日数'] = 180
                features['放牧区分'] = 5
        
        if data.get('days_between_races'):
            days_between = data['days_between_races']
            if days_between:
                features['平均中間日数'] = np.mean(days_between)
                features['中間日数標準偏差'] = np.std(days_between) if len(days_between) > 1 else 0
                for i, days in enumerate(days_between[:3], 1):
                    features[f'中間日数{i}'] = days
            else:
                features['平均中間日数'] = 30
                features['中間日数標準偏差'] = 0
        
        return features
    
    def get_jockey_stats(self, jockey_name: str) -> Dict[str, float]:
        """騎手統計を取得"""
        if jockey_name in self.jockey_stats:
            return self.jockey_stats[jockey_name]
        return {
            'win_rate': 0.08, 
            'place_rate': 0.25, 
            'count': 100,
            'avg_position': 8.0,
            'best_position': 3,
            'roi': 1.0
        }
    
    def get_jockey_context_stats(self, jockey_name: str, context_type: str, context_value: Any) -> Dict[str, float]:
        """コンテキスト別騎手統計を取得"""
        key = f"{jockey_name}_{context_value}"
        if context_type in self.jockey_context_stats and key in self.jockey_context_stats[context_type]:
            return self.jockey_context_stats[context_type][key]
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_jockey_time_stats(self, jockey_name: str, window: int = 30) -> Dict[str, float]:
        """時系列騎手統計を取得"""
        key = f"{jockey_name}_{window}d"
        if key in self.jockey_time_stats:
            return self.jockey_time_stats[key]
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_jockey_streak_stats(self, jockey_name: str) -> Dict[str, float]:
        """騎手の連続成績統計を取得"""
        key = f"{jockey_name}_streak"
        if key in self.jockey_time_stats:
            return self.jockey_time_stats[key]
        return {
            'cold_streak': 0,
            'last_win_days': 30
        }
    
    def get_jockey_synergy_stats(self, jockey_name: str, trainer_name: str) -> Dict[str, float]:
        """騎手×調教師のシナジー統計を取得"""
        key = f"{jockey_name}_{trainer_name}"
        if key in self.jockey_synergy_stats:
            return self.jockey_synergy_stats[key]
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_trainer_stats(self, trainer_name: str) -> Dict[str, float]:
        """調教師統計を取得"""
        if trainer_name in self.trainer_stats:
            return self.trainer_stats[trainer_name]
        return {'win_rate': 0.08, 'place_rate': 0.25, 'count': 50}
    
    def _calculate_jockey_stats(self, df: pd.DataFrame):
        """基本騎手統計の計算"""
        if '騎手' not in df.columns or '着順' not in df.columns:
            return
        
        print("   🏇 騎手統計計算中...")
        
        has_odds = 'オッズ' in df.columns
        
        for jockey, group in df.groupby('騎手'):
            stats = {
                'win_rate': (group['着順'] == 1).mean(),
                'place_rate': (group['着順'] <= 3).mean(),
                'count': len(group),
                'avg_position': group['着順'].mean(),
                'best_position': group['着順'].min()
            }
            
            if has_odds:
                win_rows = group[group['着順'] == 1]
                if len(win_rows) > 0:
                    odds_numeric = pd.to_numeric(win_rows['オッズ'], errors='coerce')
                    odds_numeric = odds_numeric.dropna()
                    if len(odds_numeric) > 0:
                        stats['roi'] = (odds_numeric.mean() * stats['win_rate'])
                    else:
                        stats['roi'] = 1.0
                else:
                    stats['roi'] = 0.8
            else:
                stats['roi'] = 1.0
            
            self.jockey_stats[jockey] = stats
        
        print(f"      ✅ {len(self.jockey_stats)}人の騎手統計完了")
    
    def _calculate_jockey_context_stats(self, df: pd.DataFrame):
        """コンテキスト別騎手統計の計算"""
        if '騎手' not in df.columns:
            return
        
        print("   📊 コンテキスト別騎手統計計算中...")
        self.jockey_context_stats = {
            'course': {},
            'distance': {},
            'surface': {},
            'condition': {}
        }
        
        if '場id' in df.columns:
            for (jockey, course), group in df.groupby(['騎手', '場id']):
                key = f"{jockey}_{course}"
                self.jockey_context_stats['course'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        if '距離' in df.columns:
            df['距離カテゴリ'] = pd.cut(df['距離'], 
                                    bins=[0, 1400, 1800, 2200, 4000], 
                                    labels=['短距離', '中距離', '中長距離', '長距離'])
            
            for (jockey, dist_cat), group in df.groupby(['騎手', '距離カテゴリ']):
                key = f"{jockey}_{dist_cat}"
                self.jockey_context_stats['distance'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        if '芝・ダート' in df.columns:
            for (jockey, surface), group in df.groupby(['騎手', '芝・ダート']):
                key = f"{jockey}_{surface}"
                self.jockey_context_stats['surface'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
    
    def _calculate_jockey_time_stats(self, df: pd.DataFrame):
        """時系列騎手統計の計算"""
        if '騎手' not in df.columns or '日付' not in df.columns:
            return
        
        print("   📅 時系列騎手統計計算中...")
        
        latest_date = df['日付'].max()
        
        for window_days in [30, 60]:
            cutoff_date = latest_date - pd.Timedelta(days=window_days)
            recent_df = df[df['日付'] >= cutoff_date]
            
            for jockey, group in recent_df.groupby('騎手'):
                key = f"{jockey}_{window_days}d"
                self.jockey_time_stats[key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        for jockey, group in df.groupby('騎手'):
            group = group.sort_values('日付', ascending=False)
            
            cold_streak = 0
            for _, row in group.iterrows():
                if row['着順'] == 1:
                    break
                cold_streak += 1
            
            win_dates = group[group['着順'] == 1]['日付']
            if len(win_dates) > 0:
                last_win_days = (latest_date - win_dates.iloc[0]).days
            else:
                last_win_days = 365
            
            self.jockey_time_stats[f"{jockey}_streak"] = {
                'cold_streak': cold_streak,
                'last_win_days': last_win_days
            }
    
    def _calculate_jockey_synergy_stats(self, df: pd.DataFrame):
        """シナジー統計の計算"""
        print("   🤝 シナジー統計計算中...")
        
        if '騎手' in df.columns and '調教師' in df.columns:
            for (jockey, trainer), group in df.groupby(['騎手', '調教師']):
                if len(group) >= 3:
                    key = f"{jockey}_{trainer}"
                    self.jockey_synergy_stats[key] = {
                        'win_rate': (group['着順'] == 1).mean(),
                        'place_rate': (group['着順'] <= 3).mean(),
                        'count': len(group)
                    }


def load_race_data():
    """2020-2025年のエンコード済みデータを読み込む"""
    encoded_path = 'encoded/2020_2025encoded_data_v2.csv'
    
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f"{encoded_path}が見つかりません。先にencode_2020_2025_data.pyを実行してください。")
    
    df = pd.read_csv(encoded_path)
    print(f"データを読み込みました: {encoded_path}")
    print(f"データサイズ: {df.shape}")
    
    # race_idから実際の日付を抽出
    if 'race_id' in df.columns:
        df['race_id_str'] = df['race_id'].astype(str).str.replace('.0', '')
        df['actual_date'] = pd.to_datetime(df['race_id_str'].str[:8], format='%Y%m%d', errors='coerce')
        
        valid_dates = df['actual_date'].notna()
        print(f"\n日付変換成功率: {valid_dates.sum() / len(df) * 100:.1f}%")
        
        if valid_dates.sum() > 0:
            df = df[valid_dates].copy()
            print(f"有効なデータ数: {len(df)}")
    
    return df

def create_features(df, horse_db=None):
    """特徴量エンジニアリング（詳細版）"""
    df_features = df.copy()
    
    print("\n=== 特徴量エンジニアリング開始 ===")
    
    # HorseDatabaseの初期化
    if horse_db is None:
        horse_db = HorseDatabase()
        horse_db.build_database()
    
    # 基本的な特徴量作成
    if '前走着順' in df_features.columns:
        df_features['前走勝利'] = (df_features['前走着順'] == 1).astype(int)
        df_features['前走連対'] = (df_features['前走着順'] <= 2).astype(int)
        df_features['前走着内'] = (df_features['前走着順'] <= 3).astype(int)
    
    # 馬の過去成績特徴量
    if '馬' in df_features.columns:
        horse_features_list = []
        for idx, row in df_features.iterrows():
            horse_name = row['馬']
            current_date = row.get('actual_date', None)
            horse_features = horse_db.get_horse_features(horse_name, current_date)
            horse_features_list.append(horse_features)
        
        # 馬の特徴量をDataFrameに追加
        horse_features_df = pd.DataFrame(horse_features_list)
        for col in horse_features_df.columns:
            if col not in df_features.columns:
                df_features[col] = horse_features_df[col]
    
    # 騎手統計の追加
    if '騎手' in df_features.columns:
        # 基本統計
        df_features['騎手の勝率'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_stats(x)['win_rate']
        )
        df_features['騎手の複勝率'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_stats(x)['place_rate']
        )
        df_features['騎手の騎乗数'] = df_features['騎手'].apply(
            lambda x: np.log1p(horse_db.get_jockey_stats(x)['count'])
        )
        df_features['騎手の平均着順'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_stats(x)['avg_position']
        )
        df_features['騎手のROI'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_stats(x)['roi']
        )
        
        # 時系列統計
        df_features['騎手の勝率_30日'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_time_stats(x, 30)['win_rate']
        )
        df_features['騎手の複勝率_30日'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_time_stats(x, 30)['place_rate']
        )
        df_features['騎手の勝率_60日'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_time_stats(x, 60)['win_rate']
        )
        df_features['騎手の連続不勝'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_streak_stats(x)['cold_streak']
        )
        df_features['騎手の最後勝利日数'] = df_features['騎手'].apply(
            lambda x: np.exp(-horse_db.get_jockey_streak_stats(x)['last_win_days'] / 30)
        )
        
        # コンテキスト統計（芝/ダート）
        if '芝・ダート' in df_features.columns:
            df_features['騎手の勝率_芝'] = df_features['騎手'].apply(
                lambda x: horse_db.get_jockey_context_stats(x, 'surface', '芝')['win_rate']
            )
            df_features['騎手の勝率_ダート'] = df_features['騎手'].apply(
                lambda x: horse_db.get_jockey_context_stats(x, 'surface', 'ダ')['win_rate']
            )
        
        # 距離カテゴリ別
        df_features['騎手の勝率_短距離'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_context_stats(x, 'distance', '短距離')['win_rate']
        )
        df_features['騎手の勝率_中距離'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_context_stats(x, 'distance', '中距離')['win_rate']
        )
        df_features['騎手の勝率_長距離'] = df_features['騎手'].apply(
            lambda x: horse_db.get_jockey_context_stats(x, 'distance', '長距離')['win_rate']
        )
        
        # シナジー統計
        if '調教師' in df_features.columns:
            df_features['騎手調教師相性'] = df_features.apply(
                lambda row: horse_db.get_jockey_synergy_stats(row['騎手'], row['調教師'])['win_rate'],
                axis=1
            )
    
    # 調教師統計
    if '調教師' in df_features.columns:
        df_features['調教師の勝率'] = df_features['調教師'].apply(
            lambda x: horse_db.get_trainer_stats(x)['win_rate']
        )
        df_features['調教師の複勝率'] = df_features['調教師'].apply(
            lambda x: horse_db.get_trainer_stats(x)['place_rate']
        )
    
    # 距離カテゴリ
    if '距離' in df_features.columns:
        df_features['短距離'] = (df_features['距離'] <= 1400).astype(int)
        df_features['マイル'] = ((df_features['距離'] > 1400) & (df_features['距離'] <= 1800)).astype(int)
        df_features['中距離'] = ((df_features['距離'] > 1800) & (df_features['距離'] <= 2400)).astype(int)
        df_features['長距離'] = (df_features['距離'] > 2400).astype(int)
    
    # 枠番の影響
    if '枠番' in df_features.columns and '頭数' in df_features.columns:
        df_features['内枠'] = (df_features['枠番'] <= 3).astype(int)
        df_features['外枠'] = (df_features['枠番'] >= 7).astype(int)
        df_features['相対枠位置'] = df_features['枠番'] / df_features['頭数']
    
    # 時期の影響（2020-2025特有の傾向を捉える）
    if 'actual_date' in df_features.columns:
        df_features['年'] = df_features['actual_date'].dt.year
        df_features['月'] = df_features['actual_date'].dt.month
        df_features['コロナ期間'] = ((df_features['年'] == 2020) | 
                                  ((df_features['年'] == 2021) & (df_features['月'] <= 6))).astype(int)
    
    created_features = len(df_features.columns) - len(df.columns)
    print(f"作成した特徴量数: {created_features}個")
    
    return df_features

def train_model_2020_2025(df_features):
    """2020-2025年データでモデルを訓練（改良版）"""
    print("\n=== モデル訓練開始（2020-2025年データ） ===")
    
    # ターゲット作成
    df_features['target'] = (df_features['着順'] <= 3).astype(int)
    
    # 特徴量選択（新しい特徴量も除外リストから除く）
    exclude_cols = ['着順', 'target', 'オッズ', '人気', '上がり', '走破時間', 
                    '通過順', '日付', 'actual_date', 'year', '月', 'race_id', 
                    'race_id_str', '馬番', '賞金', '馬', '騎手', '調教師']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"使用する特徴量数: {len(feature_cols)}")
    
    # データを時系列順にソート
    df_features = df_features.sort_values('actual_date').reset_index(drop=True)
    
    # 訓練・検証データ分割（2020-2023を訓練、2024-2025を検証）
    train_mask = df_features['年'] <= 2023
    
    X_train = df_features[train_mask][feature_cols]
    y_train = df_features[train_mask]['target']
    X_valid = df_features[~train_mask][feature_cols]
    y_valid = df_features[~train_mask]['target']
    
    print(f"\n訓練データ: {len(X_train)}件 (2020-2023)")
    print(f"検証データ: {len(X_valid)}件 (2024-2025)")
    
    # 欠損値処理（数値型の列のみ）
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_means = X_train[numeric_cols].mean()
    
    # 数値型の列のみ欠損値を埋める
    X_train.loc[:, numeric_cols] = X_train[numeric_cols].fillna(train_means)
    X_valid.loc[:, numeric_cols] = X_valid[numeric_cols].fillna(train_means)
    
    # 非数値型の列は削除または適切に処理
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"警告: 非数値型の列が見つかりました: {list(non_numeric_cols)}")
        # 非数値型の列を削除
        X_train = X_train.drop(columns=non_numeric_cols)
        X_valid = X_valid.drop(columns=non_numeric_cols)
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    # クラス重み
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # LightGBMパラメータ（改良版：詳細な特徴量に対応）
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': -1,
        'num_leaves': 100,  # 増加（特徴量が増えたため）
        'learning_rate': 0.02,  # やや減少（過学習防止）
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,  # 減少（より詳細な学習）
        'n_estimators': 1000,  # 増加
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,  # 増加（正則化強化）
        'max_depth': 8,  # 追加（深さ制限）
        'min_split_gain': 0.01  # 追加（分割の最小利得）
    }
    
    # モデル訓練
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
    )
    
    # 評価
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred)
    
    print(f"\n検証AUCスコア: {auc_score:.4f}")
    
    # 特徴量重要度（更新されたfeature_colsを使用）
    # 実際に使用された特徴量数を確認
    n_features_used = len(model.feature_importances_)
    if len(feature_cols) != n_features_used:
        print(f"警告: 特徴量数が一致しません。元: {len(feature_cols)}, 使用: {n_features_used}")
        # 実際に使用された特徴量のリストを再構築
        actual_feature_cols = list(X_train.columns)[:n_features_used]
    else:
        actual_feature_cols = feature_cols
    
    feature_importance = pd.DataFrame({
        'feature': actual_feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n重要な特徴量トップ20:")
    for _, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:35} 重要度: {row['importance']:.0f}")
    
    # 騎手関連の特徴量重要度を確認
    jockey_features = feature_importance[feature_importance['feature'].str.contains('騎手')]
    if not jockey_features.empty:
        print("\n騎手関連特徴量の重要度:")
        for _, row in jockey_features.head(10).iterrows():
            print(f"{row['feature']:35} 重要度: {row['importance']:.0f}")
    
    # 中間日数関連の特徴量重要度を確認
    interval_features = feature_importance[feature_importance['feature'].str.contains('中間日数|放牧|前走から')]
    if not interval_features.empty:
        print("\n中間日数関連特徴量の重要度:")
        for _, row in interval_features.iterrows():
            print(f"{row['feature']:35} 重要度: {row['importance']:.0f}")
    
    # 実際に使用された特徴量を返す
    return model, actual_feature_cols, auc_score

def save_model(model, feature_cols):
    """モデルと関連情報を保存"""
    model_dir = Path('model_2020_2025')
    model_dir.mkdir(exist_ok=True)
    
    # モデル保存
    joblib.dump(model, model_dir / 'model_2020_2025.pkl')
    
    # 特徴量リスト保存
    joblib.dump(feature_cols, model_dir / 'feature_cols_2020_2025.pkl')
    
    # モデル情報保存
    model_info = {
        'training_period': '2020-2025',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(feature_cols),
        'model_type': 'LightGBM'
    }
    
    import json
    with open(model_dir / 'model_info_2020_2025.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")

def main():
    """メイン処理"""
    print("=" * 60)
    print("競馬予測モデル訓練（2020-2025年データ）- 改良版")
    print("=" * 60)
    
    # データ読み込み
    try:
        df = load_race_data()
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}")
        return
    
    # データ期間の確認
    if 'actual_date' in df.columns:
        print(f"\n=== データ期間の確認 ===")
        print(f"データ期間: {df['actual_date'].min()} ~ {df['actual_date'].max()}")
        df['年'] = df['actual_date'].dt.year
        print(f"\n年別レコード数:")
        print(df['年'].value_counts().sort_index())
    
    # HorseDatabaseの初期化
    print("\n=== 馬・騎手データベース構築 ===")
    horse_db = HorseDatabase()
    horse_db.build_database()
    
    # 特徴量エンジニアリング（HorseDatabaseを渡す）
    df_features = create_features(df, horse_db)
    
    # モデル訓練
    model, feature_cols, auc_score = train_model_2020_2025(df_features)
    
    # モデル保存
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("訓練完了サマリー")
    print("=" * 60)
    print(f"訓練期間: 2020-2025年")
    print(f"検証AUCスコア: {auc_score:.4f}")
    print(f"使用特徴量数: {len(feature_cols)}")
    print("\n改良点:")
    print("- 騎手統計（基本、時系列、コンテキスト、シナジー）を追加")
    print("- 中間日数関連の特徴量を追加")
    print("- 馬の詳細な過去成績を追加")
    print("- LightGBMパラメータを最適化")
    print("\n次のステップ:")
    print("1. integrated_betting_system.pyの'model_path'を'model_2020_2025/model_2020_2025.pkl'に更新")
    print("2. python integrated_betting_system.pyで明日以降のレースを分析")

if __name__ == "__main__":
    main()