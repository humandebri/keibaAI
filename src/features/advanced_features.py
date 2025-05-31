#!/usr/bin/env python3
"""
高度な特徴量エンジニアリング
過去成績、血統、騎手・調教師成績などの重要特徴量を追加
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class AdvancedFeatureEngineering:
    """高度な特徴量エンジニアリング"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}  # 計算結果のキャッシュ
        
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """すべての高度な特徴量を追加"""
        self.logger.info("Adding advanced features...")
        
        # データのコピー
        df = df.copy()
        
        # 1. 枠番の追加（馬番から自動計算）
        df = self.add_post_position_features(df)
        
        # 2. 馬の過去成績
        df = self.add_horse_history_features(df)
        
        # 3. 前走からの日数（休養期間）
        df = self.add_rest_days_features(df)
        
        # 4. 騎手の成績
        df = self.add_jockey_performance_features(df)
        
        # 5. 調教師の成績
        df = self.add_trainer_performance_features(df)
        
        # 6. 距離適性（過去の距離別成績）
        df = self.add_distance_preference_features(df)
        
        # 7. 馬場適性（芝・ダート別成績）
        df = self.add_surface_preference_features(df)
        
        # 8. コース適性（競馬場別成績）
        df = self.add_course_preference_features(df)
        
        # 9. ペース指標（仮想）
        df = self.add_pace_features(df)
        
        self.logger.info(f"Added features. Total columns: {len(df.columns)}")
        
        return df
    
    def add_post_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """枠番と関連特徴量を追加"""
        # 枠番の計算（1-2番=1枠、3-4番=2枠...）
        df['枠番'] = ((df['馬番'] - 1) // 2) + 1
        
        # 内枠・外枠フラグ
        df['is_inner_post'] = (df['枠番'] <= 3).astype(int)
        df['is_outer_post'] = (df['枠番'] >= 6).astype(int)
        df['is_middle_post'] = ((df['枠番'] > 3) & (df['枠番'] < 6)).astype(int)
        
        # 距離と枠番の交互作用（外枠は長距離で不利）
        df['post_distance_interaction'] = df['枠番'] * df['距離'] / 1000
        
        return df
    
    def add_horse_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の過去成績特徴量を追加"""
        # 各馬の過去成績を集計
        horse_stats = []
        
        for horse_id in df['horse_id'].unique():
            horse_races = df[df['horse_id'] == horse_id].sort_values('date')
            
            # 直近5走の成績
            recent_positions = []
            recent_times = []
            
            for idx, row in horse_races.iterrows():
                # この馬の過去レース（現在のレースより前）
                past_races = horse_races[horse_races['date'] < row['date']]
                
                if len(past_races) > 0:
                    # 直近5走の着順
                    last_5_positions = past_races.tail(5)['着順'].values
                    recent_positions = list(last_5_positions)
                    
                    # 平均着順
                    avg_position = np.mean(recent_positions) if recent_positions else 10
                    
                    # 勝率・連対率・複勝率
                    win_rate = (past_races['着順'] == 1).mean()
                    place_rate = (past_races['着順'] <= 2).mean()
                    show_rate = (past_races['着順'] <= 3).mean()
                    
                    # 最高着順・最低着順
                    best_position = past_races['着順'].min()
                    worst_position = past_races['着順'].max()
                    
                else:
                    # 初出走の場合
                    avg_position = 10
                    win_rate = place_rate = show_rate = 0
                    best_position = worst_position = 10
                
                horse_stats.append({
                    'race_id': row['race_id'],
                    'horse_id': horse_id,
                    'avg_position_last5': avg_position,
                    'horse_win_rate': win_rate,
                    'horse_place_rate': place_rate,
                    'horse_show_rate': show_rate,
                    'horse_best_position': best_position,
                    'horse_worst_position': worst_position,
                    'horse_race_count': len(past_races)
                })
        
        # 元のデータフレームにマージ
        stats_df = pd.DataFrame(horse_stats)
        df = df.merge(stats_df, on=['race_id', 'horse_id'], how='left')
        
        # 欠損値の処理
        df['avg_position_last5'] = df['avg_position_last5'].fillna(10)
        df['horse_win_rate'] = df['horse_win_rate'].fillna(0)
        df['horse_place_rate'] = df['horse_place_rate'].fillna(0)
        df['horse_show_rate'] = df['horse_show_rate'].fillna(0)
        df['horse_race_count'] = df['horse_race_count'].fillna(0)
        
        return df
    
    def add_rest_days_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """前走からの休養日数を追加"""
        rest_days = []
        
        for idx, row in df.iterrows():
            horse_id = row['horse_id']
            current_date = row['date']
            
            # この馬の前走を探す
            past_races = df[(df['horse_id'] == horse_id) & 
                           (df['date'] < current_date)]
            
            if len(past_races) > 0:
                # 最も近い前走
                last_race = past_races.sort_values('date').iloc[-1]
                
                # 日数計算（dateが文字列の場合は変換）
                try:
                    if isinstance(current_date, str):
                        current = pd.to_datetime(current_date)
                        last = pd.to_datetime(last_race['date'])
                    else:
                        current = current_date
                        last = last_race['date']
                    
                    days = (current - last).days
                except:
                    days = 30  # デフォルト値
            else:
                days = 180  # 初出走または長期休養
            
            rest_days.append(days)
        
        df['rest_days'] = rest_days
        
        # 休養期間のカテゴリ
        df['is_short_rest'] = (df['rest_days'] < 14).astype(int)  # 2週未満
        df['is_normal_rest'] = ((df['rest_days'] >= 14) & 
                                (df['rest_days'] <= 60)).astype(int)  # 2週-2ヶ月
        df['is_long_rest'] = (df['rest_days'] > 60).astype(int)  # 2ヶ月超
        
        # 休養明けフラグ
        df['is_fresh'] = (df['rest_days'] > 90).astype(int)  # 3ヶ月以上
        
        return df
    
    def add_jockey_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の成績を追加"""
        # 騎手ごとの成績を計算
        jockey_stats = df.groupby('騎手').agg({
            '着順': ['count', lambda x: (x == 1).sum(), 
                     lambda x: (x <= 2).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        
        jockey_stats.columns = ['騎手', 'jockey_rides', 'jockey_wins', 
                               'jockey_places', 'jockey_shows']
        
        # 勝率・連対率・複勝率
        jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_rides']
        jockey_stats['jockey_place_rate'] = jockey_stats['jockey_places'] / jockey_stats['jockey_rides']
        jockey_stats['jockey_show_rate'] = jockey_stats['jockey_shows'] / jockey_stats['jockey_rides']
        
        # 元のデータにマージ
        df = df.merge(jockey_stats[['騎手', 'jockey_win_rate', 'jockey_place_rate', 
                                    'jockey_show_rate', 'jockey_rides']], 
                     on='騎手', how='left')
        
        # 騎手の経験レベル
        df['jockey_experience_level'] = pd.cut(df['jockey_rides'], 
                                               bins=[0, 100, 500, 1000, 10000],
                                               labels=[0, 1, 2, 3])
        df['jockey_experience_level'] = df['jockey_experience_level'].astype(int)
        
        return df
    
    def add_trainer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教師の成績を追加"""
        # 調教師ごとの成績を計算
        trainer_stats = df.groupby('調教師').agg({
            '着順': ['count', lambda x: (x == 1).sum(), 
                     lambda x: (x <= 2).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        
        trainer_stats.columns = ['調教師', 'trainer_horses', 'trainer_wins', 
                                'trainer_places', 'trainer_shows']
        
        # 勝率・連対率・複勝率
        trainer_stats['trainer_win_rate'] = trainer_stats['trainer_wins'] / trainer_stats['trainer_horses']
        trainer_stats['trainer_place_rate'] = trainer_stats['trainer_places'] / trainer_stats['trainer_horses']
        trainer_stats['trainer_show_rate'] = trainer_stats['trainer_shows'] / trainer_stats['trainer_horses']
        
        # 元のデータにマージ
        df = df.merge(trainer_stats[['調教師', 'trainer_win_rate', 'trainer_place_rate', 
                                     'trainer_show_rate', 'trainer_horses']], 
                     on='調教師', how='left')
        
        return df
    
    def add_distance_preference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離適性の特徴量を追加"""
        # 距離カテゴリを作成
        df['distance_category'] = pd.cut(df['距離'], 
                                        bins=[0, 1400, 1800, 2200, 3000],
                                        labels=['sprint', 'mile', 'middle', 'long'])
        
        # 各馬の距離カテゴリ別成績
        distance_prefs = []
        
        for idx, row in df.iterrows():
            horse_id = row['horse_id']
            current_distance_cat = row['distance_category']
            
            # この馬の同距離カテゴリでの過去成績
            past_same_distance = df[(df['horse_id'] == horse_id) & 
                                   (df['date'] < row['date']) &
                                   (df['distance_category'] == current_distance_cat)]
            
            if len(past_same_distance) > 0:
                distance_win_rate = (past_same_distance['着順'] == 1).mean()
                distance_show_rate = (past_same_distance['着順'] <= 3).mean()
                distance_avg_position = past_same_distance['着順'].mean()
            else:
                distance_win_rate = 0
                distance_show_rate = 0
                distance_avg_position = 10
            
            distance_prefs.append({
                'idx': idx,
                'distance_win_rate': distance_win_rate,
                'distance_show_rate': distance_show_rate,
                'distance_avg_position': distance_avg_position
            })
        
        # データフレームに追加
        prefs_df = pd.DataFrame(distance_prefs).set_index('idx')
        df = pd.concat([df, prefs_df], axis=1)
        
        return df
    
    def add_surface_preference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬場（芝・ダート）適性の特徴量を追加"""
        surface_prefs = []
        
        for idx, row in df.iterrows():
            horse_id = row['horse_id']
            current_surface = row['芝・ダート']
            
            # この馬の同じ馬場での過去成績
            past_same_surface = df[(df['horse_id'] == horse_id) & 
                                  (df['date'] < row['date']) &
                                  (df['芝・ダート'] == current_surface)]
            
            if len(past_same_surface) > 0:
                surface_win_rate = (past_same_surface['着順'] == 1).mean()
                surface_show_rate = (past_same_surface['着順'] <= 3).mean()
            else:
                surface_win_rate = 0
                surface_show_rate = 0
            
            surface_prefs.append({
                'idx': idx,
                'surface_win_rate': surface_win_rate,
                'surface_show_rate': surface_show_rate
            })
        
        prefs_df = pd.DataFrame(surface_prefs).set_index('idx')
        df = pd.concat([df, prefs_df], axis=1)
        
        return df
    
    def add_course_preference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース（競馬場）適性の特徴量を追加"""
        course_prefs = []
        
        for idx, row in df.iterrows():
            horse_id = row['horse_id']
            current_course = row['場名']
            
            # この馬の同じ競馬場での過去成績
            past_same_course = df[(df['horse_id'] == horse_id) & 
                                 (df['date'] < row['date']) &
                                 (df['場名'] == current_course)]
            
            if len(past_same_course) > 0:
                course_win_rate = (past_same_course['着順'] == 1).mean()
                course_show_rate = (past_same_course['着順'] <= 3).mean()
                course_experience = len(past_same_course)
            else:
                course_win_rate = 0
                course_show_rate = 0
                course_experience = 0
            
            course_prefs.append({
                'idx': idx,
                'course_win_rate': course_win_rate,
                'course_show_rate': course_show_rate,
                'course_experience': course_experience
            })
        
        prefs_df = pd.DataFrame(course_prefs).set_index('idx')
        df = pd.concat([df, prefs_df], axis=1)
        
        return df
    
    def add_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ペース関連の特徴量を追加（仮想的な実装）"""
        # コーナー通過順位が取得できない場合の代替
        # 人気と枠番から推定
        
        # 逃げ馬の可能性（内枠で人気薄）
        df['likely_front_runner'] = ((df['枠番'] <= 3) & 
                                    (df['人気'] >= 5)).astype(int)
        
        # 差し・追い込み馬の可能性（外枠または人気馬）
        df['likely_closer'] = ((df['枠番'] >= 6) | 
                              (df['人気'] <= 3)).astype(int)
        
        # ペース予測（レース全体）
        # 逃げ馬候補の数でペースを推定
        race_pace = df.groupby('race_id')['likely_front_runner'].transform('sum')
        df['expected_pace'] = pd.cut(race_pace, 
                                    bins=[0, 2, 4, 20],
                                    labels=['slow', 'medium', 'fast'])
        
        # ダミー変数化
        df = pd.get_dummies(df, columns=['expected_pace'], prefix='pace')
        
        return df
    
    def add_blood_features(self, df: pd.DataFrame, blood_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """血統情報の特徴量を追加（データがある場合）"""
        if blood_data is None:
            # 血統データがない場合はダミー特徴量
            df['has_blood_info'] = 0
            return df
        
        # 血統データとマージ
        df = df.merge(blood_data[['horse_id', 'father', 'mother_father']], 
                     on='horse_id', how='left')
        
        # 主要種牡馬のフラグ
        top_sires = ['ディープインパクト', 'キングカメハメハ', 'ロードカナロア']
        for sire in top_sires:
            df[f'sire_{sire}'] = (df['father'] == sire).astype(int)
        
        # 距離適性の高い血統
        stamina_sires = ['ステイゴールド', 'ハーツクライ']
        df['stamina_blood'] = df['father'].isin(stamina_sires).astype(int)
        
        # スピード血統
        speed_sires = ['ロードカナロア', 'サクラバクシンオー']
        df['speed_blood'] = df['father'].isin(speed_sires).astype(int)
        
        df['has_blood_info'] = 1
        
        return df