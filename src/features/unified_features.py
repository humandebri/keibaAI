#!/usr/bin/env python3
"""
統一特徴量エンジニアリング

複数のシステム実装で重複していた特徴量作成ロジックを統合し、
一貫性のある特徴量エンジニアリングを提供します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


class FeatureBuilder(ABC):
    """特徴量ビルダーの基底クラス"""
    
    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を構築する"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリストを返す"""
        pass


class BasicFeatureBuilder(FeatureBuilder):
    """基本特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量を構築"""
        result = df.copy()
        
        # 1. 人気関連特徴量（オッズは除外）
        if '人気' in df.columns:
            result = self._build_popularity_features(result)
        
        # 2. 馬番（枠番）の影響
        if '馬番' in df.columns:
            result = self._build_draw_features(result)
        
        # 3. 斤量の影響
        if '斤量' in df.columns:
            result = self._build_weight_features(result)
        
        # 4. 体重の処理
        if '体重' in df.columns:
            result = self._build_horse_weight_features(result)
        
        # 5. 性別・年齢の影響
        if '性' in df.columns:
            result = self._build_gender_features(result)
        
        if '年齢' in df.columns:
            result = self._build_age_features(result)
        
        return result
    
    def _build_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """人気関連特徴量（オッズを除外）"""
        # 人気関連フラグ
        df['is_favorite'] = (df['人気'] <= 3).astype(int)
        df['is_longshot'] = (df['人気'] >= 10).astype(int)
        
        # 人気の正規化
        df['popularity_rank_norm'] = df['人気'] / df.groupby('race_id')['人気'].transform('count')
        
        # 人気の相対順位
        df['popularity_rank'] = df.groupby('race_id')['人気'].rank()
        
        return df
    
    def _build_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬番・枠番関連特徴量"""
        # 枠番の計算（馬番から導出）
        df['枠番'] = ((df['馬番'] - 1) // 2) + 1  # 1,2→1, 3,4→2, ...
        df['枠番'] = df['枠番'].clip(upper=8)  # 最大8枠
        
        # ポジション関連特徴量
        df['is_inside_draw'] = (df['馬番'] <= 4).astype(int)
        df['is_outside_draw'] = (df['馬番'] >= 13).astype(int)
        df['is_inner_post'] = (df['枠番'] <= 2).astype(int)
        df['is_outer_post'] = (df['枠番'] >= 7).astype(int)
        
        # 相対位置
        if '出走頭数' in df.columns:
            df['draw_position_ratio'] = df['馬番'] / df['出走頭数']
        else:
            df['draw_position_ratio'] = df['馬番'] / df.groupby('race_id')['馬番'].transform('count')
        df['frame_position_ratio'] = df['枠番'] / 8.0
        
        return df
    
    def _build_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """斤量関連特徴量"""
        # 重量カテゴリ
        df['weight_heavy'] = (df['斤量'] >= 57).astype(int)
        df['weight_light'] = (df['斤量'] <= 54).astype(int)
        df['weight_medium'] = ((df['斤量'] > 54) & (df['斤量'] < 57)).astype(int)
        
        # 正規化重量
        df['weight_norm'] = (df['斤量'] - df['斤量'].mean()) / df['斤量'].std()
        
        # レース内での相対重量
        df['weight_relative'] = df.groupby('race_id')['斤量'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        return df
    
    def _build_horse_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬体重関連特徴量"""
        # 体重の数値化
        weight_values = []
        for w in df['体重']:
            try:
                weight = int(str(w).split('(')[0]) if pd.notna(w) else 480
            except:
                weight = 480
            weight_values.append(weight)
        
        df['体重_numeric'] = weight_values
        
        # 体重カテゴリ
        df['is_heavy_horse'] = (df['体重_numeric'] >= 500).astype(int)
        df['is_light_horse'] = (df['体重_numeric'] <= 440).astype(int)
        df['is_medium_horse'] = (
            (df['体重_numeric'] > 440) & (df['体重_numeric'] < 500)
        ).astype(int)
        
        # 体重変化
        if '体重変化' in df.columns:
            df['weight_change_abs'] = df['体重変化'].abs()
            df['weight_increased'] = (df['体重変化'] > 0).astype(int)
            df['weight_decreased'] = (df['体重変化'] < 0).astype(int)
            df['weight_stable'] = (df['体重変化'] == 0).astype(int)
            
            # 体重変化の相対値
            df['weight_change_ratio'] = df['体重変化'] / df['体重_numeric']
        
        return df
    
    def _build_gender_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """性別関連特徴量"""
        # 性別エンコーディング
        gender_map = {'牡': 0, '牝': 1, 'セ': 2}
        df['性_encoded'] = df['性'].map(gender_map).fillna(0)
        
        # 性別フラグ
        df['is_male'] = (df['性'] == '牡').astype(int)
        df['is_female'] = (df['性'] == '牝').astype(int)
        df['is_gelding'] = (df['性'] == 'セ').astype(int)
        
        return df
    
    def _build_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """年齢関連特徴量"""
        # 年齢カテゴリ
        df['is_young'] = (df['年齢'] <= 3).astype(int)
        df['is_prime'] = ((df['年齢'] >= 4) & (df['年齢'] <= 6)).astype(int)
        df['is_veteran'] = (df['年齢'] >= 7).astype(int)
        
        # 年齢の二乗項（非線形関係）
        df['age_squared'] = df['年齢'] ** 2
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト（オッズ関連は除外）"""
        return [
            'is_favorite', 'is_longshot', 'popularity_rank_norm', 'popularity_rank',
            '枠番', 'is_inside_draw', 'is_outside_draw', 'is_inner_post', 'is_outer_post',
            'draw_position_ratio', 'frame_position_ratio',
            'weight_heavy', 'weight_light', 'weight_medium', 'weight_norm', 'weight_relative',
            '体重_numeric', 'is_heavy_horse', 'is_light_horse', 'is_medium_horse',
            'weight_change_abs', 'weight_increased', 'weight_decreased', 'weight_stable', 'weight_change_ratio',
            '性_encoded', 'is_male', 'is_female', 'is_gelding',
            'is_young', 'is_prime', 'is_veteran', 'age_squared'
        ]


class TrackFeatureBuilder(FeatureBuilder):
    """コース・馬場関連特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース・馬場関連特徴量を構築"""
        result = df.copy()
        
        # 1. 馬場状態の処理
        if '馬場' in df.columns:
            result = self._build_track_condition_features(result)
        
        # 2. コース種別の処理
        if '芝・ダート' in df.columns:
            result = self._build_surface_features(result)
        
        # 3. 距離の処理
        if '距離' in df.columns:
            result = self._build_distance_features(result)
        
        # 4. 天気の処理
        if '天気' in df.columns:
            result = self._build_weather_features(result)
        
        # 5. 交互作用特徴量
        result = self._build_interaction_features(result)
        
        return result
    
    def _build_track_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬場状態特徴量"""
        # 馬場状態の数値化
        track_map = {'良': 1.0, '稍': 0.8, '稍重': 0.8, '重': 0.6, '不': 0.4, '不良': 0.4}
        df['track_condition_numeric'] = df['馬場'].map(track_map).fillna(0.7)
        
        # 馬場指数
        df['track_moisture_index'] = df['track_condition_numeric']
        df['track_cushion_value'] = df['track_moisture_index'] * 0.5 + 0.5
        
        # 馬場状態フラグ
        df['is_good_track'] = (df['馬場'] == '良').astype(int)
        df['is_heavy_track'] = (df['馬場'].isin(['重', '不', '不良'])).astype(int)
        df['is_soft_track'] = (df['馬場'].isin(['稍', '稍重'])).astype(int)
        
        return df
    
    def _build_surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース種別特徴量"""
        # コース種別エンコーディング
        surface_map = {'芝': 0, 'ダ': 1, '障': 2}
        df['surface_encoded'] = df['芝・ダート'].map(surface_map).fillna(0)
        
        # コース種別フラグ
        df['is_turf'] = (df['芝・ダート'] == '芝').astype(int)
        df['is_dirt'] = (df['芝・ダート'] == 'ダ').astype(int)
        df['is_jump'] = (df['芝・ダート'] == '障').astype(int)
        
        return df
    
    def _build_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離関連特徴量"""
        # 距離カテゴリ
        df['is_sprint'] = (df['距離'] <= 1200).astype(int)
        df['is_mile'] = ((df['距離'] > 1200) & (df['距離'] <= 1600)).astype(int)
        df['is_middle'] = ((df['距離'] > 1600) & (df['距離'] <= 2200)).astype(int)
        df['is_long'] = (df['距離'] > 2200).astype(int)
        
        # 距離の正規化
        df['distance_norm'] = (df['距離'] - df['距離'].mean()) / df['距離'].std()
        
        # 距離の対数変換
        df['distance_log'] = np.log(df['距離'])
        
        return df
    
    def _build_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """天気関連特徴量"""
        # 天気エンコーディング
        weather_map = {'晴': 1.0, '曇': 0.7, '小雨': 0.5, '雨': 0.3, '雪': 0.1}
        df['weather_numeric'] = df['天気'].map(weather_map).fillna(0.7)
        
        # 天気フラグ
        df['is_sunny'] = (df['天気'] == '晴').astype(int)
        df['is_rainy'] = (df['天気'].isin(['小雨', '雨'])).astype(int)
        df['is_cloudy'] = (df['天気'] == '曇').astype(int)
        
        return df
    
    def _build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量"""
        # 枠番×馬場状態
        if '馬番' in df.columns and 'track_moisture_index' in df.columns:
            df['draw_track_interaction'] = df['馬番'] * df['track_moisture_index']
            
            # 内枠有利/不利の指標
            df['inside_draw_advantage'] = np.where(
                (df['馬番'] <= 4) & (df['track_moisture_index'] < 0.8),
                1.2,  # 重馬場で内枠有利
                1.0
            )
        
        # 距離×コース種別
        if '距離' in df.columns and 'surface_encoded' in df.columns:
            df['distance_surface_interaction'] = df['距離'] * df['surface_encoded']
        
        # 天気×馬場状態
        if 'weather_numeric' in df.columns and 'track_condition_numeric' in df.columns:
            df['weather_track_interaction'] = df['weather_numeric'] * df['track_condition_numeric']
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return [
            'track_condition_numeric', 'track_moisture_index', 'track_cushion_value',
            'is_good_track', 'is_heavy_track', 'is_soft_track',
            'surface_encoded', 'is_turf', 'is_dirt', 'is_jump',
            'is_sprint', 'is_mile', 'is_middle', 'is_long', 'distance_norm', 'distance_log',
            'weather_numeric', 'is_sunny', 'is_rainy', 'is_cloudy',
            'draw_track_interaction', 'inside_draw_advantage', 'distance_surface_interaction',
            'weather_track_interaction'
        ]


class HistoricalFeatureBuilder(FeatureBuilder):
    """過去成績関連特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去成績特徴量を構築"""
        result = df.copy()
        
        # 馬の過去成績
        if '馬名' in df.columns:
            result = self._build_horse_history_features(result)
        
        # 騎手の過去成績  
        if '騎手' in df.columns:
            result = self._build_jockey_history_features(result)
        
        # 調教師の過去成績
        if '調教師' in df.columns:
            result = self._build_trainer_history_features(result)
        
        return result
    
    def _build_horse_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の過去成績特徴量"""
        # 馬別の勝率・連対率を計算（全データを使用）
        horse_stats = df.groupby('馬名')['着順'].agg([
            ('total_races', 'count'),
            ('wins', lambda x: (x == 1).sum()),
            ('places', lambda x: (x <= 3).sum())
        ]).reset_index()
        
        horse_stats['horse_win_rate'] = horse_stats['wins'] / horse_stats['total_races']
        horse_stats['horse_place_rate'] = horse_stats['places'] / horse_stats['total_races']
        
        # 元のデータフレームにマージ
        df = df.merge(
            horse_stats[['馬名', 'horse_win_rate', 'horse_place_rate', 'total_races']],
            on='馬名',
            how='left'
        )
        
        # 欠損値を埋める
        df['horse_win_rate'] = df['horse_win_rate'].fillna(0.1)
        df['horse_place_rate'] = df['horse_place_rate'].fillna(0.3)
        df['total_races'] = df['total_races'].fillna(1)
        
        return df
    
    def _build_jockey_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の過去成績特徴量"""
        # 騎手別の勝率・連対率を計算
        jockey_stats = df.groupby('騎手')['着順'].agg([
            ('jockey_total_races', 'count'),
            ('jockey_wins', lambda x: (x == 1).sum()),
            ('jockey_places', lambda x: (x <= 3).sum())
        ]).reset_index()
        
        jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_total_races']
        jockey_stats['jockey_place_rate'] = jockey_stats['jockey_places'] / jockey_stats['jockey_total_races']
        
        # 元のデータフレームにマージ
        df = df.merge(
            jockey_stats[['騎手', 'jockey_win_rate', 'jockey_place_rate', 'jockey_total_races']],
            on='騎手',
            how='left'
        )
        
        # 欠損値を埋める
        df['jockey_win_rate'] = df['jockey_win_rate'].fillna(0.1)
        df['jockey_place_rate'] = df['jockey_place_rate'].fillna(0.3)
        df['jockey_total_races'] = df['jockey_total_races'].fillna(1)
        
        return df
    
    def _build_trainer_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教師の過去成績特徴量"""
        # 調教師別の勝率・連対率を計算
        trainer_stats = df.groupby('調教師')['着順'].agg([
            ('trainer_total_races', 'count'),
            ('trainer_wins', lambda x: (x == 1).sum()),
            ('trainer_places', lambda x: (x <= 3).sum())
        ]).reset_index()
        
        trainer_stats['trainer_win_rate'] = trainer_stats['trainer_wins'] / trainer_stats['trainer_total_races']
        trainer_stats['trainer_place_rate'] = trainer_stats['trainer_places'] / trainer_stats['trainer_total_races']
        
        # 元のデータフレームにマージ
        df = df.merge(
            trainer_stats[['調教師', 'trainer_win_rate', 'trainer_place_rate', 'trainer_total_races']],
            on='調教師',
            how='left'
        )
        
        # 欠損値を埋める
        df['trainer_win_rate'] = df['trainer_win_rate'].fillna(0.1)
        df['trainer_place_rate'] = df['trainer_place_rate'].fillna(0.3)
        df['trainer_total_races'] = df['trainer_total_races'].fillna(1)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return [
            'horse_win_rate', 'horse_place_rate', 'total_races',
            'jockey_win_rate', 'jockey_place_rate', 'jockey_total_races',
            'trainer_win_rate', 'trainer_place_rate', 'trainer_total_races'
        ]


class PayoutFeatureBuilder(FeatureBuilder):
    """配当データ関連特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """配当データ特徴量を構築"""
        result = df.copy()
        
        if 'payout_data' in df.columns:
            result = self._build_payout_features(result)
        
        return result
    
    def _build_payout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """配当データから特徴量を作成"""
        # 配当データの解析
        payout_features = []
        
        for idx, payout_str in df['payout_data'].items():
            features = {
                '単勝最高配当': 0,
                '複勝最低配当': 0,
                '三連単配当': 0,
                '高配当レース': 0
            }
            
            try:
                if pd.notna(payout_str) and payout_str != '{}':
                    import json
                    payout_data = json.loads(payout_str)
                    
                    # 単勝最高配当
                    if 'win' in payout_data and payout_data['win']:
                        win_payouts = [int(v) for v in payout_data['win'].values()]
                        features['単勝最高配当'] = max(win_payouts) if win_payouts else 0
                    
                    # 複勝最低配当
                    if 'place' in payout_data and payout_data['place']:
                        place_payouts = [int(v) for v in payout_data['place'].values()]
                        features['複勝最低配当'] = min(place_payouts) if place_payouts else 0
                    
                    # 三連単配当
                    if 'trifecta' in payout_data and payout_data['trifecta']:
                        trifecta_payouts = [int(v) for v in payout_data['trifecta'].values()]
                        features['三連単配当'] = max(trifecta_payouts) if trifecta_payouts else 0
                    
                    # 高配当レース判定
                    features['高配当レース'] = 1 if features['単勝最高配当'] > 1000 else 0
                    
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            
            payout_features.append(features)
        
        # 特徴量をデータフレームに追加
        payout_df = pd.DataFrame(payout_features, index=df.index)
        for col in payout_df.columns:
            df[col] = payout_df[col]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return ['単勝最高配当', '複勝最低配当', '三連単配当', '高配当レース']


class SpeedIndexFeatureBuilder(FeatureBuilder):
    """スピード指数関連特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """スピード指数特徴量を構築"""
        result = df.copy()
        
        # 基本スピード指数（距離/時間）
        if '距離' in df.columns and '走破時間' in df.columns:
            result = self._build_basic_speed_index(result)
        
        # トラック調整スピード指数
        if 'track_condition_numeric' in df.columns:
            result = self._build_track_adjusted_speed(result)
        
        # 過去走からのスピード関連特徴量
        result = self._build_historical_speed_features(result)
        
        return result
    
    def _build_basic_speed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本スピード指数の計算"""
        # 走破時間を秒に変換
        time_seconds = self._convert_time_to_seconds(df['走破時間'])
        
        # 基本スピード指数 = 距離 / 時間
        df['基本スピード指数'] = df['距離'] / time_seconds
        
        # スピード指数の正規化（レース内）
        df['スピード指数_正規化'] = df.groupby('race_id')['基本スピード指数'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        return df
    
    def _build_track_adjusted_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """トラック調整スピード指数"""
        if '基本スピード指数' in df.columns and 'track_condition_numeric' in df.columns:
            # 馬場状態による調整係数
            track_adjustment = 1.0 + (1.0 - df['track_condition_numeric']) * 0.05
            df['トラック調整スピード'] = df['基本スピード指数'] * track_adjustment
        
        return df
    
    def _build_historical_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去走のスピード関連特徴量"""
        # 過去走のスピード指数を計算
        speed_columns = []
        for i in range(1, 4):  # 過去3走
            distance_col = f'距離{i}'
            time_col = f'走破時間{i}'
            if distance_col in df.columns and time_col in df.columns:
                time_seconds = self._convert_time_to_seconds(df[time_col])
                df[f'スピード指数{i}'] = df[distance_col] / time_seconds
                speed_columns.append(f'スピード指数{i}')
        
        # ベストスピード指数（過去3走の最高値）
        if speed_columns:
            df['ベストスピード指数'] = df[speed_columns].max(axis=1, skipna=True)
            df['平均スピード指数'] = df[speed_columns].mean(axis=1, skipna=True)
            
            # スピード指数トレンド（最新 - 最古）
            if len(speed_columns) >= 2:
                df['スピード指数トレンド'] = df[speed_columns[0]] - df[speed_columns[-1]]
        
        return df
    
    def _convert_time_to_seconds(self, time_series: pd.Series) -> pd.Series:
        """時間を秒に変換"""
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                if isinstance(time_str, (int, float)):
                    return float(time_str)
                time_str = str(time_str)
                if ':' in time_str:
                    parts = time_str.split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_str)
            except:
                return np.nan
        
        return time_series.apply(time_to_seconds)
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return [
            '基本スピード指数', 'スピード指数_正規化', 'トラック調整スピード',
            'スピード指数1', 'スピード指数2', 'スピード指数3',
            'ベストスピード指数', '平均スピード指数', 'スピード指数トレンド'
        ]


class RelativeRankingFeatureBuilder(FeatureBuilder):
    """レース内相対順位特徴量ビルダー"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """相対順位特徴量を構築"""
        result = df.copy()
        
        # レース内での相対順位
        if 'race_id' in df.columns:
            result = self._build_race_relative_rankings(result)
        
        # パフォーマンス関連順位
        result = self._build_performance_rankings(result)
        
        return result
    
    def _build_race_relative_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """レース内相対順位の計算"""
        # 人気順位（既に存在する場合はスキップ）
        if '人気' in df.columns and 'popularity_rank_norm' not in df.columns:
            df['人気順位_レース内'] = df.groupby('race_id')['人気'].rank()
            df['人気順位_正規化'] = df['人気順位_レース内'] / df.groupby('race_id')['人気順位_レース内'].transform('count')
        
        # オッズ順位
        if 'オッズ' in df.columns:
            df['オッズ順位_レース内'] = df.groupby('race_id')['オッズ'].rank()
            df['オッズ順位_正規化'] = df['オッズ順位_レース内'] / df.groupby('race_id')['オッズ順位_レース内'].transform('count')
        
        # 斤量順位
        if '斤量' in df.columns:
            df['斤量順位_レース内'] = df.groupby('race_id')['斤量'].rank(ascending=False)  # 重い方が上位
            df['斤量順位_正規化'] = df['斤量順位_レース内'] / df.groupby('race_id')['斤量順位_レース内'].transform('count')
        
        # 体重順位
        if '体重_numeric' in df.columns:
            df['体重順位_レース内'] = df.groupby('race_id')['体重_numeric'].rank(ascending=False)
            df['体重順位_正規化'] = df['体重順位_レース内'] / df.groupby('race_id')['体重順位_レース内'].transform('count')
        
        return df
    
    def _build_performance_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """パフォーマンス関連順位"""
        # スピード指数順位
        if '基本スピード指数' in df.columns:
            df['スピード順位_レース内'] = df.groupby('race_id')['基本スピード指数'].rank(ascending=False)
            df['スピード順位_正規化'] = df['スピード順位_レース内'] / df.groupby('race_id')['スピード順位_レース内'].transform('count')
        
        # 過去成績順位
        if 'horse_win_rate' in df.columns:
            df['勝率順位_レース内'] = df.groupby('race_id')['horse_win_rate'].rank(ascending=False)
            df['勝率順位_正規化'] = df['勝率順位_レース内'] / df.groupby('race_id')['勝率順位_レース内'].transform('count')
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return [
            '人気順位_レース内', '人気順位_正規化',
            'オッズ順位_レース内', 'オッズ順位_正規化',
            '斤量順位_レース内', '斤量順位_正規化',
            '体重順位_レース内', '体重順位_正規化',
            'スピード順位_レース内', 'スピード順位_正規化',
            '勝率順位_レース内', '勝率順位_正規化'
        ]


class ChangeDetectionFeatureBuilder(FeatureBuilder):
    """変化検出特徴量ビルダー（クラス・距離・コース変更など）"""
    
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """変化検出特徴量を構築"""
        result = df.copy()
        
        # クラス変化
        if 'クラス' in df.columns and 'クラス1' in df.columns:
            result = self._build_class_changes(result)
        
        # 距離変化
        if '距離' in df.columns and '距離1' in df.columns:
            result = self._build_distance_changes(result)
        
        # コース変化
        if '芝・ダート' in df.columns and '芝・ダート1' in df.columns:
            result = self._build_surface_changes(result)
        
        # 負担重量比
        if '斤量' in df.columns and '体重_numeric' in df.columns:
            result = self._build_weight_burden_ratio(result)
        
        return result
    
    def _build_class_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """クラス変化の検出"""
        df['クラス変化'] = df['クラス'] - df['クラス1']
        df['クラス昇級'] = (df['クラス変化'] > 0).astype(int)
        df['クラス降級'] = (df['クラス変化'] < 0).astype(int)
        df['クラス維持'] = (df['クラス変化'] == 0).astype(int)
        
        return df
    
    def _build_distance_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離変化の検出"""
        df['距離変化_絶対値'] = (df['距離'] - df['距離1']).abs()
        df['距離延長'] = (df['距離'] > df['距離1']).astype(int)
        df['距離短縮'] = (df['距離'] < df['距離1']).astype(int)
        df['距離変化率'] = (df['距離'] - df['距離1']) / df['距離1']
        
        # 距離カテゴリ変化
        def distance_category(distance):
            if pd.isna(distance):
                return 0
            if distance <= 1200:
                return 1  # スプリント
            elif distance <= 1600:
                return 2  # マイル
            elif distance <= 2200:
                return 3  # 中距離
            else:
                return 4  # 長距離
        
        df['距離カテゴリ'] = df['距離'].apply(distance_category)
        df['距離カテゴリ1'] = df['距離1'].apply(distance_category)
        df['距離カテゴリ変化'] = df['距離カテゴリ'] - df['距離カテゴリ1']
        
        return df
    
    def _build_surface_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース変化の検出"""
        df['コース変更'] = (df['芝・ダート'] != df['芝・ダート1']).astype(int)
        df['芝からダート'] = ((df['芝・ダート1'] == 0) & (df['芝・ダート'] == 1)).astype(int)
        df['ダートから芝'] = ((df['芝・ダート1'] == 1) & (df['芝・ダート'] == 0)).astype(int)
        
        return df
    
    def _build_weight_burden_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """負担重量比の計算"""
        # 斤量/体重の比率
        df['負担重量比'] = df['斤量'] / df['体重_numeric']
        
        # レース内での相対的な負担重量
        df['相対負担重量'] = df.groupby('race_id')['負担重量比'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """構築される特徴量名のリスト"""
        return [
            'クラス変化', 'クラス昇級', 'クラス降級', 'クラス維持',
            '距離変化_絶対値', '距離延長', '距離短縮', '距離変化率',
            '距離カテゴリ', '距離カテゴリ1', '距離カテゴリ変化',
            'コース変更', '芝からダート', 'ダートから芝',
            '負担重量比', '相対負担重量'
        ]


class UnifiedFeatureEngine:
    """統一特徴量エンジン"""
    
    def __init__(self):
        """初期化"""
        self.builders: List[FeatureBuilder] = [
            BasicFeatureBuilder(),
            TrackFeatureBuilder(),
            HistoricalFeatureBuilder(),
            PayoutFeatureBuilder(),
            SpeedIndexFeatureBuilder(),
            RelativeRankingFeatureBuilder(),
            ChangeDetectionFeatureBuilder()
        ]
        self.feature_names: List[str] = []
    
    def add_builder(self, builder: FeatureBuilder) -> None:
        """特徴量ビルダーを追加"""
        self.builders.append(builder)
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ての特徴量を構築"""
        result = df.copy()
        
        # 特徴量名リストをクリアして再構築
        self.feature_names = []
        
        # 各ビルダーで特徴量を構築
        for builder in self.builders:
            try:
                result = builder.build(result)
                builder_features = builder.get_feature_names()
                # 重複を避けて特徴量名を追加
                for feature in builder_features:
                    if feature not in self.feature_names:
                        self.feature_names.append(feature)
                print(f"✓ {builder.__class__.__name__}: {len(builder_features)} features added")
            except Exception as e:
                print(f"Warning: Feature builder {builder.__class__.__name__} failed: {e}")
        
        print(f"Total features registered: {len(self.feature_names)}")
        return result
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """実際に存在する特徴量カラムのリストを返す"""
        return [col for col in self.feature_names if col in df.columns]
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """ターゲット変数を作成"""
        result = df.copy()
        
        if '着順' in df.columns:
            result['is_win'] = (result['着順'] == 1).astype(int)
            result['is_place'] = (result['着順'] <= 3).astype(int)
            result['is_exacta'] = (result['着順'] <= 2).astype(int)
        
        return result