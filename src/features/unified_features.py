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
        
        # 1. 人気とオッズの関係
        if '人気' in df.columns and 'オッズ' in df.columns:
            result = self._build_popularity_odds_features(result)
        
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
    
    def _build_popularity_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """人気・オッズ関連特徴量"""
        # オッズの数値化
        odds_numeric = pd.to_numeric(df['オッズ'], errors='coerce').fillna(99.9)
        df['オッズ_numeric'] = odds_numeric
        
        # 人気とオッズの関係
        df['popularity_odds_ratio'] = df['人気'] / (odds_numeric + 1)
        df['is_favorite'] = (df['人気'] <= 3).astype(int)
        df['is_longshot'] = (df['人気'] >= 10).astype(int)
        
        # レース内でのランク
        df['odds_rank'] = df.groupby('race_id')['オッズ_numeric'].rank()
        df['popularity_rank_norm'] = df['人気'] / df.groupby('race_id')['人気'].transform('count')
        
        # オッズレンジ
        df['odds_low'] = (odds_numeric <= 2.0).astype(int)
        df['odds_medium'] = ((odds_numeric > 2.0) & (odds_numeric <= 10.0)).astype(int)
        df['odds_high'] = (odds_numeric > 10.0).astype(int)
        
        return df
    
    def _build_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬番・枠番関連特徴量"""
        # 枠番の計算
        df['枠番'] = ((df['馬番'] - 1) // 2) + 1
        
        # ポジション関連
        df['is_inside_draw'] = (df['馬番'] <= 4).astype(int)
        df['is_outside_draw'] = (df['馬番'] >= 12).astype(int)
        df['is_inner_post'] = (df['枠番'] <= 3).astype(int)
        df['is_outer_post'] = (df['枠番'] >= 7).astype(int)
        
        # 相対ポジション
        if '出走頭数' in df.columns:
            df['draw_position_ratio'] = df['馬番'] / df['出走頭数']
            df['frame_position_ratio'] = df['枠番'] / 8  # 8枠で正規化
        
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
        """構築される特徴量名のリスト"""
        return [
            'オッズ_numeric', 'popularity_odds_ratio', 'is_favorite', 'is_longshot',
            'odds_rank', 'popularity_rank_norm', 'odds_low', 'odds_medium', 'odds_high',
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


class UnifiedFeatureEngine:
    """統一特徴量エンジン"""
    
    def __init__(self):
        """初期化"""
        self.builders: List[FeatureBuilder] = [
            BasicFeatureBuilder(),
            TrackFeatureBuilder(),
            HistoricalFeatureBuilder(),
            PayoutFeatureBuilder()
        ]
        self.feature_names: List[str] = []
    
    def add_builder(self, builder: FeatureBuilder) -> None:
        """特徴量ビルダーを追加"""
        self.builders.append(builder)
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ての特徴量を構築"""
        result = df.copy()
        
        # 各ビルダーで特徴量を構築
        for builder in self.builders:
            try:
                result = builder.build(result)
                self.feature_names.extend(builder.get_feature_names())
            except Exception as e:
                print(f"Warning: Feature builder {builder.__class__.__name__} failed: {e}")
        
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