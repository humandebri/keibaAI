#!/usr/bin/env python3
"""
効率的な特徴量エンジニアリング
ベクトル化された計算で高速化
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

class EfficientFeatureEngineering:
    """効率的な特徴量エンジニアリング"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量を高速に追加"""
        self.logger.info("Adding basic features...")
        df = df.copy()
        
        # 1. 枠番（ベクトル化）
        df['枠番'] = ((df['馬番'] - 1) // 2) + 1
        df['is_inner_post'] = (df['枠番'] <= 3).astype(int)
        df['is_outer_post'] = (df['枠番'] >= 6).astype(int)
        
        # 2. 基本的な集計特徴量（グループごとの一括計算）
        
        # 騎手の勝率（全データで計算してからマージ）
        jockey_stats = df.groupby('騎手').agg({
            '着順': ['count', lambda x: (x == 1).mean(), 
                     lambda x: (x <= 3).mean()]
        })
        jockey_stats.columns = ['jockey_rides', 'jockey_win_rate', 'jockey_show_rate']
        jockey_stats = jockey_stats.reset_index()
        
        # 調教師の勝率
        trainer_stats = df.groupby('調教師').agg({
            '着順': ['count', lambda x: (x == 1).mean(), 
                     lambda x: (x <= 3).mean()]
        })
        trainer_stats.columns = ['trainer_rides', 'trainer_win_rate', 'trainer_show_rate']
        trainer_stats = trainer_stats.reset_index()
        
        # マージ
        df = df.merge(jockey_stats, on='騎手', how='left')
        df = df.merge(trainer_stats, on='調教師', how='left')
        
        # 3. 距離カテゴリ
        df['distance_category'] = pd.cut(df['距離'], 
                                        bins=[0, 1400, 1800, 2200, 5000],
                                        labels=['sprint', 'mile', 'middle', 'long'])
        
        # 4. 簡易的な馬の強さ指標（人気とオッズから）
        df['horse_strength'] = 1 / (df['人気'] * np.log1p(df['オッズ']))
        
        # 5. レース内での相対指標
        df['relative_popularity'] = df.groupby('race_id')['人気'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # 欠損値処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        self.logger.info(f"Added basic features. Total: {len(df.columns)} columns")
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, 
                           window_sizes: list = [3, 5, 10]) -> pd.DataFrame:
        """ローリング特徴量を追加（簡易版）"""
        self.logger.info("Adding rolling features...")
        df = df.copy()
        
        # 馬ごとの過去成績（シンプルな実装）
        # ソート済みと仮定
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['race_id'].astype(str).str[:8], 
                                       format='%Y%m%d', errors='coerce')
        
        # 馬の識別子作成
        if 'horse_id' not in df.columns:
            df['horse_id'] = df['馬'].astype(str) + '_' + df.index.astype(str)
        
        # 各馬の過去N走の平均着順（シンプルな移動平均）
        for window in window_sizes:
            df[f'avg_position_last{window}'] = df.groupby('horse_id')['着順'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # 勝率
            df[f'win_rate_last{window}'] = df.groupby('horse_id')['着順'].transform(
                lambda x: (x == 1).rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # 前走からの間隔（日数）の推定
        df['days_since_last'] = df.groupby('horse_id')['date'].diff().dt.days
        df['days_since_last'] = df['days_since_last'].fillna(60)  # デフォルト60日
        
        # 休養カテゴリ
        df['is_fresh'] = (df['days_since_last'] > 90).astype(int)
        df['is_short_rest'] = (df['days_since_last'] < 14).astype(int)
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量を追加"""
        self.logger.info("Adding interaction features...")
        df = df.copy()
        
        # 距離と枠番の交互作用
        df['distance_post_interaction'] = df['距離'] * df['枠番'] / 1000
        
        # 人気と騎手勝率の交互作用
        if 'jockey_win_rate' in df.columns:
            df['popularity_jockey_interaction'] = df['人気'] * df['jockey_win_rate']
        
        # 体重変化と距離の交互作用
        df['weight_distance_interaction'] = df['体重変化'].abs() * df['距離'] / 1000
        
        return df
    
    def add_all_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """すべての特徴量を高速に追加"""
        # 基本特徴量
        df = self.add_basic_features(df)
        
        # ローリング特徴量（計算コストが高いので小さいウィンドウのみ）
        df = self.add_rolling_features(df, window_sizes=[3, 5])
        
        # 交互作用
        df = self.add_interaction_features(df)
        
        return df