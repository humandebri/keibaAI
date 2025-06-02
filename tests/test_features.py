#!/usr/bin/env python3
"""
特徴量エンジニアリングのテスト
"""

import pytest
import pandas as pd
import numpy as np

from src.features.unified_features import (
    UnifiedFeatureEngine, BasicFeatureBuilder, TrackFeatureBuilder,
    HistoricalFeatureBuilder, PayoutFeatureBuilder
)


class TestBasicFeatureBuilder:
    """基本特徴量ビルダーのテスト"""
    
    def test_popularity_odds_features(self, sample_race_data):
        """人気・オッズ特徴量のテスト"""
        builder = BasicFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 生成される特徴量の確認
        assert 'オッズ_numeric' in result.columns
        assert 'popularity_odds_ratio' in result.columns
        assert 'is_favorite' in result.columns
        assert 'is_longshot' in result.columns
        assert 'odds_rank' in result.columns
        
        # 値の妥当性確認
        assert result['is_favorite'].sum() == 3  # 人気1-3位
        assert result['is_longshot'].sum() == 2  # 人気10位以上
        assert result['オッズ_numeric'].min() == 1.5
        assert result['オッズ_numeric'].max() == 45.6
    
    def test_draw_features(self, sample_race_data):
        """馬番・枠番特徴量のテスト"""
        builder = BasicFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 枠番の確認
        assert '枠番' in result.columns
        assert result['枠番'].min() == 1
        assert result['枠番'].max() == 5  # 馬番10頭の場合
        
        # ポジション特徴量の確認
        assert 'is_inside_draw' in result.columns
        assert 'is_outside_draw' in result.columns
        assert result['is_inside_draw'].sum() == 4  # 馬番1-4
        assert result['is_outside_draw'].sum() == 0  # 馬番12以上なし
    
    def test_weight_features(self, sample_race_data):
        """斤量特徴量のテスト"""
        builder = BasicFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 重量カテゴリの確認
        assert 'weight_heavy' in result.columns
        assert 'weight_light' in result.columns
        assert 'weight_medium' in result.columns
        assert 'weight_norm' in result.columns
        
        # カテゴリの排他性確認
        total_cats = (result['weight_heavy'] + 
                     result['weight_light'] + 
                     result['weight_medium'])
        assert total_cats.sum() == len(result)
    
    def test_horse_weight_features(self, sample_race_data):
        """馬体重特徴量のテスト"""
        builder = BasicFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 体重の数値化確認
        assert '体重_numeric' in result.columns
        assert result['体重_numeric'].min() >= 400  # 現実的な最小値
        assert result['体重_numeric'].max() <= 600  # 現実的な最大値
        
        # 体重変化特徴量の確認
        assert 'weight_change_abs' in result.columns
        assert 'weight_increased' in result.columns
        assert 'weight_decreased' in result.columns
        assert 'weight_stable' in result.columns


class TestTrackFeatureBuilder:
    """コース・馬場特徴量ビルダーのテスト"""
    
    def test_track_condition_features(self, sample_race_data):
        """馬場状態特徴量のテスト"""
        builder = TrackFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 馬場状態の数値化確認
        assert 'track_condition_numeric' in result.columns
        assert 'track_moisture_index' in result.columns
        assert 'track_cushion_value' in result.columns
        
        # 馬場状態フラグの確認
        assert 'is_good_track' in result.columns
        assert 'is_heavy_track' in result.columns
        assert 'is_soft_track' in result.columns
        
        # 値の妥当性確認
        assert result['track_condition_numeric'].min() >= 0.4
        assert result['track_condition_numeric'].max() <= 1.0
    
    def test_surface_features(self, sample_race_data):
        """コース種別特徴量のテスト"""
        builder = TrackFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # コース種別エンコーディング確認
        assert 'surface_encoded' in result.columns
        assert 'is_turf' in result.columns
        assert 'is_dirt' in result.columns
        assert 'is_jump' in result.columns
        
        # 全て芝コースの場合
        assert result['is_turf'].sum() == len(result)
        assert result['is_dirt'].sum() == 0
    
    def test_distance_features(self, sample_race_data):
        """距離特徴量のテスト"""
        builder = TrackFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 距離カテゴリの確認
        assert 'is_sprint' in result.columns
        assert 'is_mile' in result.columns
        assert 'is_middle' in result.columns
        assert 'is_long' in result.columns
        
        # 2000mなのでmiddleカテゴリ
        assert result['is_middle'].sum() == len(result)
        assert result['is_sprint'].sum() == 0


class TestHistoricalFeatureBuilder:
    """過去成績特徴量ビルダーのテスト"""
    
    def test_horse_history_features(self, sample_race_data):
        """馬の過去成績特徴量のテスト"""
        builder = HistoricalFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 馬の成績特徴量の確認
        assert 'horse_win_rate' in result.columns
        assert 'horse_place_rate' in result.columns
        assert 'total_races' in result.columns
        
        # 値の妥当性確認
        assert result['horse_win_rate'].min() >= 0
        assert result['horse_win_rate'].max() <= 1
        assert result['horse_place_rate'].min() >= 0
        assert result['horse_place_rate'].max() <= 1
    
    def test_jockey_history_features(self, sample_race_data):
        """騎手の過去成績特徴量のテスト"""
        builder = HistoricalFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 騎手の成績特徴量の確認
        assert 'jockey_win_rate' in result.columns
        assert 'jockey_place_rate' in result.columns
        assert 'jockey_total_races' in result.columns
        
        # 値の妥当性確認
        assert result['jockey_win_rate'].min() >= 0
        assert result['jockey_win_rate'].max() <= 1
    
    def test_trainer_history_features(self, sample_race_data):
        """調教師の過去成績特徴量のテスト"""
        builder = HistoricalFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 調教師の成績特徴量の確認
        assert 'trainer_win_rate' in result.columns
        assert 'trainer_place_rate' in result.columns
        assert 'trainer_total_races' in result.columns


class TestPayoutFeatureBuilder:
    """配当特徴量ビルダーのテスト"""
    
    def test_empty_payout_data(self, sample_race_data):
        """空配当データのテスト"""
        builder = PayoutFeatureBuilder()
        result = builder.build(sample_race_data)
        
        # 配当特徴量の確認
        assert '単勝最高配当' in result.columns
        assert '複勝最低配当' in result.columns
        assert '三連単配当' in result.columns
        assert '高配当レース' in result.columns
        
        # 空データなので全て0
        assert result['単勝最高配当'].sum() == 0
        assert result['高配当レース'].sum() == 0
    
    def test_valid_payout_data(self):
        """有効な配当データのテスト"""
        # 配当データ付きのサンプルデータ
        data = pd.DataFrame({
            'race_id': ['202401010101'] * 3,
            '馬名': ['Horse1', 'Horse2', 'Horse3'],
            'payout_data': [
                '{"win": {"1": 150}, "place": {"1": 110}, "trifecta": {"1-2-3": 2500}}',
                '{"win": {"2": 320}, "place": {"2": 180}}',
                '{}'
            ]
        })
        
        builder = PayoutFeatureBuilder()
        result = builder.build(data)
        
        # 配当値の確認
        assert result.loc[0, '単勝最高配当'] == 150
        assert result.loc[1, '単勝最高配当'] == 320
        assert result.loc[2, '単勝最高配当'] == 0
        
        # 高配当レース判定
        assert result.loc[1, '高配当レース'] == 0  # 320円は1000円未満
        assert result['高配当レース'].sum() == 0


class TestUnifiedFeatureEngine:
    """統一特徴量エンジンのテスト"""
    
    def test_build_all_features(self, sample_race_data):
        """全特徴量構築のテスト"""
        engine = UnifiedFeatureEngine()
        result = engine.build_all_features(sample_race_data)
        
        # 元のカラム数より増加していることを確認
        assert len(result.columns) > len(sample_race_data.columns)
        
        # 各ビルダーの特徴量が含まれていることを確認
        basic_features = ['オッズ_numeric', 'is_favorite', '枠番']
        track_features = ['track_condition_numeric', 'is_turf', 'is_middle']
        historical_features = ['horse_win_rate', 'jockey_win_rate', 'trainer_win_rate']
        payout_features = ['単勝最高配当', '高配当レース']
        
        for feature in basic_features + track_features + historical_features + payout_features:
            assert feature in result.columns, f"Feature {feature} not found"
    
    def test_create_target_variables(self, sample_race_data):
        """ターゲット変数作成のテスト"""
        engine = UnifiedFeatureEngine()
        result = engine.create_target_variables(sample_race_data)
        
        # ターゲット変数の確認
        assert 'is_win' in result.columns
        assert 'is_place' in result.columns
        assert 'is_exacta' in result.columns
        
        # 値の妥当性確認
        assert result['is_win'].sum() == 1  # 1着は1頭
        assert result['is_place'].sum() == 3  # 3着以内は3頭
        assert result['is_exacta'].sum() == 2  # 2着以内は2頭
    
    def test_get_feature_columns(self, sample_race_data):
        """特徴量カラム取得のテスト"""
        engine = UnifiedFeatureEngine()
        result = engine.build_all_features(sample_race_data)
        feature_cols = engine.get_feature_columns(result)
        
        # 特徴量カラムが返されることを確認
        assert len(feature_cols) > 0
        
        # 全ての特徴量が実際に存在することを確認
        for col in feature_cols:
            assert col in result.columns, f"Feature column {col} not found in data"
    
    def test_add_custom_builder(self, sample_race_data):
        """カスタムビルダー追加のテスト"""
        class CustomFeatureBuilder:
            def build(self, df):
                df = df.copy()
                df['custom_feature'] = 1
                return df
            
            def get_feature_names(self):
                return ['custom_feature']
        
        engine = UnifiedFeatureEngine()
        engine.add_builder(CustomFeatureBuilder())
        
        result = engine.build_all_features(sample_race_data)
        
        # カスタム特徴量が追加されていることを確認
        assert 'custom_feature' in result.columns
        assert (result['custom_feature'] == 1).all()