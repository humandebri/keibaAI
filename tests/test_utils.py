#!/usr/bin/env python3
"""
ユーティリティ関数のテスト
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
import json
from pathlib import Path

from src.core.utils import DataLoader, FeatureProcessor, ModelManager, setup_logger


class TestDataLoader:
    """データローダーのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        loader = DataLoader()
        assert loader.logger is not None
    
    def test_save_and_load_csv(self, sample_race_data, temp_directory):
        """CSV保存・読み込みのテスト"""
        loader = DataLoader()
        
        # CSV保存
        csv_path = temp_directory / "test_data.csv"
        loader.save_data(sample_race_data, csv_path)
        
        # ファイルが作成されていることを確認
        assert csv_path.exists()
        
        # データ読み込み
        loaded_data = pd.read_csv(csv_path)
        
        # データの一致確認
        assert len(loaded_data) == len(sample_race_data)
        assert list(loaded_data.columns) == list(sample_race_data.columns)
    
    def test_save_and_load_excel(self, sample_race_data, temp_directory):
        """Excel保存・読み込みのテスト"""
        loader = DataLoader()
        
        # Excel保存
        excel_path = temp_directory / "test_data.xlsx"
        loader.save_data(sample_race_data, excel_path)
        
        # ファイルが作成されていることを確認
        assert excel_path.exists()
        
        # データ読み込み
        loaded_data = pd.read_excel(excel_path)
        
        # データの一致確認
        assert len(loaded_data) == len(sample_race_data)


class TestFeatureProcessor:
    """特徴量プロセッサーのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        processor = FeatureProcessor()
        assert processor.feature_engine is not None
        assert processor.logger is not None
    
    def test_prepare_basic_features(self, sample_race_data):
        """基本特徴量準備のテスト"""
        processor = FeatureProcessor()
        result = processor.prepare_basic_features(sample_race_data)
        
        # 元のデータより特徴量が増加していることを確認
        assert len(result.columns) > len(sample_race_data.columns)
        
        # 着順が数値型であることを確認
        assert pd.api.types.is_numeric_dtype(result['着順'])
        
        # race_idが生成されていることを確認（元データにない場合）
        if 'race_id' not in sample_race_data.columns:
            assert 'race_id' in result.columns
    
    def test_create_target_variables(self, sample_race_data):
        """ターゲット変数作成のテスト"""
        processor = FeatureProcessor()
        result = processor.create_target_variables(sample_race_data)
        
        # ターゲット変数が作成されていることを確認
        assert 'is_win' in result.columns
        assert 'is_place' in result.columns
        assert 'is_exacta' in result.columns
        
        # データ型の確認
        assert result['is_win'].dtype == 'int64'
        assert result['is_place'].dtype == 'int64'
        assert result['is_exacta'].dtype == 'int64'
    
    def test_get_feature_columns(self, sample_race_data):
        """特徴量カラム取得のテスト"""
        processor = FeatureProcessor()
        
        # 特徴量を準備
        processed_data = processor.prepare_basic_features(sample_race_data)
        
        # 特徴量カラム取得
        feature_cols = processor.get_feature_columns(processed_data)
        
        # 特徴量が取得されていることを確認
        assert len(feature_cols) > 0
        assert isinstance(feature_cols, list)
        
        # 除外カラムのテスト
        exclude_cols = ['着順', 'race_id']
        feature_cols_filtered = processor.get_feature_columns(
            processed_data, exclude_cols=exclude_cols
        )
        
        for col in exclude_cols:
            if col in processed_data.columns:
                assert col not in feature_cols_filtered


class TestModelManager:
    """モデルマネージャーのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        manager = ModelManager()
        assert manager.logger is not None
    
    def test_save_and_load_models(self, mock_model_data, temp_directory):
        """モデル保存・読み込みのテスト"""
        manager = ModelManager()
        
        # モデル保存
        model_path = temp_directory / "test_model.pkl"
        manager.save_models(mock_model_data, model_path)
        
        # ファイルが作成されていることを確認
        assert model_path.exists()
        
        # モデル読み込み
        loaded_models = manager.load_models(model_path)
        
        # データの一致確認
        assert loaded_models == mock_model_data
    
    def test_save_and_load_model_info(self, temp_directory):
        """モデル情報保存・読み込みのテスト"""
        manager = ModelManager()
        
        model_info = {
            'model_type': 'ensemble',
            'training_date': '2024-01-01',
            'performance_metrics': {
                'auc': 0.85,
                'accuracy': 0.75
            },
            'feature_count': 134
        }
        
        # モデル情報保存
        info_path = temp_directory / "model_info.json"
        manager.save_model_info(model_info, info_path)
        
        # ファイルが作成されていることを確認
        assert info_path.exists()
        
        # モデル情報読み込み
        loaded_info = manager.load_model_info(info_path)
        
        # データの一致確認
        assert loaded_info == model_info
    
    def test_load_nonexistent_model(self, temp_directory):
        """存在しないモデルの読み込みエラーテスト"""
        manager = ModelManager()
        
        nonexistent_path = temp_directory / "nonexistent.pkl"
        
        with pytest.raises(FileNotFoundError):
            manager.load_models(nonexistent_path)
    
    def test_load_nonexistent_model_info(self, temp_directory):
        """存在しないモデル情報の読み込みエラーテスト"""
        manager = ModelManager()
        
        nonexistent_path = temp_directory / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            manager.load_model_info(nonexistent_path)


class TestSetupLogger:
    """ロガー設定のテスト"""
    
    def test_setup_logger_default(self):
        """デフォルトロガー設定のテスト"""
        logger = setup_logger("test_logger")
        
        # ロガーが作成されていることを確認
        assert logger.name == "test_logger"
        assert logger.level == 20  # logging.INFO
        
        # ハンドラーが設定されていることを確認
        assert len(logger.handlers) > 0
    
    def test_setup_logger_custom_level(self):
        """カスタムレベルロガー設定のテスト"""
        import logging
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    def test_logger_reuse(self):
        """同名ロガーの再利用テスト"""
        logger1 = setup_logger("reuse_test")
        logger2 = setup_logger("reuse_test")
        
        # 同じロガーインスタンスが返されることを確認
        assert logger1 is logger2
        
        # ハンドラーが重複しないことを確認
        handler_count = len(logger1.handlers)
        logger3 = setup_logger("reuse_test")
        assert len(logger3.handlers) == handler_count


class TestUtilityFunctions:
    """その他のユーティリティ関数のテスト"""
    
    def test_calculate_statistics(self):
        """統計計算のテスト"""
        from src.core.utils import calculate_statistics
        
        # テストデータ
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        stats = calculate_statistics(series)
        
        # 統計値の確認
        assert stats['mean'] == 5.5
        assert stats['median'] == 5.5
        assert stats['min'] == 1
        assert stats['max'] == 10
        assert stats['q1'] == 3.25
        assert stats['q3'] == 7.75
        assert abs(stats['std'] - 3.0276503540974917) < 1e-10
    
    def test_format_currency(self):
        """通貨フォーマットのテスト"""
        from src.core.utils import format_currency
        
        # 各種金額のフォーマット確認
        assert format_currency(1000) == "¥1,000"
        assert format_currency(1000000) == "¥1,000,000"
        assert format_currency(1234567.89) == "¥1,234,568"  # 小数点以下は四捨五入
        assert format_currency(0) == "¥0"
    
    def test_calculate_return_metrics(self):
        """リターン指標計算のテスト"""
        from src.core.utils import calculate_return_metrics
        
        # テストケース1: 1年で倍になった場合
        metrics = calculate_return_metrics(1000000, 2000000, 1)
        
        assert abs(metrics['total_return'] - 1.0) < 1e-10  # 100%リターン
        assert abs(metrics['annual_return'] - 1.0) < 1e-10  # 年率100%
        assert metrics['profit_loss'] == 1000000
        
        # テストケース2: 2年で50%増加の場合
        metrics = calculate_return_metrics(1000000, 1500000, 2)
        
        assert abs(metrics['total_return'] - 0.5) < 1e-10  # 50%リターン
        expected_annual = (1.5) ** (1/2) - 1  # 年率換算
        assert abs(metrics['annual_return'] - expected_annual) < 1e-10
        assert metrics['profit_loss'] == 500000