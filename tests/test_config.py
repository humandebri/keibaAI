#!/usr/bin/env python3
"""
設定クラスのテスト
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.core.config import (
    Config, ScrapingConfig, ModelConfig, BacktestConfig, DataConfig,
    ConfigValidationError, RangeValidator, ChoiceValidator, PathValidator
)


class TestConfigValidators:
    """設定バリデータのテスト"""
    
    def test_range_validator_valid(self):
        """範囲バリデータ - 正常値"""
        validator = RangeValidator(min_val=0, max_val=100)
        errors = validator.validate(50)
        assert len(errors) == 0
    
    def test_range_validator_invalid_min(self):
        """範囲バリデータ - 最小値未満"""
        validator = RangeValidator(min_val=10, max_val=100)
        errors = validator.validate(5)
        assert len(errors) == 1
        assert "below minimum" in errors[0]
    
    def test_range_validator_invalid_max(self):
        """範囲バリデータ - 最大値超過"""
        validator = RangeValidator(min_val=0, max_val=100)
        errors = validator.validate(150)
        assert len(errors) == 1
        assert "above maximum" in errors[0]
    
    def test_range_validator_non_numeric(self):
        """範囲バリデータ - 非数値"""
        validator = RangeValidator(min_val=0, max_val=100)
        errors = validator.validate("invalid")
        assert len(errors) == 1
        assert "must be numeric" in errors[0]
    
    def test_choice_validator_valid(self):
        """選択肢バリデータ - 正常値"""
        validator = ChoiceValidator(['A', 'B', 'C'])
        errors = validator.validate('B')
        assert len(errors) == 0
    
    def test_choice_validator_invalid(self):
        """選択肢バリデータ - 無効値"""
        validator = ChoiceValidator(['A', 'B', 'C'])
        errors = validator.validate('D')
        assert len(errors) == 1
        assert "not in allowed choices" in errors[0]


class TestScrapingConfig:
    """スクレイピング設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        config = ScrapingConfig()
        assert config.start_year == 2014
        assert config.end_year == 2024
        assert config.max_workers == 3
        assert config.timeout == 120
        assert config.retry_count == 3
        assert len(config.user_agents) > 0
    
    def test_validation_success(self):
        """バリデーション成功"""
        config = ScrapingConfig(
            start_year=2020,
            end_year=2024,
            max_workers=5,
            timeout=60,
            retry_count=2
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_invalid_year_range(self):
        """バリデーション失敗 - 年度範囲エラー"""
        config = ScrapingConfig(start_year=2025, end_year=2020)
        errors = config.validate()
        assert any("end_year must be >= start_year" in error for error in errors)
    
    def test_validation_invalid_workers(self):
        """バリデーション失敗 - ワーカー数エラー"""
        config = ScrapingConfig(max_workers=15)
        errors = config.validate()
        assert any("max_workers must be between" in error for error in errors)


class TestModelConfig:
    """モデル設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        config = ModelConfig()
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.n_trials == 100
        assert config.cv_folds == 5
        assert 'lightgbm' in config.ensemble_weights
    
    def test_validation_success(self):
        """バリデーション成功"""
        config = ModelConfig(
            test_size=0.3,
            n_trials=50,
            cv_folds=3,
            ensemble_weights={'lightgbm': 0.6, 'xgboost': 0.4}
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_invalid_test_size(self):
        """バリデーション失敗 - テストサイズエラー"""
        config = ModelConfig(test_size=1.5)
        errors = config.validate()
        assert any("test_size must be between" in error for error in errors)
    
    def test_validation_invalid_ensemble_weights(self):
        """バリデーション失敗 - アンサンブル重みエラー"""
        config = ModelConfig(ensemble_weights={'lightgbm': 0.3, 'xgboost': 0.5})
        errors = config.validate()
        assert any("must sum to 1.0" in error for error in errors)


class TestBacktestConfig:
    """バックテスト設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        config = BacktestConfig()
        assert config.initial_capital == 1_000_000
        assert config.betting_fraction == 0.005
        assert config.ev_threshold == 1.1
        assert config.kelly_fraction == 0.025
        assert 'trifecta' in config.bet_type_fractions
    
    def test_validation_success(self):
        """バリデーション成功"""
        config = BacktestConfig(
            initial_capital=500_000,
            betting_fraction=0.01,
            ev_threshold=1.2,
            kelly_fraction=0.05
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_invalid_capital(self):
        """バリデーション失敗 - 初期資金エラー"""
        config = BacktestConfig(initial_capital=-100)
        errors = config.validate()
        assert any("initial_capital must be positive" in error for error in errors)
    
    def test_validation_invalid_betting_fraction(self):
        """バリデーション失敗 - ベッティング割合エラー"""
        config = BacktestConfig(betting_fraction=0.2)
        errors = config.validate()
        assert any("betting_fraction must be between" in error for error in errors)


class TestDataConfig:
    """データ設定のテスト"""
    
    def test_default_values(self):
        """デフォルト値の確認"""
        config = DataConfig()
        assert '性' in config.categorical_columns
        assert '馬番' in config.numeric_columns
        assert '東京' in config.place_dict
        assert config.feature_engineering['basic_features'] is True
    
    def test_validation_success(self):
        """バリデーション成功"""
        config = DataConfig(
            categorical_columns=['性', '馬場'],
            numeric_columns=['馬番', '斤量'],
            place_dict={'東京': '05', '京都': '08'}
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_column_overlap(self):
        """バリデーション失敗 - カラム重複エラー"""
        config = DataConfig(
            categorical_columns=['性', '馬番'],
            numeric_columns=['馬番', '斤量']
        )
        errors = config.validate()
        assert any("appear in both categorical and numeric" in error for error in errors)


class TestConfig:
    """統合設定クラスのテスト"""
    
    def test_default_initialization(self):
        """デフォルト初期化"""
        config = Config.__new__(Config)  # バリデーションをスキップ
        config.scraping = ScrapingConfig()
        config.model = ModelConfig()
        config.backtest = BacktestConfig()
        config.data = DataConfig()
        
        assert isinstance(config.scraping, ScrapingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.backtest, BacktestConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_from_dict(self):
        """辞書からの設定作成"""
        config_dict = {
            'scraping': {
                'start_year': 2022,
                'max_workers': 2
            },
            'model': {
                'test_size': 0.25,
                'n_trials': 50
            }
        }
        
        config = Config.from_dict(config_dict)
        assert config.scraping.start_year == 2022
        assert config.scraping.max_workers == 2
        assert config.model.test_size == 0.25
        assert config.model.n_trials == 50
    
    def test_yaml_save_load(self):
        """YAML保存・読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = Path(f.name)
        
        try:
            # 設定作成
            config1 = Config.__new__(Config)
            config1.scraping = ScrapingConfig(start_year=2023, max_workers=5)
            config1.model = ModelConfig(test_size=0.3)
            config1.backtest = BacktestConfig(initial_capital=2_000_000)
            config1.data = DataConfig()
            
            # 保存
            config1.save_to_yaml(config_file)
            
            # 読み込み
            config2 = Config.load_from_file(config_file)
            
            # 検証
            assert config2.scraping.start_year == 2023
            assert config2.scraping.max_workers == 5
            assert config2.model.test_size == 0.3
            assert config2.backtest.initial_capital == 2_000_000
            
        finally:
            config_file.unlink(missing_ok=True)
    
    def test_to_dict(self):
        """辞書変換"""
        config = Config.__new__(Config)
        config.scraping = ScrapingConfig(start_year=2023)
        config.model = ModelConfig(test_size=0.3)
        config.backtest = BacktestConfig(initial_capital=500_000)
        config.data = DataConfig()
        
        config_dict = config.to_dict()
        
        assert config_dict['scraping']['start_year'] == 2023
        assert config_dict['model']['test_size'] == 0.3
        assert config_dict['backtest']['initial_capital'] == 500_000
        assert 'data' in config_dict