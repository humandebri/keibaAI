#!/usr/bin/env python3
"""
統一システムのテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.core.unified_system import UnifiedKeibaAISystem, SystemMode, SystemStatus
from src.core.config import Config


class TestSystemMode:
    """システムモードのテスト"""
    
    def test_system_mode_values(self):
        """システムモードの値確認"""
        assert SystemMode.BACKTEST.value == "backtest"
        assert SystemMode.PAPER_TRADING.value == "paper_trading"
        assert SystemMode.LIVE_TRADING.value == "live_trading"
        assert SystemMode.RESEARCH.value == "research"


class TestSystemStatus:
    """システム状態のテスト"""
    
    def test_system_status_initialization(self):
        """システム状態の初期化テスト"""
        status = SystemStatus(mode=SystemMode.BACKTEST)
        
        assert status.mode == SystemMode.BACKTEST
        assert status.is_running is False
        assert status.start_time is None
        assert status.total_trades == 0
        assert status.daily_pnl == 0.0
        assert status.total_pnl == 0.0
        assert status.last_update is None


class TestUnifiedKeibaAISystem:
    """統一競馬AIシステムのテスト"""
    
    def test_initialization_default(self):
        """デフォルト初期化のテスト"""
        # バリデーションエラーを避けるためMockを使用
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            
            assert system.mode == SystemMode.BACKTEST
            assert system.status.mode == SystemMode.BACKTEST
            assert system.data_loader is not None
            assert system.feature_processor is not None
            assert system.model_manager is not None
    
    def test_initialization_with_config(self, sample_config):
        """設定付き初期化のテスト"""
        system = UnifiedKeibaAISystem(config=sample_config, mode=SystemMode.RESEARCH)
        
        assert system.mode == SystemMode.RESEARCH
        assert system.config == sample_config
        assert system.status.mode == SystemMode.RESEARCH
    
    def test_initialization_with_dict_config(self):
        """辞書設定での初期化テスト"""
        config_dict = {
            'model': {'test_size': 0.25, 'n_trials': 50},
            'backtest': {'initial_capital': 500000}
        }
        
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.from_dict.return_value = mock_config
            
            system = UnifiedKeibaAISystem(config=config_dict)
            
            mock_config_class.from_dict.assert_called_once_with(config_dict)
            assert system.config == mock_config
    
    @patch('src.core.unified_system.DataLoader')
    def test_load_data(self, mock_data_loader_class, sample_race_data):
        """データ読み込みのテスト"""
        # MockのDataLoaderを設定
        mock_loader = Mock()
        mock_loader.load_race_data.return_value = sample_race_data
        mock_data_loader_class.return_value = mock_loader
        
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            result = system.load_data(years=[2024], use_payout_data=True)
            
            # DataLoaderのメソッドが正しく呼ばれたことを確認
            mock_loader.load_race_data.assert_called_once_with(
                years=[2024], use_payout_data=True
            )
            
            # データが設定されていることを確認
            assert system.data is not None
            pd.testing.assert_frame_equal(result, sample_race_data)
    
    @patch('src.core.unified_system.FeatureProcessor')
    def test_prepare_features(self, mock_processor_class, sample_race_data):
        """特徴量準備のテスト"""
        # MockのFeatureProcessorを設定
        mock_processor = Mock()
        mock_processor.prepare_basic_features.return_value = sample_race_data
        mock_processor.create_target_variables.return_value = sample_race_data
        mock_processor.get_feature_columns.return_value = ['feature1', 'feature2']
        mock_processor_class.return_value = mock_processor
        
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            system.data = sample_race_data
            
            result = system.prepare_features()
            
            # FeatureProcessorのメソッドが呼ばれたことを確認
            mock_processor.prepare_basic_features.assert_called_once()
            mock_processor.create_target_variables.assert_called_once()
            mock_processor.get_feature_columns.assert_called_once()
            
            # 特徴量カラムが設定されていることを確認
            assert system.feature_cols == ['feature1', 'feature2']
    
    def test_set_strategy(self):
        """戦略設定のテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            mock_strategy = Mock()
            
            system.set_strategy(mock_strategy)
            
            assert system.strategy == mock_strategy
    
    def test_get_status(self):
        """ステータス取得のテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem(mode=SystemMode.PAPER_TRADING)
            status = system.get_status()
            
            assert isinstance(status, SystemStatus)
            assert status.mode == SystemMode.PAPER_TRADING
    
    def test_get_results(self):
        """結果取得のテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            
            # 初期状態では空の辞書
            results = system.get_results()
            assert results == {}
            
            # 結果を設定
            test_results = {'total_return': 0.15, 'win_rate': 0.25}
            system.results = test_results
            
            results = system.get_results()
            assert results == test_results
    
    def test_export_results(self, temp_directory):
        """結果エクスポートのテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            system.results = {
                'total_return': 0.15,
                'win_rate': 0.25,
                'total_trades': 100
            }
            
            output_path = temp_directory / "test_results.json"
            system.export_results(output_path)
            
            # ファイルが作成されていることを確認
            assert output_path.exists()
            
            # ファイル内容を確認
            import json
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results['total_return'] == 0.15
            assert loaded_results['win_rate'] == 0.25
            assert loaded_results['total_trades'] == 100
    
    @patch('src.core.unified_system.EnsembleRacePredictor')
    def test_train_models(self, mock_predictor_class, sample_race_data):
        """モデル訓練のテスト"""
        # MockのEnsembleRacePredictorを設定
        mock_predictor = Mock()
        mock_models = {'lightgbm': 'mock_model', 'xgboost': 'mock_model2'}
        mock_predictor.train.return_value = mock_models
        mock_predictor_class.return_value = mock_predictor
        
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.model = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            system.data = sample_race_data
            system.feature_cols = ['feature1', 'feature2']
            
            with patch.object(system.model_manager, 'save_models') as mock_save:
                result = system.train_models(save_models=True)
                
                # 予測器の訓練が呼ばれたことを確認
                mock_predictor.train.assert_called_once_with(
                    data=sample_race_data,
                    feature_cols=['feature1', 'feature2'],
                    target_col='着順',
                    race_id_col='race_id'
                )
                
                # モデルが保存されたことを確認
                mock_save.assert_called_once()
                
                # 結果の確認
                assert result == mock_models
                assert system.models == mock_models
    
    def test_predict_without_predictor(self, sample_race_data):
        """予測器なしでの予測エラーテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            
            with pytest.raises(ValueError, match="No predictor available"):
                system.predict(sample_race_data)
    
    def test_predict_with_predictor(self, sample_race_data):
        """予測器ありでの予測テスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            
            # MockのPredicatorを設定
            mock_predictor = Mock()
            mock_predictor.predict.return_value = [1.5, 2.3, 3.1, 4.2, 5.0, 6.1, 7.5, 8.2, 9.3, 10.1]
            system.predictor = mock_predictor
            system.feature_cols = ['feature1', 'feature2']
            
            result = system.predict(sample_race_data)
            
            # 予測が呼ばれたことを確認
            mock_predictor.predict.assert_called_once_with(
                sample_race_data, ['feature1', 'feature2']
            )
            
            # 予測結果が追加されていることを確認
            assert 'predicted_rank' in result.columns
    
    def test_load_models(self, temp_directory):
        """モデル読み込みのテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.model = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            
            # テスト用モデルファイルを作成
            test_models = {'lightgbm': 'test_model', 'feature_cols': ['f1', 'f2']}
            model_path = temp_directory / "test_model.pkl"
            
            with patch.object(system.model_manager, 'load_models') as mock_load:
                mock_load.return_value = test_models
                
                result = system.load_models(model_path)
                
                # ModelManagerのload_modelsが呼ばれたことを確認
                mock_load.assert_called_once_with(model_path)
                
                # モデルが設定されていることを確認
                assert system.models == test_models
                assert result == test_models
                assert system.predictor is not None
    
    def test_stop_realtime(self):
        """リアルタイム停止のテスト"""
        with patch('src.core.unified_system.Config') as mock_config_class:
            mock_config = Mock()
            mock_config_class.load_default.return_value = mock_config
            
            system = UnifiedKeibaAISystem()
            system.status.is_running = True
            
            system.stop_realtime()
            
            assert system.status.is_running is False