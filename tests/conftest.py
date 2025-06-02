#!/usr/bin/env python3
"""
pytest設定とフィクスチャ
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

# プロジェクトルートをパスに追加
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config, ScrapingConfig, ModelConfig, BacktestConfig, DataConfig
from src.core.utils import DataLoader, FeatureProcessor, ModelManager
from src.features.unified_features import UnifiedFeatureEngine


@pytest.fixture
def sample_race_data():
    """サンプル競馬データを生成"""
    np.random.seed(42)
    
    data = {
        'race_id': ['202401010101'] * 10,
        '馬名': [f'テストホース{i}' for i in range(1, 11)],
        '馬番': list(range(1, 11)),
        '着順': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        '人気': [1, 3, 2, 5, 4, 8, 6, 7, 10, 9],
        'オッズ': [1.5, 3.2, 2.8, 8.1, 6.5, 15.2, 12.3, 18.9, 45.6, 32.1],
        '斤量': [57, 56, 55, 54, 57, 56, 54, 55, 56, 57],
        '性': ['牡', '牝', '牡', 'セ', '牡', '牝', '牡', '牝', 'セ', '牡'],
        '年齢': [4, 3, 5, 6, 4, 3, 7, 4, 5, 6],
        '体重': ['480(+2)', '456(-3)', '502(+5)', '478(0)', '492(-1)', 
                '445(+4)', '510(-2)', '467(+1)', '489(+3)', '501(-1)'],
        '体重変化': [2, -3, 5, 0, -1, 4, -2, 1, 3, -1],
        '騎手': [f'騎手{i}' for i in range(1, 11)],
        '調教師': [f'調教師{i}' for i in range(1, 6)] * 2,
        '芝・ダート': ['芝'] * 10,
        '距離': [2000] * 10,
        '馬場': ['良'] * 8 + ['稍'] * 2,
        '天気': ['晴'] * 6 + ['曇'] * 4,
        '場名': ['東京'] * 10,
        '出走頭数': [10] * 10,
        'year': [2024] * 10,
        'payout_data': ['{}'] * 10  # 空の配当データ
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """テスト用設定を生成"""
    config = Config.__new__(Config)
    config.scraping = ScrapingConfig(start_year=2024, end_year=2024, max_workers=1)
    config.model = ModelConfig(test_size=0.3, n_trials=5)
    config.backtest = BacktestConfig(initial_capital=100000, betting_fraction=0.01)
    config.data = DataConfig()
    
    return config


@pytest.fixture
def temp_directory():
    """テンポラリディレクトリを作成"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def feature_engine():
    """統一特徴量エンジンを提供"""
    return UnifiedFeatureEngine()


@pytest.fixture
def data_loader():
    """データローダーを提供"""
    return DataLoader()


@pytest.fixture
def feature_processor():
    """特徴量プロセッサーを提供"""
    return FeatureProcessor()


@pytest.fixture
def model_manager():
    """モデルマネージャーを提供"""
    return ModelManager()


@pytest.fixture
def mock_model_data():
    """モックモデルデータを生成"""
    return {
        'lightgbm': 'mock_lgb_model',
        'xgboost': 'mock_xgb_model',
        'feature_cols': ['feature_1', 'feature_2', 'feature_3']
    }


class MockModel:
    """テスト用モックモデル"""
    
    def __init__(self, predictions=None):
        self.predictions = predictions or [1, 2, 3, 4, 5]
    
    def predict(self, X):
        """予測を実行"""
        return np.array(self.predictions[:len(X)])
    
    def save_model(self, path):
        """モデル保存のモック"""
        pass


@pytest.fixture
def mock_model():
    """モックモデルを提供"""
    return MockModel()


@pytest.fixture
def mock_ensemble_predictor():
    """モックアンサンブル予測器を提供"""
    class MockEnsemblePredictor:
        def __init__(self):
            self.models = {
                'lightgbm': MockModel([1.2, 2.1, 3.5, 4.2, 5.1]),
                'xgboost': MockModel([1.1, 2.3, 3.2, 4.5, 5.0])
            }
        
        def predict(self, data, feature_cols):
            # 簡単な平均予測
            return np.array([2.5, 3.1, 1.8, 4.2, 5.5])
        
        def train(self, data, feature_cols, target_col, race_id_col):
            return self.models
    
    return MockEnsemblePredictor()


# テストデータのパス
TEST_DATA_DIR = project_root / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)