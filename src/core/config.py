"""
統一設定モジュール
プロジェクト全体の設定を一元管理
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

# プロジェクトルートの定義
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ENCODED_DIR = PROJECT_ROOT / "encoded"

# ディレクトリの作成
for dir_path in [DATA_DIR, CONFIG_DIR, MODELS_DIR, RESULTS_DIR, ENCODED_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class ScrapingConfig:
    """スクレイピング設定"""
    start_year: int = 2014
    end_year: int = 2024
    max_workers: int = 3
    timeout: int = 120
    retry_count: int = 3
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            ]


@dataclass
class ModelConfig:
    """モデル訓練設定"""
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 100  # Optuna trials
    
    # LightGBM parameters
    lgb_params: Dict = None
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    initial_capital: float = 1_000_000
    betting_fraction: float = 0.005
    ev_threshold: float = 1.0
    monthly_stop_loss: float = 0.1
    
    # 戦略別設定
    min_odds_high_value: float = 10.0
    confidence_threshold: float = 0.65


@dataclass
class DataConfig:
    """データ処理設定"""
    # カラム名マッピング（日本語→英語）
    column_mapping: Dict[str, str] = None
    
    # カテゴリカル変数
    categorical_columns: List[str] = None
    
    # 数値変数
    numeric_columns: List[str] = None
    
    # 場所コードマッピング
    place_dict: Dict[str, str] = None
    
    def __post_init__(self):
        if self.column_mapping is None:
            self.column_mapping = {
                '着順': 'finish_position',
                '枠番': 'gate_number',
                '馬番': 'horse_number',
                '馬': 'horse_name',
                '性': 'gender',
                '齢': 'age',
                '斤量': 'weight_carried',
                '騎手': 'jockey',
                'タイム': 'time',
                '着差': 'margin',
                '人気': 'popularity',
                'オッズ': 'odds',
                '体重': 'horse_weight',
                '体重変化': 'weight_change',
                '調教師': 'trainer',
                '馬主': 'owner',
                '賞金': 'prize_money',
                '芝・ダート': 'track_type',
                '距離': 'distance',
                '馬場': 'track_condition',
                '天気': 'weather',
                '場名': 'venue'
            }
        
        if self.categorical_columns is None:
            self.categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        
        if self.numeric_columns is None:
            self.numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        
        if self.place_dict is None:
            self.place_dict = {
                '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                '阪神': '09', '小倉': '10'
            }


class Config:
    """統合設定クラス"""
    def __init__(self, config_file: Optional[Path] = None):
        self.scraping = ScrapingConfig()
        self.model = ModelConfig()
        self.backtest = BacktestConfig()
        self.data = DataConfig()
        
        # 設定ファイルから読み込み
        if config_file and config_file.exists():
            self.load_from_yaml(config_file)
        elif (CONFIG_DIR / "config.yaml").exists():
            self.load_from_yaml(CONFIG_DIR / "config.yaml")
    
    def load_from_yaml(self, config_file: Path):
        """YAMLファイルから設定を読み込み"""
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # スクレイピング設定
        if 'scraping' in data:
            for key, value in data['scraping'].items():
                if hasattr(self.scraping, key):
                    setattr(self.scraping, key, value)
        
        # モデル設定
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # バックテスト設定
        if 'backtest' in data:
            for key, value in data['backtest'].items():
                if hasattr(self.backtest, key):
                    setattr(self.backtest, key, value)
    
    def save_to_yaml(self, config_file: Path):
        """設定をYAMLファイルに保存"""
        data = {
            'scraping': {
                'start_year': self.scraping.start_year,
                'end_year': self.scraping.end_year,
                'max_workers': self.scraping.max_workers,
                'timeout': self.scraping.timeout,
                'retry_count': self.scraping.retry_count
            },
            'model': {
                'test_size': self.model.test_size,
                'random_state': self.model.random_state,
                'n_trials': self.model.n_trials
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'betting_fraction': self.backtest.betting_fraction,
                'ev_threshold': self.backtest.ev_threshold,
                'monthly_stop_loss': self.backtest.monthly_stop_loss
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)


# シングルトンインスタンス
config = Config()