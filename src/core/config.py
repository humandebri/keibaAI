"""
統一設定モジュール
プロジェクト全体の設定を一元管理
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import yaml
from abc import ABC, abstractmethod

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


class ConfigValidationError(Exception):
    """設定バリデーションエラー"""
    pass


class ConfigValidator(ABC):
    """設定バリデータの基底クラス"""
    
    @abstractmethod
    def validate(self, config_data: Any) -> List[str]:
        """設定の検証を行い、エラーメッセージのリストを返す"""
        pass


class RangeValidator(ConfigValidator):
    """数値範囲バリデータ"""
    
    def __init__(self, min_val: float = None, max_val: float = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, config_data: Any) -> List[str]:
        errors = []
        if not isinstance(config_data, (int, float)):
            errors.append(f"Value must be numeric, got {type(config_data)}")
            return errors
        
        if self.min_val is not None and config_data < self.min_val:
            errors.append(f"Value {config_data} is below minimum {self.min_val}")
        
        if self.max_val is not None and config_data > self.max_val:
            errors.append(f"Value {config_data} is above maximum {self.max_val}")
        
        return errors


class ChoiceValidator(ConfigValidator):
    """選択肢バリデータ"""
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
    
    def validate(self, config_data: Any) -> List[str]:
        if config_data not in self.choices:
            return [f"Value {config_data} not in allowed choices: {self.choices}"]
        return []


class PathValidator(ConfigValidator):
    """パスバリデータ"""
    
    def __init__(self, must_exist: bool = False, create_if_missing: bool = False):
        self.must_exist = must_exist
        self.create_if_missing = create_if_missing
    
    def validate(self, config_data: Any) -> List[str]:
        errors = []
        
        try:
            path = Path(config_data)
        except Exception:
            return [f"Invalid path: {config_data}"]
        
        if self.must_exist and not path.exists():
            errors.append(f"Path does not exist: {path}")
        elif self.create_if_missing and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create path {path}: {e}")
        
        return errors


@dataclass
class ScrapingConfig:
    """スクレイピング設定"""
    start_year: int = 2014
    end_year: int = 2024
    max_workers: int = 3
    timeout: int = 120
    retry_count: int = 3
    min_delay: float = 1.0
    max_delay: float = 3.0
    user_agents: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.user_agents:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
            ]
    
    def validate(self) -> List[str]:
        """設定値の検証"""
        errors = []
        
        # 年度の検証
        if self.start_year < 2014 or self.start_year > 2030:
            errors.append("start_year must be between 2014 and 2030")
        if self.end_year < self.start_year:
            errors.append("end_year must be >= start_year")
        if self.end_year > 2030:
            errors.append("end_year must be <= 2030")
        
        # ワーカー数の検証
        if self.max_workers < 1 or self.max_workers > 10:
            errors.append("max_workers must be between 1 and 10")
        
        # タイムアウトの検証
        if self.timeout < 10 or self.timeout > 300:
            errors.append("timeout must be between 10 and 300 seconds")
        
        # リトライ回数の検証
        if self.retry_count < 1 or self.retry_count > 10:
            errors.append("retry_count must be between 1 and 10")
        
        # 遅延時間の検証
        if self.min_delay < 0.1 or self.min_delay > 10:
            errors.append("min_delay must be between 0.1 and 10 seconds")
        if self.max_delay < self.min_delay:
            errors.append("max_delay must be >= min_delay")
        
        return errors


@dataclass
class ModelConfig:
    """モデル訓練設定"""
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 100  # Optuna trials
    cv_folds: int = 5
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.lgb_params:
            self.lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'seed': 42
            }
        
        if not self.ensemble_weights:
            self.ensemble_weights = {
                'lightgbm': 0.4,
                'xgboost': 0.3,
                'random_forest': 0.15,
                'gradient_boosting': 0.15
            }
    
    def validate(self) -> List[str]:
        """設定値の検証"""
        errors = []
        
        # テストサイズの検証
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        # trial数の検証
        if self.n_trials < 10 or self.n_trials > 1000:
            errors.append("n_trials must be between 10 and 1000")
        
        # CV fold数の検証
        if self.cv_folds < 2 or self.cv_folds > 20:
            errors.append("cv_folds must be between 2 and 20")
        
        # アンサンブル重みの検証
        if self.ensemble_weights:
            total_weight = sum(self.ensemble_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"ensemble_weights must sum to 1.0, got {total_weight}")
            
            for model, weight in self.ensemble_weights.items():
                if weight < 0 or weight > 1:
                    errors.append(f"ensemble weight for {model} must be between 0 and 1")
        
        return errors


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    initial_capital: float = 1_000_000
    betting_fraction: float = 0.005
    ev_threshold: float = 1.1
    monthly_stop_loss: float = 0.15
    max_bet_per_race: float = 10_000
    max_daily_loss: float = 50_000
    kelly_fraction: float = 0.025
    
    # 戦略別設定
    min_odds_high_value: float = 5.0
    confidence_threshold: float = 0.65
    
    # 券種別設定
    bet_type_fractions: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.bet_type_fractions:
            self.bet_type_fractions = {
                'trifecta': 0.7,
                'quinella': 0.9,
                'wide': 1.0,
                'win': 1.0,
                'place': 1.0
            }
    
    def validate(self) -> List[str]:
        """設定値の検証"""
        errors = []
        
        # 初期資金の検証
        if self.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        
        # ベッティング割合の検証
        if self.betting_fraction <= 0 or self.betting_fraction > 0.1:
            errors.append("betting_fraction must be between 0 and 0.1 (10%)")
        
        # EV閾値の検証
        if self.ev_threshold < 1.0 or self.ev_threshold > 3.0:
            errors.append("ev_threshold must be between 1.0 and 3.0")
        
        # 停止損失の検証
        if self.monthly_stop_loss <= 0 or self.monthly_stop_loss > 0.5:
            errors.append("monthly_stop_loss must be between 0 and 0.5 (50%)")
        
        # Kelly割合の検証
        if self.kelly_fraction <= 0 or self.kelly_fraction > 0.1:
            errors.append("kelly_fraction must be between 0 and 0.1")
        
        # 券種別割合の検証
        for bet_type, fraction in self.bet_type_fractions.items():
            if fraction < 0 or fraction > 1:
                errors.append(f"bet_type_fraction for {bet_type} must be between 0 and 1")
        
        return errors


@dataclass
class PathConfig:
    """パス設定"""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    config_dir: Path = CONFIG_DIR
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    encoded_dir: Path = ENCODED_DIR
    logs_dir: Path = PROJECT_ROOT / "logs"
    cache_dir: Path = PROJECT_ROOT / "cache"
    
    def __post_init__(self):
        # 必要なディレクトリを作成
        for path in [self.data_dir, self.config_dir, self.models_dir, 
                    self.results_dir, self.encoded_dir, self.logs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """パス設定の検証"""
        errors = []
        
        # プロジェクトルートの検証
        if not self.project_root.exists():
            errors.append(f"Project root does not exist: {self.project_root}")
        
        # 書き込み権限の検証
        for path_name, path in [
            ("data_dir", self.data_dir),
            ("models_dir", self.models_dir),
            ("results_dir", self.results_dir),
            ("logs_dir", self.logs_dir)
        ]:
            if not os.access(path, os.W_OK):
                errors.append(f"No write permission for {path_name}: {path}")
        
        return errors


@dataclass
class DataConfig:
    """データ処理設定"""
    # カラム名マッピング（日本語→英語）
    column_mapping: Dict[str, str] = field(default_factory=dict)
    
    # カテゴリカル変数
    categorical_columns: List[str] = field(default_factory=list)
    
    # 数値変数
    numeric_columns: List[str] = field(default_factory=list)
    
    # 場所コードマッピング
    place_dict: Dict[str, str] = field(default_factory=dict)
    
    # 特徴量設定
    feature_engineering: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.column_mapping:
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
        
        if not self.categorical_columns:
            self.categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        
        if not self.numeric_columns:
            self.numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        
        if not self.place_dict:
            self.place_dict = {
                '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                '阪神': '09', '小倉': '10'
            }
        
        if not self.feature_engineering:
            self.feature_engineering = {
                'basic_features': True,
                'track_features': True,
                'historical_features': True,
                'payout_features': True,
                'interaction_features': True
            }
    
    def validate(self) -> List[str]:
        """データ設定の検証"""
        errors = []
        
        # カテゴリカルカラムと数値カラムの重複チェック
        overlap = set(self.categorical_columns) & set(self.numeric_columns)
        if overlap:
            errors.append(f"Columns appear in both categorical and numeric: {overlap}")
        
        # 場所辞書のバリデーション
        if len(self.place_dict) == 0:
            errors.append("place_dict cannot be empty")
        
        return errors


class Config:
    """統合設定クラス（バリデーション機能付き）"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.paths = PathConfig()
        self.scraping = ScrapingConfig()
        self.model = ModelConfig()
        self.backtest = BacktestConfig()
        self.data = DataConfig()
        self.logger = logging.getLogger(__name__)
        
        # 設定ファイルから読み込み
        if config_file and config_file.exists():
            self.load_from_yaml(config_file)
        elif (CONFIG_DIR / "config.yaml").exists():
            self.load_from_yaml(CONFIG_DIR / "config.yaml")
        
        # 初期化後にバリデーション実行
        self.validate_all()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """辞書から設定を作成"""
        config = cls()
        
        # 各セクションの設定を更新
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        return config
    
    @classmethod
    def load_default(cls) -> 'Config':
        """デフォルト設定を読み込み"""
        return cls()
    
    @classmethod
    def load_from_file(cls, config_file: Union[str, Path]) -> 'Config':
        """ファイルから設定を読み込み"""
        return cls(Path(config_file))
    
    def load_from_yaml(self, config_file: Path):
        """YAMLファイルから設定を読み込み"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigValidationError(f"Failed to load config file {config_file}: {e}")
        
        if not isinstance(data, dict):
            raise ConfigValidationError("Config file must contain a dictionary")
        
        # 各セクションの設定を更新
        sections = {
            'paths': self.paths,
            'scraping': self.scraping,
            'model': self.model,
            'backtest': self.backtest,
            'data': self.data
        }
        
        for section_name, section_obj in sections.items():
            if section_name in data and isinstance(data[section_name], dict):
                for key, value in data[section_name].items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        self.logger.warning(f"Unknown config key: {section_name}.{key}")
    
    def save_to_yaml(self, config_file: Path):
        """設定をYAMLファイルに保存"""
        data = {
            'scraping': {
                'start_year': self.scraping.start_year,
                'end_year': self.scraping.end_year,
                'max_workers': self.scraping.max_workers,
                'timeout': self.scraping.timeout,
                'retry_count': self.scraping.retry_count,
                'min_delay': self.scraping.min_delay,
                'max_delay': self.scraping.max_delay
            },
            'model': {
                'test_size': self.model.test_size,
                'random_state': self.model.random_state,
                'n_trials': self.model.n_trials,
                'cv_folds': self.model.cv_folds,
                'ensemble_weights': self.model.ensemble_weights,
                'lgb_params': self.model.lgb_params
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'betting_fraction': self.backtest.betting_fraction,
                'ev_threshold': self.backtest.ev_threshold,
                'monthly_stop_loss': self.backtest.monthly_stop_loss,
                'max_bet_per_race': self.backtest.max_bet_per_race,
                'max_daily_loss': self.backtest.max_daily_loss,
                'kelly_fraction': self.backtest.kelly_fraction,
                'bet_type_fractions': self.backtest.bet_type_fractions
            },
            'data': {
                'categorical_columns': self.data.categorical_columns,
                'numeric_columns': self.data.numeric_columns,
                'feature_engineering': self.data.feature_engineering
            }
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, indent=2)
        
        self.logger.info(f"Config saved to {config_file}")
    
    def validate_all(self) -> None:
        """全ての設定をバリデーション"""
        all_errors = []
        
        # 各セクションのバリデーション
        sections = [
            ('paths', self.paths),
            ('scraping', self.scraping),
            ('model', self.model),
            ('backtest', self.backtest),
            ('data', self.data)
        ]
        
        for section_name, section_obj in sections:
            if hasattr(section_obj, 'validate'):
                try:
                    errors = section_obj.validate()
                    if errors:
                        all_errors.extend([f"{section_name}.{error}" for error in errors])
                except Exception as e:
                    all_errors.append(f"{section_name}: Validation failed - {e}")
        
        # エラーがある場合は例外を発生
        if all_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in all_errors)
            raise ConfigValidationError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書として返す"""
        return {
            'scraping': {
                'start_year': self.scraping.start_year,
                'end_year': self.scraping.end_year,
                'max_workers': self.scraping.max_workers,
                'timeout': self.scraping.timeout,
                'retry_count': self.scraping.retry_count,
                'min_delay': self.scraping.min_delay,
                'max_delay': self.scraping.max_delay
            },
            'model': {
                'test_size': self.model.test_size,
                'random_state': self.model.random_state,
                'n_trials': self.model.n_trials,
                'cv_folds': self.model.cv_folds,
                'ensemble_weights': self.model.ensemble_weights
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'betting_fraction': self.backtest.betting_fraction,
                'ev_threshold': self.backtest.ev_threshold,
                'monthly_stop_loss': self.backtest.monthly_stop_loss,
                'kelly_fraction': self.backtest.kelly_fraction
            },
            'data': {
                'categorical_columns': self.data.categorical_columns,
                'numeric_columns': self.data.numeric_columns,
                'feature_engineering': self.data.feature_engineering
            }
        }


# シングルトンインスタンス（バリデーションをスキップして初期化）
try:
    config = Config()
except ConfigValidationError:
    # バリデーション失敗時はワーニングを出して続行
    import warnings
    warnings.warn("Configuration validation failed, using defaults", UserWarning)
    config = Config.__new__(Config)
    config.paths = PathConfig()
    config.scraping = ScrapingConfig()
    config.model = ModelConfig()
    config.backtest = BacktestConfig()
    config.data = DataConfig()
    config.logger = logging.getLogger(__name__)