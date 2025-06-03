"""
共通ユーティリティモジュール
データ読み込み、保存、ログ出力などの共通処理
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os
import lightgbm as lgb
import pickle
import json

from .config import config, DATA_DIR, ENCODED_DIR, MODELS_DIR

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ロガーの設定
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """統一されたロガーの設定"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ハンドラーが既に設定されている場合はスキップ
    if not logger.handlers:
        # コンソール出力
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        # フォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
    
    return logger


class DataLoader:
    """データ読み込みの共通クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(__name__)
    
    def load_race_data(self, years: Union[int, List[int]], 
                      data_dir: Optional[Path] = None,
                      use_payout_data: bool = False) -> pd.DataFrame:
        """レースデータの読み込み"""
        if isinstance(years, int):
            years = [years]
        
        # Check for data_with_payout directory first if requested
        if use_payout_data:
            payout_dir = Path("data_with_payout")
            if payout_dir.exists():
                data_dir = payout_dir
                self.logger.info("Using data_with_payout directory")
        
        data_dir = data_dir or DATA_DIR
        dfs = []
        
        for year in years:
            # Try multiple file patterns for payout data
            file_patterns = [
                f"{year}_with_payout.xlsx",
                f"{year}.xlsx"
            ] if use_payout_data else [f"{year}.xlsx"]
            
            file_loaded = False
            for pattern in file_patterns:
                file_path = data_dir / pattern
                if file_path.exists():
                    try:
                        df = pd.read_excel(file_path)
                        # 着順を数値に変換し、無効な値を除去
                        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                        df = df.dropna(subset=['着順'])
                        df['year'] = year
                        self.logger.info(f"Loaded {file_path.name}: {len(df)} rows")
                        dfs.append(df)
                        file_loaded = True
                        break
                    except Exception as e:
                        self.logger.error(f"Error loading {file_path}: {e}")
            
            if not file_loaded:
                # Fallback to regular data directory
                fallback_path = DATA_DIR / f"{year}.xlsx"
                if fallback_path.exists() and data_dir != DATA_DIR:
                    try:
                        df = pd.read_excel(fallback_path)
                        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                        df = df.dropna(subset=['着順'])
                        df['year'] = year
                        self.logger.info(f"Loaded {fallback_path.name} (fallback): {len(df)} rows")
                        dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Error loading {fallback_path}: {e}")
                else:
                    self.logger.warning(f"No data file found for year {year}")
        
        if not dfs:
            raise ValueError("No data files were loaded")
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_encoded_data(self, file_name: str = "encoded_data.csv",
                         encoded_dir: Optional[Path] = None) -> pd.DataFrame:
        """エンコード済みデータの読み込み"""
        encoded_dir = encoded_dir or ENCODED_DIR
        file_path = encoded_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Encoded data not found: {file_path}")
        
        self.logger.info(f"Loading encoded data from {file_path}")
        return pd.read_csv(file_path)
    
    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path],
                  index: bool = False):
        """データの保存（CSV/Excel自動判定）"""
        file_path = Path(file_path)
        
        # ディレクトリが存在しない場合は作成
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.csv':
            df.to_csv(file_path, index=index, encoding='utf-8-sig')
        elif file_path.suffix in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=index)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.logger.info(f"Saved data to {file_path} ({len(df)} rows)")


class FeatureProcessor:
    """特徴量処理の共通クラス（統一特徴量エンジンを使用）"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(__name__)
        self.config = config.data
        self._feature_engine = None
    
    @property
    def feature_engine(self):
        """遅延インポートでUnifiedFeatureEngineを取得"""
        if self._feature_engine is None:
            from ..features.unified_features import UnifiedFeatureEngine
            self._feature_engine = UnifiedFeatureEngine()
        return self._feature_engine
    
    def prepare_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量の準備"""
        df = df.copy()
        
        # 着順を数値に変換
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        
        # race_idが存在しない場合は作成
        if 'race_id' not in df.columns:
            if all(col in df.columns for col in ['year', '開催', '回', '日', 'レース']):
                df['race_id'] = (df['year'].astype(str) + 
                               df['開催'].astype(str).str.zfill(2) + 
                               df['回'].astype(str).str.zfill(2) + 
                               df['日'].astype(str).str.zfill(2) + 
                               df['レース'].astype(str).str.zfill(2))
        
        # 統一特徴量エンジンで特徴量を構築
        self.logger.info("Building features using unified feature engine...")
        df = self.feature_engine.build_all_features(df)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """ターゲット変数の作成"""
        return self.feature_engine.create_target_variables(df)
    
    def get_feature_columns(self, df: pd.DataFrame, 
                           exclude_cols: Optional[List[str]] = None) -> List[str]:
        """特徴量カラムのリストを取得"""
        feature_cols = self.feature_engine.get_feature_columns(df)
        
        if exclude_cols:
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        return feature_cols


class ModelManager:
    """モデル管理の共通クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(__name__)
    
    def save_models(self, models: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """モデルの保存"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(models, f)
        
        self.logger.info(f"Models saved to {file_path}")
    
    def load_models(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """モデルの読み込み"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            models = pickle.load(f)
        
        self.logger.info(f"Models loaded from {file_path}")
        return models
    
    def save_model_info(self, info: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """モデル情報の保存"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Model info saved to {file_path}")
    
    def load_model_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """モデル情報の読み込み"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model info file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        self.logger.info(f"Model info loaded from {file_path}")
        return info


def calculate_statistics(series: pd.Series) -> dict:
    """統計情報の計算"""
    return {
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'median': series.median(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75)
    }


def format_currency(amount: float) -> str:
    """通貨フォーマット"""
    return f"¥{amount:,.0f}"


def calculate_return_metrics(initial_capital: float, final_capital: float, 
                           n_years: float = 1) -> dict:
    """リターン指標の計算"""
    total_return = (final_capital - initial_capital) / initial_capital
    annual_return = (final_capital / initial_capital) ** (1 / n_years) - 1
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'profit_loss': final_capital - initial_capital
    }