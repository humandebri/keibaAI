"""
共通ユーティリティモジュール
データ読み込み、保存、ログ出力などの共通処理
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Union
import os
import lightgbm as lgb

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
                      data_dir: Optional[Path] = None) -> pd.DataFrame:
        """レースデータの読み込み"""
        if isinstance(years, int):
            years = [years]
        
        data_dir = data_dir or DATA_DIR
        dfs = []
        
        for year in years:
            file_path = data_dir / f"{year}.xlsx"
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    # 着順を数値に変換し、無効な値を除去
                    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                    df = df.dropna(subset=['着順'])
                    df['year'] = year
                    self.logger.info(f"Loaded {file_path.name}: {len(df)} rows")
                    dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
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
    """特徴量処理の共通クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(__name__)
        self.config = config.data
    
    def prepare_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量の準備"""
        df = df.copy()
        
        # 着順を数値に変換
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        
        # カテゴリカル変数のエンコーディング
        for col in self.config.categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # 数値変数の処理
        for col in self.config.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 欠損値を中央値で埋める
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.logger.debug(f"Filled {col} NaN values with median: {median_val}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """ターゲット変数の作成"""
        df = df.copy()
        
        # 勝利フラグ
        df['is_win'] = (df['着順'] == 1).astype(int)
        
        # 複勝フラグ（3着以内）
        df['is_place'] = (df['着順'] <= 3).astype(int)
        
        # 連対フラグ（2着以内）
        df['is_exacta'] = (df['着順'] <= 2).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, 
                           exclude_cols: Optional[List[str]] = None) -> List[str]:
        """特徴量カラムのリストを取得"""
        if exclude_cols is None:
            exclude_cols = ['着順', 'is_win', 'is_place', 'is_exacta', 
                           'race_id', 'horse_id', 'year', 'date']
        
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
        
        return feature_cols


class ModelManager:
    """モデル管理の共通クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(__name__)
        self.models_dir = MODELS_DIR
    
    def save_model(self, model, model_name: str, model_type: str = 'lightgbm'):
        """モデルの保存"""
        file_path = self.models_dir / f"{model_name}.txt"
        
        if model_type == 'lightgbm':
            model.save_model(str(file_path))
        else:
            import joblib
            file_path = self.models_dir / f"{model_name}.pkl"
            joblib.dump(model, file_path)
        
        self.logger.info(f"Model saved to {file_path}")
    
    def load_model(self, model_name: str, model_type: str = 'lightgbm'):
        """モデルの読み込み"""
        if model_type == 'lightgbm':
            file_path = self.models_dir / f"{model_name}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Model not found: {file_path}")
            return lgb.Booster(model_file=str(file_path))
        else:
            import joblib
            file_path = self.models_dir / f"{model_name}.pkl"
            if not file_path.exists():
                raise FileNotFoundError(f"Model not found: {file_path}")
            return joblib.load(file_path)


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