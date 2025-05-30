"""
データ処理用ユーティリティ関数
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_race_data(file_paths: Union[str, List[str]], 
                  encoding: str = 'shift-jis') -> pd.DataFrame:
    """
    レースデータを読み込む
    
    Args:
        file_paths: CSVファイルパスまたはパスのリスト
        encoding: ファイルエンコーディング
        
    Returns:
        読み込んだDataFrame
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, encoding=encoding)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {path}")
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, any]:
    """
    データ品質をチェック
    
    Args:
        df: チェック対象のDataFrame
        required_columns: 必須カラムのリスト
        
    Returns:
        検証結果の辞書
    """
    results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'duplicate_count': 0,
        'row_count': len(df),
        'issues': []
    }
    
    # 必須カラムのチェック
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        results['is_valid'] = False
        results['missing_columns'] = list(missing_cols)
        results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # NULL値のチェック
    null_counts = df[required_columns].isnull().sum()
    results['null_counts'] = null_counts.to_dict()
    
    # 重複チェック
    if 'race_id' in df.columns and '馬番' in df.columns:
        duplicate_count = df.duplicated(subset=['race_id', '馬番']).sum()
        results['duplicate_count'] = duplicate_count
        if duplicate_count > 0:
            results['issues'].append(f"Found {duplicate_count} duplicate entries")
    
    # データ型チェック
    numeric_columns = ['着順', '馬番', '人気', '体重', '斤量']
    for col in numeric_columns:
        if col in df.columns:
            non_numeric = df[col].apply(lambda x: not pd.isna(x) and not str(x).isdigit())
            if non_numeric.any():
                results['issues'].append(f"Non-numeric values found in {col}")
    
    return results


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    数値カラムをクリーニング
    
    Args:
        series: クリーニング対象のSeries
        
    Returns:
        クリーニング済みのSeries
    """
    # 文字列から数値を抽出
    def extract_number(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x)
        # 数字以外を除去
        import re
        numbers = re.findall(r'\d+', x_str)
        if numbers:
            return int(numbers[0])
        return np.nan
    
    return series.apply(extract_number)


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                      encoding_type: str = 'label') -> pd.DataFrame:
    """
    カテゴリカル変数をエンコード
    
    Args:
        df: 対象のDataFrame
        columns: エンコード対象のカラム
        encoding_type: 'label' or 'onehot'
        
    Returns:
        エンコード済みのDataFrame
    """
    df_encoded = df.copy()
    
    if encoding_type == 'label':
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # NaNを文字列に変換してからエンコード
                df_encoded[col] = df_encoded[col].fillna('missing')
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                encoders[col] = le
                
    elif encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, dummy_na=True)
        
    return df_encoded


def create_time_features(df: pd.DataFrame, date_column: str = '日付') -> pd.DataFrame:
    """
    時系列特徴量を作成
    
    Args:
        df: 対象のDataFrame
        date_column: 日付カラム名
        
    Returns:
        特徴量追加済みのDataFrame
    """
    df_with_features = df.copy()
    
    if date_column in df_with_features.columns:
        # 日付をdatetimeに変換
        df_with_features[date_column] = pd.to_datetime(
            df_with_features[date_column], 
            format='%Y/%m/%d', 
            errors='coerce'
        )
        
        # 時系列特徴量の作成
        df_with_features['year'] = df_with_features[date_column].dt.year
        df_with_features['month'] = df_with_features[date_column].dt.month
        df_with_features['day'] = df_with_features[date_column].dt.day
        df_with_features['dayofweek'] = df_with_features[date_column].dt.dayofweek
        df_with_features['quarter'] = df_with_features[date_column].dt.quarter
        
    return df_with_features


def calculate_running_statistics(df: pd.DataFrame, 
                               group_columns: List[str],
                               target_column: str,
                               window_size: int = 5) -> pd.DataFrame:
    """
    移動統計量を計算
    
    Args:
        df: 対象のDataFrame
        group_columns: グループ化するカラム
        target_column: 統計量を計算する対象カラム
        window_size: ウィンドウサイズ
        
    Returns:
        統計量追加済みのDataFrame
    """
    df_with_stats = df.copy()
    
    # ソート
    if '日付' in df.columns:
        df_with_stats = df_with_stats.sort_values(['日付'] + group_columns)
    
    # グループごとに移動統計量を計算
    for col in group_columns:
        if col in df_with_stats.columns and target_column in df_with_stats.columns:
            grouped = df_with_stats.groupby(col)[target_column]
            
            # 移動平均
            df_with_stats[f'{col}_{target_column}_rolling_mean'] = (
                grouped.transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            )
            
            # 移動標準偏差
            df_with_stats[f'{col}_{target_column}_rolling_std'] = (
                grouped.transform(lambda x: x.rolling(window_size, min_periods=1).std())
            )
            
    return df_with_stats