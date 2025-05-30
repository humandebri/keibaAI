"""
データ処理ユーティリティのテスト
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.data_utils import (
    validate_data, clean_numeric_column, encode_categorical,
    create_time_features, calculate_running_statistics
)


class TestDataUtils(unittest.TestCase):
    """データユーティリティ関数のテスト"""
    
    def setUp(self):
        """テストデータの準備"""
        self.test_df = pd.DataFrame({
            'race_id': ['202301010101', '202301010102', '202301010103'],
            '馬番': [1, 2, 3],
            '着順': ['1', '2', '失格'],
            '馬': ['ウマA', 'ウマB', 'ウマC'],
            '騎手': ['騎手A', '騎手B', '騎手A'],
            'オッズ': [2.5, 10.0, 5.5],
            '人気': [1, 3, 2],
            '日付': ['2023/01/01', '2023/01/01', '2023/01/02']
        })
        
    def test_validate_data_valid(self):
        """正常なデータの検証テスト"""
        required_cols = ['race_id', '馬番', '着順']
        results = validate_data(self.test_df, required_cols)
        
        self.assertTrue(results['is_valid'])
        self.assertEqual(len(results['missing_columns']), 0)
        self.assertEqual(results['row_count'], 3)
        
    def test_validate_data_missing_columns(self):
        """必須カラム不足のテスト"""
        required_cols = ['race_id', '馬番', '不存在カラム']
        results = validate_data(self.test_df, required_cols)
        
        self.assertFalse(results['is_valid'])
        self.assertIn('不存在カラム', results['missing_columns'])
        
    def test_validate_data_duplicates(self):
        """重複データの検出テスト"""
        df_with_dup = self.test_df.copy()
        df_with_dup.loc[3] = df_with_dup.loc[0]  # 重複追加
        
        results = validate_data(df_with_dup, ['race_id', '馬番'])
        self.assertEqual(results['duplicate_count'], 1)
        
    def test_clean_numeric_column(self):
        """数値カラムのクリーニングテスト"""
        cleaned = clean_numeric_column(self.test_df['着順'])
        
        self.assertEqual(cleaned.iloc[0], 1)
        self.assertEqual(cleaned.iloc[1], 2)
        self.assertTrue(pd.isna(cleaned.iloc[2]))  # '失格'はNaN
        
    def test_encode_categorical_label(self):
        """ラベルエンコーディングのテスト"""
        encoded_df = encode_categorical(self.test_df, ['騎手'], 'label')
        
        self.assertIn('騎手_encoded', encoded_df.columns)
        self.assertEqual(encoded_df['騎手_encoded'].nunique(), 2)  # 騎手A, B
        
    def test_encode_categorical_onehot(self):
        """ワンホットエンコーディングのテスト"""
        encoded_df = encode_categorical(self.test_df, ['騎手'], 'onehot')
        
        self.assertIn('騎手_騎手A', encoded_df.columns)
        self.assertIn('騎手_騎手B', encoded_df.columns)
        
    def test_create_time_features(self):
        """時系列特徴量作成のテスト"""
        df_with_features = create_time_features(self.test_df)
        
        self.assertIn('year', df_with_features.columns)
        self.assertIn('month', df_with_features.columns)
        self.assertIn('dayofweek', df_with_features.columns)
        
        self.assertEqual(df_with_features['year'].iloc[0], 2023)
        self.assertEqual(df_with_features['month'].iloc[0], 1)
        
    def test_calculate_running_statistics(self):
        """移動統計量計算のテスト"""
        df_with_stats = calculate_running_statistics(
            self.test_df, ['騎手'], '人気', window_size=2
        )
        
        self.assertIn('騎手_人気_rolling_mean', df_with_stats.columns)
        self.assertIn('騎手_人気_rolling_std', df_with_stats.columns)


if __name__ == '__main__':
    unittest.main()