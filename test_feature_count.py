#!/usr/bin/env python3
"""
特徴量カウントテスト - 改善された統一特徴量エンジンの検証
"""

import pandas as pd
import numpy as np
import sys
import os

# パス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.unified_features import UnifiedFeatureEngine

def create_sample_data():
    """サンプルデータを作成"""
    print("サンプルデータ作成中...")
    
    np.random.seed(42)
    n_races = 10
    n_horses_per_race = 12
    n_total = n_races * n_horses_per_race
    
    # 基本データを作成
    data = {
        'race_id': np.repeat([f'20240101010{i+1:02d}' for i in range(n_races)], n_horses_per_race),
        '馬': [f'Horse_{i}' for i in range(n_total)],
        '馬番': np.tile(range(1, n_horses_per_race + 1), n_races),
        '騎手': [f'Jockey_{i%20}' for i in range(n_total)],
        '調教師': [f'Trainer_{i%15}' for i in range(n_total)],
        '着順': np.random.randint(1, n_horses_per_race + 1, n_total),
        '人気': np.random.randint(1, n_horses_per_race + 1, n_total),
        'オッズ': np.random.uniform(1.5, 50.0, n_total),
        '斤量': np.random.uniform(52.0, 58.0, n_total),
        '体重': [f'{np.random.randint(420, 520)}(+{np.random.randint(-10, 15)})' for _ in range(n_total)],
        '体重変化': np.random.randint(-10, 15, n_total),
        '性': np.random.choice(['牡', '牝', 'セ'], n_total),
        '年齢': np.random.randint(3, 8, n_total),
        '走破時間': [f'1:{np.random.randint(20, 40)}.{np.random.randint(0, 9)}' for _ in range(n_total)],
        '距離': np.random.choice([1200, 1400, 1600, 1800, 2000, 2400], n_total),
        'クラス': np.random.randint(1, 10, n_total),
        '芝・ダート': np.random.choice(['芝', 'ダ'], n_total),
        '馬場': np.random.choice(['良', '稍', '重', '不'], n_total),
        '天気': np.random.choice(['晴', '曇', '雨'], n_total),
        '通過順': [f'{np.random.randint(1, 12)}-{np.random.randint(1, 12)}-{np.random.randint(1, 12)}' for _ in range(n_total)],
        '上がり': np.random.uniform(33.0, 38.0, n_total),
        'レース名': ['テストレース'] * n_total,
        '開催': ['東京'] * n_total,
        '場名': ['東京'] * n_total,
        '日付': pd.to_datetime('2024-01-01')
    }
    
    # 過去走データを追加
    for i in range(1, 6):
        data[f'馬番{i}'] = np.random.randint(1, 16, n_total)
        data[f'騎手{i}'] = [f'Jockey_{j%20}' for j in range(n_total)]
        data[f'オッズ{i}'] = np.random.uniform(1.5, 50.0, n_total)
        data[f'着順{i}'] = np.random.randint(1, 16, n_total)
        data[f'距離{i}'] = np.random.choice([1200, 1400, 1600, 1800, 2000], n_total)
        data[f'クラス{i}'] = np.random.randint(1, 10, n_total)
        data[f'走破時間{i}'] = [f'1:{np.random.randint(20, 40)}.{np.random.randint(0, 9)}' for _ in range(n_total)]
        data[f'芝・ダート{i}'] = np.random.choice([0, 1], n_total)  # エンコード済みと仮定
        data[f'日付{i}'] = pd.to_datetime('2024-01-01') - pd.Timedelta(days=i*30)
    
    df = pd.DataFrame(data)
    print(f"サンプルデータ作成完了: {len(df)}行, {len(df.columns)}列")
    return df

def test_feature_engine():
    """統一特徴量エンジンをテスト"""
    print("\n" + "="*50)
    print("統一特徴量エンジンテスト開始")
    print("="*50)
    
    # サンプルデータ作成
    df = create_sample_data()
    
    # 統一特徴量エンジンを初期化
    engine = UnifiedFeatureEngine()
    
    print(f"\n登録されたビルダー数: {len(engine.builders)}")
    for i, builder in enumerate(engine.builders):
        print(f"  {i+1}. {builder.__class__.__name__}")
    
    # 特徴量構築前の状態
    print(f"\n構築前の特徴量名数: {len(engine.feature_names)}")
    print(f"構築前のデータ列数: {len(df.columns)}")
    
    # 特徴量構築実行
    print("\n特徴量構築実行中...")
    result_df = engine.build_all_features(df)
    
    # 構築後の状態
    print(f"\n構築後の特徴量名数: {len(engine.feature_names)}")
    print(f"構築後のデータ列数: {len(result_df.columns)}")
    
    # 実際に存在する特徴量カラム
    available_features = engine.get_feature_columns(result_df)
    print(f"実際に利用可能な特徴量数: {len(available_features)}")
    
    # 詳細情報
    print("\n" + "="*50)
    print("詳細分析")
    print("="*50)
    
    # 各ビルダーの特徴量数
    total_expected = 0
    for builder in engine.builders:
        builder_features = builder.get_feature_names()
        available_count = sum(1 for f in builder_features if f in result_df.columns)
        total_expected += len(builder_features)
        print(f"{builder.__class__.__name__}:")
        print(f"  期待特徴量数: {len(builder_features)}")
        print(f"  実際利用可能: {available_count}")
        if available_count < len(builder_features):
            missing = [f for f in builder_features if f not in result_df.columns]
            print(f"  欠損特徴量: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    print(f"\n合計期待特徴量数: {total_expected}")
    print(f"合計利用可能特徴量数: {len(available_features)}")
    print(f"特徴量利用率: {len(available_features)/total_expected*100:.1f}%")
    
    # 新機能の検証
    print("\n" + "="*50)
    print("新機能検証")
    print("="*50)
    
    new_features = [
        '基本スピード指数', 'スピード指数_正規化', 'トラック調整スピード',
        '人気順位_レース内', 'オッズ順位_レース内', '斤量順位_レース内',
        'クラス変化', '距離変化_絶対値', '負担重量比'
    ]
    
    for feature in new_features:
        if feature in result_df.columns:
            print(f"✓ {feature}: 利用可能")
        else:
            print(f"✗ {feature}: 欠損")
    
    # サンプル値の確認
    print("\n新特徴量のサンプル値:")
    for feature in new_features[:3]:  # 最初の3つだけ表示
        if feature in result_df.columns:
            sample_values = result_df[feature].dropna().head(3).tolist()
            print(f"  {feature}: {sample_values}")
    
    print("\n" + "="*50)
    print("テスト完了")
    print("="*50)
    
    return len(available_features), total_expected

if __name__ == "__main__":
    available, expected = test_feature_engine()
    
    print(f"\n最終結果:")
    print(f"利用可能特徴量: {available}")
    print(f"期待特徴量: {expected}")
    
    if available >= 100:
        print("✓ 十分な特徴量数を達成 (100+)")
    else:
        print("✗ 特徴量数が不足")
    
    print("\n期待値改善への影響:")
    if available >= 100:
        print("- 十分な特徴量により期待値1.0+達成の可能性が高い")
    elif available >= 80:
        print("- 特徴量数は良好、期待値0.8-1.0程度の改善を期待")
    else:
        print("- さらなる特徴量追加が必要")