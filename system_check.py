#!/usr/bin/env python3
"""
システム状態チェックスクリプト
使い方: python system_check.py
"""

import os
import sys
import pandas as pd
import subprocess
from datetime import datetime

def check_environment():
    """環境チェック"""
    print("🔍 環境チェック")
    print("-" * 30)
    
    # Pythonバージョン
    print(f"Python版: {sys.version}")
    
    # 仮想環境チェック
    venv_active = os.environ.get('VIRTUAL_ENV') is not None
    print(f"仮想環境: {'✅ アクティブ' if venv_active else '❌ 非アクティブ'}")
    
    # 必要パッケージ
    required_packages = ['pandas', 'numpy', 'lightgbm', 'scikit-learn']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}: ✅")
        except ImportError:
            print(f"{package}: ❌ 未インストール")

def check_data_files():
    """データファイルチェック"""
    print("\n📊 データファイルチェック")
    print("-" * 30)
    
    # エンコード済みデータ
    encoded_files = [
        'encoded/2020_2025encoded_data_v2.csv',
        'encoded/2022_2023encoded_data_v2.csv'
    ]
    
    for file_path in encoded_files:
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path, nrows=5)
                print(f"✅ {file_path}")
                print(f"   行数確認中...")
                row_count = len(pd.read_csv(file_path))
                print(f"   {row_count:,}行, {len(data.columns)}列")
            except Exception as e:
                print(f"❌ {file_path} (読み込みエラー: {e})")
        else:
            print(f"❌ {file_path} (ファイル未発見)")
    
    # 生データ
    data_dirs = ['data', 'data_with_payout']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
            print(f"✅ {data_dir}: {len(files)}ファイル")
        else:
            print(f"❌ {data_dir}: ディレクトリ未発見")

def check_models():
    """モデルファイルチェック"""
    print("\n🤖 モデルファイルチェック")
    print("-" * 30)
    
    model_files = [
        'model_2020_2025/model_2020_2025.pkl',
        'model_2020_2025/feature_cols_2020_2025.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({file_size:.1f}MB)")
        else:
            print(f"❌ {file_path}")

def check_scripts():
    """スクリプトファイルチェック"""
    print("\n📝 スクリプトファイルチェック")
    print("-" * 30)
    
    script_files = [
        'run_backtest.py',
        'predict_races.py',
        'scrape_and_process.py',
        'demo_optimal_system.py',
        'src/strategies/optimized_kelly_strategy.py',
        'src/features/unified_features.py'
    ]
    
    for file_path in script_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

def test_import():
    """インポートテスト"""
    print("\n🧪 インポートテスト")
    print("-" * 30)
    
    test_imports = [
        ('src.strategies.optimized_kelly_strategy', 'OptimizedKellyStrategy'),
        ('src.features.unified_features', 'UnifiedFeatureEngine'),
        ('src.core.config', 'config'),
        ('src.core.utils', 'setup_logger')
    ]
    
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")

def test_basic_functionality():
    """基本機能テスト"""
    print("\n⚡ 基本機能テスト")
    print("-" * 30)
    
    try:
        # 統一特徴量エンジンテスト
        from src.features.unified_features import UnifiedFeatureEngine
        
        engine = UnifiedFeatureEngine()
        print(f"✅ UnifiedFeatureEngine初期化")
        print(f"   登録ビルダー数: {len(engine.builders)}")
        
        # サンプルデータでテスト
        import numpy as np
        
        sample_data = pd.DataFrame({
            'race_id': ['test001'] * 5,
            '馬': ['Horse1', 'Horse2', 'Horse3', 'Horse4', 'Horse5'],
            '馬番': [1, 2, 3, 4, 5],
            '人気': [1, 2, 3, 4, 5],
            'オッズ': [2.1, 3.4, 5.6, 8.2, 12.3],
            '斤量': [57, 56, 55, 57, 54],
            '体重': ['480(+2)', '465(-1)', '478(+3)', '492(0)', '458(-4)'],
            '体重変化': [2, -1, 3, 0, -4],
            '性': ['牡', '牝', '牡', '牡', '牝'],
            '年齢': [4, 3, 5, 6, 4],
            '走破時間': ['1:22.3', '1:23.1', '1:22.8', '1:24.2', '1:23.5'],
            '距離': [1600] * 5,
            'クラス': [5] * 5,
            '芝・ダート': [0] * 5,
            '馬場': [0] * 5,
            '天気': [0] * 5,
            '日付': pd.to_datetime('2024-01-01'),
            'year': [2024] * 5
        })
        
        enhanced_data = engine.build_all_features(sample_data)
        feature_count = len(engine.get_feature_columns(enhanced_data))
        
        print(f"✅ 特徴量構築テスト")
        print(f"   構築特徴量数: {feature_count}")
        
        if feature_count >= 80:
            print("✅ 十分な特徴量数を確認")
        else:
            print("⚠️ 特徴量数が少ない可能性があります")
    
    except Exception as e:
        print(f"❌ 基本機能テスト失敗: {e}")

def system_recommendations():
    """システム推奨事項"""
    print("\n💡 推奨事項")
    print("-" * 30)
    
    recommendations = []
    
    # データファイルチェック
    if not os.path.exists('encoded/2020_2025encoded_data_v2.csv'):
        recommendations.append("エンコード済みデータを準備してください")
    
    # 仮想環境チェック
    if not os.environ.get('VIRTUAL_ENV'):
        recommendations.append("仮想環境をアクティベートしてください: source .venv/bin/activate")
    
    # サンプルファイルチェック
    if not os.path.exists('sample_today_races.csv'):
        recommendations.append("サンプルレースファイルが利用可能です")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ システムは正常に設定されています！")

def main():
    print("🔧 競馬AI システム状態チェック")
    print("=" * 50)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_environment()
    check_data_files()
    check_models()
    check_scripts()
    test_import()
    test_basic_functionality()
    system_recommendations()
    
    print(f"\n✅ システムチェック完了")
    print(f"詳細ガイド: OPTIMAL_SYSTEM_USAGE_GUIDE.md")

if __name__ == "__main__":
    main()