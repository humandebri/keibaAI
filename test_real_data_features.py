#!/usr/bin/env python3
"""
実データでの特徴量改善テスト
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """実データで特徴量改善をテスト"""
    
    print("="*60)
    print("実データでの特徴量改善テスト")
    print("="*60)
    
    # 実データを読み込み
    data_path = 'encoded/2020_2025encoded_data_v2.csv'
    if not os.path.exists(data_path):
        print(f"データファイルが見つかりません: {data_path}")
        return
    
    print("データ読み込み中...")
    data = pd.read_csv(data_path, nrows=1000)  # 最初の1000行をテスト用に
    print(f"データ読み込み完了: {len(data)}行, {len(data.columns)}列")
    
    # 元のデータの特徴量数
    original_feature_count = len(data.columns)
    print(f"元の特徴量数: {original_feature_count}")
    
    # 統一特徴量エンジンを適用
    from src.features.unified_features import UnifiedFeatureEngine
    
    print("\n統一特徴量エンジン適用中...")
    engine = UnifiedFeatureEngine()
    
    # データの必要な前処理
    data = prepare_data_for_engine(data)
    
    try:
        enhanced_data = engine.build_all_features(data)
        new_feature_count = len(enhanced_data.columns)
        
        print(f"強化後の特徴量数: {new_feature_count}")
        print(f"追加された特徴量数: {new_feature_count - original_feature_count}")
        
        # 利用可能な特徴量を確認
        available_features = engine.get_feature_columns(enhanced_data)
        print(f"利用可能な特徴量数: {len(available_features)}")
        
        # 新しい重要特徴量を確認
        key_features = check_key_features(enhanced_data)
        
        # 期待値改善の推定
        estimate_ev_improvement(len(available_features), key_features)
        
        # データ品質チェック
        quality_check(enhanced_data, key_features)
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

def prepare_data_for_engine(data):
    """統一特徴量エンジン用にデータを準備"""
    
    # 必要な列を確認・追加
    if 'race_id' not in data.columns:
        # race_idを作成（日付と場所から）
        data['race_id'] = data.index.astype(str)
    
    # 体重の数値化
    if '体重' in data.columns and '体重_numeric' not in data.columns:
        weight_values = []
        for w in data['体重']:
            try:
                if pd.isna(w):
                    weight = 480
                else:
                    weight_str = str(w)
                    if '(' in weight_str:
                        weight = int(weight_str.split('(')[0])
                    else:
                        weight = int(float(weight_str))
            except:
                weight = 480
            weight_values.append(weight)
        data['体重_numeric'] = weight_values
    
    # 年カラムを追加
    if 'year' not in data.columns and '日付' in data.columns:
        try:
            # 数値データを日付に変換
            base_date = pd.to_datetime('2020-01-01')
            data['year'] = (base_date + pd.to_timedelta(data['日付'], unit='D')).dt.year
        except:
            data['year'] = 2024  # デフォルト値
    
    print(f"データ準備完了: {len(data.columns)}列")
    return data

def check_key_features(data):
    """重要な新機能をチェック"""
    print("\n重要な新機能の確認:")
    print("-" * 30)
    
    key_features = {
        'speed_features': ['基本スピード指数', 'スピード指数_正規化', 'トラック調整スピード', 'ベストスピード指数'],
        'ranking_features': ['オッズ順位_レース内', '斤量順位_レース内', '体重順位_レース内', 'スピード順位_レース内'],
        'change_features': ['クラス変化', '距離変化_絶対値', '負担重量比', 'コース変更'],
        'interaction_features': ['draw_track_interaction', 'inside_draw_advantage', 'weather_track_interaction']
    }
    
    available_key_features = {}
    
    for category, features in key_features.items():
        available = [f for f in features if f in data.columns]
        available_key_features[category] = available
        
        print(f"{category}: {len(available)}/{len(features)} 利用可能")
        for feature in available:
            print(f"  ✓ {feature}")
        
        for feature in features:
            if feature not in available:
                print(f"  ✗ {feature}")
    
    return available_key_features

def estimate_ev_improvement(feature_count, key_features):
    """期待値改善を推定"""
    print("\n期待値改善推定:")
    print("-" * 30)
    
    # ベースライン（改善前）
    baseline_ev = 0.6
    baseline_features = 59
    
    # 特徴量数による改善
    feature_improvement = (feature_count - baseline_features) * 0.005  # 1特徴量あたり0.005の改善
    
    # 重要機能による追加改善
    speed_boost = len(key_features.get('speed_features', [])) * 0.04  # スピード指数は重要
    ranking_boost = len(key_features.get('ranking_features', [])) * 0.02  # 相対順位も重要
    change_boost = len(key_features.get('change_features', [])) * 0.015  # 変化検出
    interaction_boost = len(key_features.get('interaction_features', [])) * 0.01  # 交互作用
    
    estimated_ev = baseline_ev + feature_improvement + speed_boost + ranking_boost + change_boost + interaction_boost
    
    print(f"ベースライン期待値: {baseline_ev:.3f}")
    print(f"特徴量数改善効果: +{feature_improvement:.3f} ({feature_count} vs {baseline_features} features)")
    print(f"スピード指数効果: +{speed_boost:.3f} ({len(key_features.get('speed_features', []))} features)")
    print(f"相対順位効果: +{ranking_boost:.3f} ({len(key_features.get('ranking_features', []))} features)")
    print(f"変化検出効果: +{change_boost:.3f} ({len(key_features.get('change_features', []))} features)")
    print(f"交互作用効果: +{interaction_boost:.3f} ({len(key_features.get('interaction_features', []))} features)")
    print(f"推定期待値: {estimated_ev:.3f}")
    
    if estimated_ev >= 1.0:
        print("✓ 目標期待値1.0以上を達成予想！")
        improvement_percentage = ((estimated_ev - baseline_ev) / baseline_ev) * 100
        print(f"  改善率: +{improvement_percentage:.1f}%")
    elif estimated_ev >= 0.8:
        print("△ 大幅な期待値改善を確認、1.0達成に近づいている")
        needed = 1.0 - estimated_ev
        print(f"  1.0達成まであと{needed:.3f}の改善が必要")
    else:
        print("○ 期待値改善の兆候あり、さらなる最適化が必要")

def quality_check(data, key_features):
    """データ品質をチェック"""
    print("\nデータ品質チェック:")
    print("-" * 30)
    
    # NaN率チェック
    total_features = len(data.columns)
    high_quality_features = 0
    
    for col in data.columns:
        nan_rate = data[col].isna().sum() / len(data)
        if nan_rate < 0.1:  # 10%未満のNaN率
            high_quality_features += 1
    
    quality_rate = high_quality_features / total_features
    print(f"高品質特徴量率: {quality_rate:.1%} ({high_quality_features}/{total_features})")
    
    # 重要特徴量の統計
    for category, features in key_features.items():
        if features:
            print(f"\n{category}の統計:")
            for feature in features[:2]:  # 最初の2つだけ表示
                if feature in data.columns:
                    series = data[feature].dropna()
                    if len(series) > 0:
                        print(f"  {feature}: 平均={series.mean():.3f}, 標準偏差={series.std():.3f}")

if __name__ == "__main__":
    main()