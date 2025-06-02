#!/usr/bin/env python3
"""
2020-2025年のデータを使用してモデルを訓練するスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# 日本語フォント設定
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def load_race_data():
    """2020-2025年のエンコード済みデータを読み込む"""
    encoded_path = 'encoded/2020_2025encoded_data_v2.csv'
    
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(f"{encoded_path}が見つかりません。先にencode_2020_2025_data.pyを実行してください。")
    
    df = pd.read_csv(encoded_path)
    print(f"データを読み込みました: {encoded_path}")
    print(f"データサイズ: {df.shape}")
    
    # race_idから実際の日付を抽出
    if 'race_id' in df.columns:
        df['race_id_str'] = df['race_id'].astype(str).str.replace('.0', '')
        df['actual_date'] = pd.to_datetime(df['race_id_str'].str[:8], format='%Y%m%d', errors='coerce')
        
        valid_dates = df['actual_date'].notna()
        print(f"\n日付変換成功率: {valid_dates.sum() / len(df) * 100:.1f}%")
        
        if valid_dates.sum() > 0:
            df = df[valid_dates].copy()
            print(f"有効なデータ数: {len(df)}")
    
    return df

def create_features(df):
    """特徴量エンジニアリング（2020-2025データ用に最適化）"""
    df_features = df.copy()
    
    print("\n=== 特徴量エンジニアリング開始 ===")
    
    # 基本的な特徴量作成
    if '前走着順' in df_features.columns:
        df_features['前走勝利'] = (df_features['前走着順'] == 1).astype(int)
        df_features['前走連対'] = (df_features['前走着順'] <= 2).astype(int)
        df_features['前走着内'] = (df_features['前走着順'] <= 3).astype(int)
    
    # 距離カテゴリ
    if '距離' in df_features.columns:
        df_features['短距離'] = (df_features['距離'] <= 1400).astype(int)
        df_features['マイル'] = ((df_features['距離'] > 1400) & (df_features['距離'] <= 1800)).astype(int)
        df_features['中距離'] = ((df_features['距離'] > 1800) & (df_features['距離'] <= 2400)).astype(int)
        df_features['長距離'] = (df_features['距離'] > 2400).astype(int)
    
    # 枠番の影響
    if '枠番' in df_features.columns and '頭数' in df_features.columns:
        df_features['内枠'] = (df_features['枠番'] <= 3).astype(int)
        df_features['外枠'] = (df_features['枠番'] >= 7).astype(int)
        df_features['相対枠位置'] = df_features['枠番'] / df_features['頭数']
    
    # 時期の影響（2020-2025特有の傾向を捉える）
    if 'actual_date' in df_features.columns:
        df_features['年'] = df_features['actual_date'].dt.year
        df_features['月'] = df_features['actual_date'].dt.month
        df_features['コロナ期間'] = ((df_features['年'] == 2020) | 
                                  ((df_features['年'] == 2021) & (df_features['月'] <= 6))).astype(int)
    
    created_features = len(df_features.columns) - len(df.columns)
    print(f"作成した特徴量数: {created_features}個")
    
    return df_features

def train_model_2020_2025(df_features):
    """2020-2025年データでモデルを訓練"""
    print("\n=== モデル訓練開始（2020-2025年データ） ===")
    
    # ターゲット作成
    df_features['target'] = (df_features['着順'] <= 3).astype(int)
    
    # 特徴量選択
    exclude_cols = ['着順', 'target', 'オッズ', '人気', '上がり', '走破時間', 
                    '通過順', '日付', 'actual_date', 'year', '月', 'race_id', 
                    'race_id_str', '馬番', '賞金']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"使用する特徴量数: {len(feature_cols)}")
    
    # データを時系列順にソート
    df_features = df_features.sort_values('actual_date').reset_index(drop=True)
    
    # 訓練・検証データ分割（2020-2023を訓練、2024-2025を検証）
    train_mask = df_features['年'] <= 2023
    
    X_train = df_features[train_mask][feature_cols]
    y_train = df_features[train_mask]['target']
    X_valid = df_features[~train_mask][feature_cols]
    y_valid = df_features[~train_mask]['target']
    
    print(f"\n訓練データ: {len(X_train)}件 (2020-2023)")
    print(f"検証データ: {len(X_valid)}件 (2024-2025)")
    
    # 欠損値処理（数値型の列のみ）
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_means = X_train[numeric_cols].mean()
    
    # 数値型の列のみ欠損値を埋める
    X_train.loc[:, numeric_cols] = X_train[numeric_cols].fillna(train_means)
    X_valid.loc[:, numeric_cols] = X_valid[numeric_cols].fillna(train_means)
    
    # 非数値型の列は削除または適切に処理
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"警告: 非数値型の列が見つかりました: {list(non_numeric_cols)}")
        # 非数値型の列を削除
        X_train = X_train.drop(columns=non_numeric_cols)
        X_valid = X_valid.drop(columns=non_numeric_cols)
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    # クラス重み
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # LightGBMパラメータ（2020-2025データ用に調整）
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': -1,
        'num_leaves': 50,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'n_estimators': 500,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    # モデル訓練
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
    )
    
    # 評価
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred)
    
    print(f"\n検証AUCスコア: {auc_score:.4f}")
    
    # 特徴量重要度（更新されたfeature_colsを使用）
    # 実際に使用された特徴量数を確認
    n_features_used = len(model.feature_importances_)
    if len(feature_cols) != n_features_used:
        print(f"警告: 特徴量数が一致しません。元: {len(feature_cols)}, 使用: {n_features_used}")
        # 実際に使用された特徴量のリストを再構築
        actual_feature_cols = list(X_train.columns)[:n_features_used]
    else:
        actual_feature_cols = feature_cols
    
    feature_importance = pd.DataFrame({
        'feature': actual_feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n重要な特徴量トップ10:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30} 重要度: {row['importance']:.0f}")
    
    # 実際に使用された特徴量を返す
    return model, actual_feature_cols, auc_score

def save_model(model, feature_cols):
    """モデルと関連情報を保存"""
    model_dir = Path('model_2020_2025')
    model_dir.mkdir(exist_ok=True)
    
    # モデル保存
    joblib.dump(model, model_dir / 'model_2020_2025.pkl')
    
    # 特徴量リスト保存
    joblib.dump(feature_cols, model_dir / 'feature_cols_2020_2025.pkl')
    
    # モデル情報保存
    model_info = {
        'training_period': '2020-2025',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(feature_cols),
        'model_type': 'LightGBM'
    }
    
    import json
    with open(model_dir / 'model_info_2020_2025.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")

def main():
    """メイン処理"""
    print("=" * 60)
    print("競馬予測モデル訓練（2020-2025年データ）")
    print("=" * 60)
    
    # データ読み込み
    try:
        df = load_race_data()
    except FileNotFoundError as e:
        print(f"\n❌ エラー: {e}")
        return
    
    # データ期間の確認
    if 'actual_date' in df.columns:
        print(f"\n=== データ期間の確認 ===")
        print(f"データ期間: {df['actual_date'].min()} ~ {df['actual_date'].max()}")
        df['年'] = df['actual_date'].dt.year
        print(f"\n年別レコード数:")
        print(df['年'].value_counts().sort_index())
    
    # 特徴量エンジニアリング
    df_features = create_features(df)
    
    # モデル訓練
    model, feature_cols, auc_score = train_model_2020_2025(df_features)
    
    # モデル保存
    save_model(model, feature_cols)
    
    print("\n" + "=" * 60)
    print("訓練完了サマリー")
    print("=" * 60)
    print(f"訓練期間: 2020-2025年")
    print(f"検証AUCスコア: {auc_score:.4f}")
    print(f"使用特徴量数: {len(feature_cols)}")
    print("\n次のステップ:")
    print("1. integrated_betting_system.pyの'model_path'を'model_2020_2025/model_2020_2025.pkl'に更新")
    print("2. python integrated_betting_system.pyで明日以降のレースを分析")

if __name__ == "__main__":
    main()