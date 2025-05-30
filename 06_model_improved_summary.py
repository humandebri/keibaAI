#!/usr/bin/env python3
"""
競馬予測モデルの改善版（要約版）
- 実行時間を短縮したバージョン
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """データ読み込みと前処理"""
    # データ読み込み
    path = 'encoded/2022_2023encoded_data.csv'
    df = pd.read_csv(path)
    
    # race_idから実際の日付を抽出
    df['race_id_str'] = df['race_id'].astype(str).str.replace('.0', '')
    df['actual_date'] = pd.to_datetime(df['race_id_str'].str[:8], format='%Y%m%d', errors='coerce')
    
    # 有効な日付のみ使用
    df = df[df['actual_date'].notna()].copy()
    
    print(f"データサイズ: {df.shape}")
    print(f"データ期間: {df['actual_date'].min()} ~ {df['actual_date'].max()}")
    
    return df


def create_features(df):
    """重要な特徴量のみを作成"""
    df_features = df.copy()
    
    # 1. 前走成績
    if '前走着順' in df_features.columns:
        df_features['前走勝利'] = (df_features['前走着順'] == 1).astype(int)
        df_features['前走連対'] = (df_features['前走着順'] <= 2).astype(int)
        df_features['前走着内'] = (df_features['前走着順'] <= 3).astype(int)
    
    # 2. 距離適性
    if '距離' in df_features.columns:
        df_features['短距離'] = (df_features['距離'] <= 1400).astype(int)
        df_features['マイル'] = ((df_features['距離'] > 1400) & (df_features['距離'] <= 1800)).astype(int)
        df_features['中距離'] = ((df_features['距離'] > 1800) & (df_features['距離'] <= 2400)).astype(int)
    
    # 3. 枠順
    if '枠番' in df_features.columns and '頭数' in df_features.columns:
        df_features['内枠'] = (df_features['枠番'] <= 3).astype(int)
        df_features['相対枠位置'] = df_features['枠番'] / df_features['頭数']
    
    # 4. 季節
    df_features['月'] = df_features['actual_date'].dt.month
    df_features['春'] = df_features['月'].apply(lambda x: 1 if 3 <= x <= 5 else 0)
    df_features['夏'] = df_features['月'].apply(lambda x: 1 if 6 <= x <= 8 else 0)
    df_features['秋'] = df_features['月'].apply(lambda x: 1 if 9 <= x <= 11 else 0)
    
    print(f"作成した特徴量数: {len(df_features.columns) - len(df.columns)}")
    
    return df_features


def run_evaluation(df_features):
    """モデル評価を実行"""
    # ターゲット作成
    df_features['target'] = (df_features['着順'] <= 3).astype(int)
    
    # 特徴量選択
    exclude_cols = ['着順', 'target', 'オッズ', '人気', '上がり', '走破時間', '通過順', 
                    '日付', 'actual_date', 'year', '月', 'race_id', 'race_id_str', '馬番']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # データを時系列順にソート
    df_features = df_features.sort_values('actual_date').reset_index(drop=True)
    
    X = df_features[feature_cols]
    y = df_features['target']
    
    print(f"\n使用する特徴量数: {len(feature_cols)}")
    print(f"正例（3着以内）の割合: {y.mean():.2%}")
    
    # 3分割の時系列交差検証
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("\n=== 時系列交差検証 ===")
    
    # ベースラインモデル
    print("\n[ベースライン: ロジスティック回帰]")
    baseline_scores = []
    scaler = StandardScaler()
    
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 前処理
        X_train = X_train.fillna(X_train.mean())
        X_valid = X_valid.fillna(X_train.mean())
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        # モデル学習
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        
        # 評価
        y_pred = lr_model.predict_proba(X_valid_scaled)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        baseline_scores.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
    
    print(f"  平均AUC: {np.mean(baseline_scores):.4f}")
    
    # LightGBM
    print("\n[改善版: LightGBM]")
    lgb_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 前処理
        X_train = X_train.fillna(X_train.mean())
        X_valid = X_valid.fillna(X_train.mean())
        
        # モデル学習
        model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        
        # 評価
        y_pred = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        lgb_scores.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
    
    print(f"  平均AUC: {np.mean(lgb_scores):.4f}")
    print(f"\n改善率: +{(np.mean(lgb_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100:.1f}%")
    
    # 特徴量重要度（最後のfoldのモデルを使用）
    print("\n=== 重要な特徴量トップ10 ===")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"{row['feature']:20} {row['importance']:>6}")
    
    return model, X, y, df_features


def main():
    """メイン処理"""
    print("=" * 50)
    print("競馬予測モデルの改善版（要約）")
    print("=" * 50)
    
    # データ読み込み
    df = load_and_prepare_data()
    
    # 特徴量作成
    df_features = create_features(df)
    
    # モデル評価
    model, X, y, df_features = run_evaluation(df_features)
    
    print("\n=== 実装した改善点 ===")
    print("✅ 1. 競馬ドメイン知識を活かした特徴量エンジニアリング")
    print("✅ 2. TimeSeriesSplitによる適切な交差検証")
    print("✅ 3. ベースラインモデルとの比較")
    print("✅ 4. 特徴量重要度の分析")
    
    print("\n✅ 完了しました！")


if __name__ == "__main__":
    main()