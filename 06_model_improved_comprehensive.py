#!/usr/bin/env python3
"""
競馬予測モデルの包括的改善
- 競馬ドメイン知識を活かした特徴量エンジニアリング
- TimeSeriesSplitによる適切な交差検証
- ベースラインモデルとの比較
- Optunaによるハイパーパラメータ最適化
- SHAP値によるモデル解釈性分析
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


def load_race_data():
    """エンコード済みデータを読み込む"""
    possible_paths = [
        'encoded/encoded_data.csv',
        'encoded/2022_2023encoded_data.csv',
        'encoded/2022encoded_data.csv',
        'encoded/2023encoded_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"データを読み込みました: {path}")
            print(f"データサイズ: {df.shape}")
            
            # race_idから実際の日付を抽出
            if 'race_id' in df.columns:
                # race_idを文字列に変換し、最初の8文字を抽出
                df['race_id_str'] = df['race_id'].astype(str).str.replace('.0', '')
                df['actual_date'] = pd.to_datetime(df['race_id_str'].str[:8], format='%Y%m%d', errors='coerce')
                
                # 日付が正しく変換されたか確認
                valid_dates = df['actual_date'].notna()
                print(f"\n日付変換成功率: {valid_dates.sum() / len(df) * 100:.1f}%")
                
                if valid_dates.sum() > 0:
                    df = df[valid_dates].copy()
                    print(f"有効なデータ数: {len(df)}")
            
            return df
    
    raise FileNotFoundError("エンコード済みデータが見つかりません")


def create_domain_features(df):
    """競馬のドメイン知識を活かした特徴量を作成"""
    df_features = df.copy()
    
    print("\n=== 特徴量エンジニアリング開始 ===")
    
    # 1. 馬の能力指標
    if '前走からの間隔' in df_features.columns:
        # 適度な休養（3-5週）フラグ
        df_features['適度な休養'] = df_features['前走からの間隔'].apply(
            lambda x: 1 if 21 <= x <= 35 else 0 if pd.notna(x) else 0
        )
        # 長期休養明け（12週以上）フラグ
        df_features['長期休養明け'] = df_features['前走からの間隔'].apply(
            lambda x: 1 if x >= 84 else 0 if pd.notna(x) else 0
        )
    
    # 2. 過去成績の集約特徴量
    if '前走着順' in df_features.columns:
        df_features['前走勝利'] = (df_features['前走着順'] == 1).astype(int)
        df_features['前走連対'] = (df_features['前走着順'] <= 2).astype(int)
        df_features['前走着内'] = (df_features['前走着順'] <= 3).astype(int)
    
    # 3. 距離適性
    if '距離' in df_features.columns:
        df_features['短距離'] = (df_features['距離'] <= 1400).astype(int)
        df_features['マイル'] = ((df_features['距離'] > 1400) & (df_features['距離'] <= 1800)).astype(int)
        df_features['中距離'] = ((df_features['距離'] > 1800) & (df_features['距離'] <= 2400)).astype(int)
        df_features['長距離'] = (df_features['距離'] > 2400).astype(int)
    
    # 4. 馬場状態の影響
    if '馬場' in df_features.columns:
        baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        df_features['馬場状態数値'] = df_features['馬場'].map(baba_map).fillna(0)
        df_features['重馬場'] = (df_features['馬場状態数値'] >= 2).astype(int)
    
    # 5. 斤量の影響
    if '斤量' in df_features.columns and '性別' in df_features.columns:
        df_features['牡馬'] = (df_features['性別'] == '牡').astype(int)
        df_features['牝馬'] = (df_features['性別'] == '牝').astype(int)
        # 斤量負担
        df_features['斤量負担'] = df_features.apply(
            lambda row: row['斤量'] - 57 if row['牡馬'] else row['斤量'] - 55,
            axis=1
        )
    
    # 6. 枠順の影響
    if '枠番' in df_features.columns and '頭数' in df_features.columns:
        df_features['内枠'] = (df_features['枠番'] <= 3).astype(int)
        df_features['外枠'] = (df_features['枠番'] >= 7).astype(int)
        df_features['相対枠位置'] = df_features['枠番'] / df_features['頭数']
    
    # 7. 季節・時期の影響
    if 'actual_date' in df_features.columns:
        df_features['月'] = df_features['actual_date'].dt.month
        df_features['春'] = df_features['月'].apply(lambda x: 1 if 3 <= x <= 5 else 0)
        df_features['夏'] = df_features['月'].apply(lambda x: 1 if 6 <= x <= 8 else 0)
        df_features['秋'] = df_features['月'].apply(lambda x: 1 if 9 <= x <= 11 else 0)
        df_features['冬'] = df_features['月'].apply(lambda x: 1 if x == 12 or x <= 2 else 0)
    
    # 8. 人気と実力の乖離
    if '単勝' in df_features.columns and '人気' in df_features.columns:
        df_features['人気オッズ乖離'] = df_features['人気'] / (np.log1p(df_features['単勝']) + 1)
    
    # 9. コース特性
    if 'コース' in df_features.columns:
        df_features['芝'] = df_features['コース'].str.contains('芝', na=False).astype(int)
        df_features['ダート'] = df_features['コース'].str.contains('ダ', na=False).astype(int)
        
        if '競馬場' in df_features.columns:
            left_courses = ['東京', '中京', '新潟']
            df_features['左回り'] = df_features['競馬場'].apply(
                lambda x: 1 if any(course in str(x) for course in left_courses) else 0
            )
    
    # 10. 騎手の影響
    if '騎手' in df_features.columns and '着順' in df_features.columns:
        # 上位騎手フラグ
        top_jockeys = df_features[df_features['着順'] == 1]['騎手'].value_counts().head(20).index
        df_features['上位騎手'] = df_features['騎手'].isin(top_jockeys).astype(int)
    
    created_features = len(df_features.columns) - len(df.columns)
    print(f"作成した特徴量数: {created_features}個")
    
    return df_features


def run_cross_validation(X, y, df_features):
    """時系列交差検証を実行"""
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n=== 時系列交差検証の分割 ===")
    for i, (train_idx, valid_idx) in enumerate(tscv.split(X)):
        train_dates = df_features.iloc[train_idx]['actual_date']
        valid_dates = df_features.iloc[valid_idx]['actual_date']
        print(f"\nFold {i+1}:")
        print(f"  訓練: {train_dates.min().date()} ~ {train_dates.max().date()} ({len(train_idx):,}件)")
        print(f"  検証: {valid_dates.min().date()} ~ {valid_dates.max().date()} ({len(valid_idx):,}件)")
    
    # ベースラインモデル（ロジスティック回帰）
    print("\n=== ベースラインモデル（ロジスティック回帰）の評価 ===")
    baseline_scores = []
    scaler = StandardScaler()
    
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 欠損値処理
        X_train = X_train.fillna(X_train.mean())
        X_valid = X_valid.fillna(X_train.mean())
        
        # 標準化
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        # クラス重み
        class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
        
        # モデル学習
        lr_model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        
        # 評価
        y_pred = lr_model.predict_proba(X_valid_scaled)[:, 1]
        auc_score = roc_auc_score(y_valid, y_pred)
        baseline_scores.append(auc_score)
        
        print(f"Fold {fold+1} AUC: {auc_score:.4f}")
    
    print(f"\nベースライン平均AUC: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
    
    # LightGBMモデル
    print("\n=== LightGBMモデルの評価 ===")
    lgb_scores = []
    lgb_models = []
    
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 欠損値処理
        X_train = X_train.fillna(X_train.mean())
        X_valid = X_valid.fillna(X_train.mean())
        
        # クラス重み
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # パラメータ
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'n_estimators': 300
        }
        
        # モデル学習
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_valid, y_valid)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        # 評価
        y_pred = model.predict_proba(X_valid)[:, 1]
        auc_score = roc_auc_score(y_valid, y_pred)
        lgb_scores.append(auc_score)
        lgb_models.append(model)
        
        print(f"Fold {fold+1} AUC: {auc_score:.4f}")
    
    print(f"\nLightGBM平均AUC: {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
    print(f"改善率: {(np.mean(lgb_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100:.1f}%")
    
    return tscv, baseline_scores, lgb_scores, lgb_models


def optimize_hyperparameters(X, y, tscv, n_trials=20):
    """Optunaによるハイパーパラメータ最適化"""
    print("\n=== Optunaによるハイパーパラメータ最適化 ===")
    print(f"試行回数: {n_trials}")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        cv_scores = []
        
        for train_idx, valid_idx in tscv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            X_train = X_train.fillna(X_train.mean())
            X_valid = X_valid.fillna(X_train.mean())
            
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.LGBMClassifier(**params, n_estimators=100)
            model.fit(X_train, y_train,
                     eval_set=[(X_valid, y_valid)],
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            
            y_pred = model.predict_proba(X_valid)[:, 1]
            cv_scores.append(roc_auc_score(y_valid, y_pred))
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n最適なAUCスコア: {study.best_value:.4f}")
    print("\n最適なパラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def analyze_with_shap(model, X_test, sample_size=1000):
    """SHAP値によるモデル解釈"""
    print("\n=== SHAP値によるモデル解釈 ===")
    
    # サンプリング
    sample_size = min(sample_size, len(X_test))
    sample_idx = np.random.choice(X_test.index, size=sample_size, replace=False)
    X_sample = X_test.loc[sample_idx]
    
    # SHAP値計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n重要な特徴量トップ20:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:30} 重要度: {row['importance']:.4f}")
    
    # プロット
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("特徴量重要度（SHAP値）", fontsize=16)
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("特徴量の影響度分析", fontsize=16)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance


def main():
    """メイン処理"""
    print("=" * 50)
    print("競馬予測モデルの包括的改善")
    print("=" * 50)
    
    # データ読み込み
    df = load_race_data()
    
    # データ期間の確認
    if 'actual_date' in df.columns:
        print(f"\n=== 時系列データの確認 ===")
        print(f"データ期間: {df['actual_date'].min()} ~ {df['actual_date'].max()}")
        df['year'] = df['actual_date'].dt.year
        print(f"\n年別レコード数:")
        print(df['year'].value_counts().sort_index())
    
    # 特徴量エンジニアリング
    df_features = create_domain_features(df)
    
    # ターゲット作成
    df_features['target'] = (df_features['着順'] <= 3).astype(int)
    
    # 特徴量選択
    exclude_cols = ['着順', 'target', 'オッズ', '人気', '上がり', '走破時間', '通過順', 
                    '日付', 'actual_date', 'year', '月', 'race_id', 'race_id_str', '馬番']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"\n使用する特徴量数: {len(feature_cols)}")
    print(f"ターゲット分布:")
    print(df_features['target'].value_counts())
    print(f"正例（3着以内）の割合: {df_features['target'].mean():.2%}")
    
    # データを時系列順にソート
    df_features = df_features.sort_values('actual_date').reset_index(drop=True)
    
    # 特徴量とターゲット
    X = df_features[feature_cols]
    y = df_features['target']
    
    # 交差検証
    tscv, baseline_scores, lgb_scores, lgb_models = run_cross_validation(X, y, df_features)
    
    # ハイパーパラメータ最適化
    study = optimize_hyperparameters(X, y, tscv, n_trials=20)
    
    # 最終モデルの学習
    print("\n=== 最終モデルの学習 ===")
    split_point = int(len(X) * 0.8)
    X_train_final = X.iloc[:split_point]
    y_train_final = y.iloc[:split_point]
    X_test_final = X.iloc[split_point:]
    y_test_final = y.iloc[split_point:]
    
    # 欠損値処理
    X_train_final = X_train_final.fillna(X_train_final.mean())
    X_test_final = X_test_final.fillna(X_train_final.mean())
    
    # 最適化されたパラメータでモデル学習
    optimized_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': 300,
        'scale_pos_weight': (y_train_final == 0).sum() / (y_train_final == 1).sum(),
        **study.best_params
    }
    
    final_model = lgb.LGBMClassifier(**optimized_params)
    final_model.fit(X_train_final, y_train_final)
    
    # 評価
    y_pred_final = final_model.predict_proba(X_test_final)[:, 1]
    final_auc = roc_auc_score(y_test_final, y_pred_final)
    
    print(f"テストAUC: {final_auc:.4f}")
    
    # SHAP分析
    feature_importance = analyze_with_shap(final_model, X_test_final)
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("=== モデル性能比較 ===")
    print(f"\n1. ベースライン（ロジスティック回帰）:")
    print(f"   平均AUC: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
    
    print(f"\n2. LightGBM（初期パラメータ）:")
    print(f"   平均AUC: {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
    print(f"   改善率: +{(np.mean(lgb_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100:.1f}%")
    
    print(f"\n3. LightGBM（最適化後）:")
    print(f"   テストAUC: {final_auc:.4f}")
    print(f"   ベースラインからの改善: +{(final_auc - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100:.1f}%")
    
    print("\n=== 実装した改善点 ===")
    print("✅ 1. 競馬ドメイン知識を活かした特徴量エンジニアリング")
    print(f"   - 作成した新規特徴量: {len(df_features.columns) - len(df.columns)}個")
    print("✅ 2. TimeSeriesSplitによる適切な交差検証")
    print("✅ 3. ベースラインモデルとの比較")
    print("✅ 4. Optunaによるハイパーパラメータ最適化")
    print("✅ 5. SHAP値によるモデル解釈性分析")
    
    # モデルとパラメータを保存
    import joblib
    joblib.dump(final_model, 'model/improved_model.pkl')
    joblib.dump(optimized_params, 'model/optimized_params.pkl')
    joblib.dump(feature_cols, 'model/feature_cols.pkl')
    
    print("\n✅ モデルを保存しました: model/improved_model.pkl")


if __name__ == "__main__":
    main()