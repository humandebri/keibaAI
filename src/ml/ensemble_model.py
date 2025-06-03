#!/usr/bin/env python3
"""
アンサンブル学習モデル
複数のアルゴリズムを組み合わせて予測精度を向上
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


class EnsembleRacePredictor:
    """複数モデルを組み合わせた競馬予測器"""
    
    def __init__(self, use_deep_learning: bool = True):
        self.use_deep_learning = use_deep_learning
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        
    def create_models(self) -> Dict:
        """各種モデルを作成"""
        models = {
            'lightgbm': lgb.LGBMRegressor(
                objective='regression',
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            ),
            
            'xgboost': XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            
            'elastic_net': ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42
            )
        }
        
        if self.use_deep_learning:
            models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        return models
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """交差検証でモデルを訓練し、重みを決定"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 各モデルのCVスコアを記録
        cv_scores = {name: [] for name in self.models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 各モデルで予測
            fold_predictions = {}
            
            for name, model in self.models.items():
                # モデルのコピーを作成して訓練
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                # 検証データで予測
                pred = model_clone.predict(X_val)
                fold_predictions[name] = pred
                
                # スコア計算（順位相関）
                pred_ranks = pd.Series(pred).rank()
                true_ranks = pd.Series(y_val).rank()
                score = pred_ranks.corr(true_ranks, method='spearman')
                cv_scores[name].append(score)
        
        # 平均スコアから重みを計算
        avg_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}
        
        # 負のスコアは0に
        avg_scores = {name: max(0, score) for name, score in avg_scores.items()}
        
        # 重みを正規化
        total_score = sum(avg_scores.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in avg_scores.items()}
        else:
            # 全て同じ重み
            self.model_weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        return cv_scores
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """アンサンブルモデルを訓練"""
        # モデルを作成
        self.models = self.create_models()
        
        # 交差検証で重みを決定
        print("交差検証でモデルの重みを決定中...")
        cv_scores = self.train_with_cv(X, y, n_splits=5)
        
        # 最終的なモデルを全データで訓練
        print("\n全データでモデルを訓練中...")
        for name, model in self.models.items():
            print(f"  {name}: ", end='', flush=True)
            model.fit(X, y)
            print(f"完了 (重み: {self.model_weights[name]:.3f})")
            
            # 特徴量重要度を取得
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # CVスコアのサマリー
        print("\n交差検証スコア:")
        for name, scores in cv_scores.items():
            print(f"  {name}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """アンサンブル予測"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.model_weights.get(name, 0)
            predictions.append(pred * weight)
        
        # 重み付き平均
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def train(self, data: pd.DataFrame, feature_cols: List[str], 
              target_col: str, race_id_col: str) -> Dict:
        """データフレームからモデルを訓練（統一システム用）"""
        # 特徴量とターゲットを準備
        X = data[feature_cols].values
        y = data[target_col].values
        
        # モデルを訓練
        self.fit(X, y)
        
        return self.models
    
    def predict_proba(self, X: np.ndarray, n_horses: int) -> np.ndarray:
        """各馬の勝率を予測"""
        # 着順予測
        predictions = self.predict(X)
        
        # 予測着順を確率に変換（順位が低いほど高確率）
        # ソフトマックス的な変換
        exp_neg_pred = np.exp(-predictions / 2)  # 温度パラメータ2
        probabilities = exp_neg_pred / exp_neg_pred.sum()
        
        return probabilities
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """特徴量重要度のサマリーを取得"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # 各モデルの重要度を重み付き平均
        importance_dict = {}
        
        for name, importance in self.feature_importance.items():
            weight = self.model_weights.get(name, 0)
            if weight > 0:
                if not importance_dict:
                    importance_dict = {i: imp * weight for i, imp in enumerate(importance)}
                else:
                    for i, imp in enumerate(importance):
                        importance_dict[i] += imp * weight
        
        # DataFrameに変換
        importance_df = pd.DataFrame({
            'feature_index': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        return importance_df.sort_values('importance', ascending=False)


class StackedEnsemble:
    """スタッキングアンサンブル"""
    
    def __init__(self, base_models: Dict, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=0.01)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """スタッキングモデルを訓練"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training base model: {name}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # モデルのコピーを作成
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                # OOF予測
                oof_predictions[val_idx, i] = model_clone.predict(X_val)
        
        # メタモデルを訓練
        print("Training meta model...")
        self.meta_model.fit(oof_predictions, y)
        
        # ベースモデルを全データで再訓練
        print("Retraining base models on full data...")
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """スタッキング予測"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # ベースモデルの予測
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, i] = model.predict(X)
        
        # メタモデルで最終予測
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions