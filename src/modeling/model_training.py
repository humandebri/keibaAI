#!/usr/bin/env python3
"""
Clean ML Model Training
オッズを使わない真の機械学習モデル訓練システム
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class CleanModelTrainer:
    """クリーンな機械学習モデル訓練クラス"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.model_metrics = {}
        self.jockey_stats = {}
        self.trainer_stats = {}
        
    def load_training_data(self, data_path="encoded/2020_2025encoded_data_v2.csv"):
        """訓練データを読み込み"""
        print("📊 訓練データ読み込み開始")
        
        try:
            df = pd.read_csv(data_path)
            print(f"   データ読み込み完了: {len(df):,}件")
            print(f"   勝利ケース: {(df['着順'] == 1).sum():,}件 ({(df['着順'] == 1).mean():.3f})")
            print(f"   特徴量数: {len(df.columns)}個")
            
            return df
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None
    
    def create_clean_features(self, df: pd.DataFrame) -> tuple:
        """オッズを使わないクリーンな特徴量エンジニアリング"""
        print("🔧 クリーンな特徴量エンジニアリング開始")
        
        # ターゲット変数
        y = (df['着順'] == 1).astype(int)
        
        # 基本特徴量（オッズ除外）
        basic_features = [
            '人気', '体重', '体重変化', '斤量', '上がり', 
            '出走頭数', '距離', 'クラス', '騎手の勝率', '性', '齢'
        ]
        
        # 過去成績（オッズ除外）
        past_features = []
        for i in range(1, 6):
            past_features.extend([f'着順{i}', f'距離{i}', f'通過順{i}', f'走破時間{i}'])
        
        # 時系列・統計・レース条件
        temporal_features = ['日付差1', '日付差2', '日付差3']
        stat_features = ['平均スピード', '過去5走の合計賞金', '平均斤量']
        race_features = ['芝・ダート', '回り', '馬場', '天気', '場id']
        
        # 使用可能特徴量
        all_candidates = basic_features + past_features + temporal_features + stat_features + race_features
        available_features = [col for col in all_candidates if col in df.columns]
        
        enhanced_df = df.copy()
        selected_features = available_features.copy()
        
        # 馬の過去成績分析
        if all(f'着順{i}' in df.columns for i in range(1, 6)):
            past_positions = [df[f'着順{i}'].fillna(10) for i in range(1, 6)]
            if past_positions:
                past_pos_df = pd.concat(past_positions, axis=1)
                enhanced_df['過去平均着順'] = past_pos_df.mean(axis=1)
                enhanced_df['過去最高着順'] = past_pos_df.min(axis=1)
                enhanced_df['勝利経験'] = (past_pos_df == 1).sum(axis=1)
                enhanced_df['複勝経験'] = (past_pos_df <= 3).sum(axis=1)
                enhanced_df['着順安定性'] = past_pos_df.std(axis=1).fillna(5)
                selected_features.extend(['過去平均着順', '過去最高着順', '勝利経験', '複勝経験', '着順安定性'])
        
        # 休養期間分析
        if '日付差1' in df.columns:
            enhanced_df['休養期間'] = df['日付差1'].fillna(30)
            enhanced_df['休養適正'] = np.where(
                (enhanced_df['休養期間'] >= 14) & (enhanced_df['休養期間'] <= 60), 1.2,
                np.where(enhanced_df['休養期間'] < 14, 0.9, 0.8)
            )
            selected_features.extend(['休養期間', '休養適正'])
        
        # 距離適性分析
        if '距離' in df.columns:
            current_distance = df['距離'].fillna(1600)
            enhanced_df['距離カテゴリ'] = pd.cut(current_distance, 
                                        bins=[0, 1400, 1800, 2200, 3000], 
                                        labels=[1, 2, 3, 4]).astype(float)
            selected_features.append('距離カテゴリ')
        
        # 騎手・調教師詳細成績
        if '騎手' in df.columns:
            print("   🏇 騎手実績計算中...")
            jockey_stats = df.groupby('騎手').agg({
                '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
            }).round(4)
            jockey_stats.columns = ['騎乗数', '勝利数', '複勝数']
            jockey_stats['実勝率'] = jockey_stats['勝利数'] / jockey_stats['騎乗数']
            jockey_stats['実複勝率'] = jockey_stats['複勝数'] / jockey_stats['騎乗数']
            
            self.jockey_stats = {
                '実勝率': dict(zip(jockey_stats.index, jockey_stats['実勝率'])),
                '実複勝率': dict(zip(jockey_stats.index, jockey_stats['実複勝率'])),
                '騎乗数': dict(zip(jockey_stats.index, jockey_stats['騎乗数']))
            }
            
            enhanced_df['騎手実勝率'] = df['騎手'].map(jockey_stats['実勝率']).fillna(0.08)
            enhanced_df['騎手実複勝率'] = df['騎手'].map(jockey_stats['実複勝率']).fillna(0.25)
            selected_features.extend(['騎手実勝率', '騎手実複勝率'])
        
        if '調教師' in df.columns:
            print("   👔 調教師実績計算中...")
            trainer_stats = df.groupby('調教師').agg({
                '着順': ['count', lambda x: (x == 1).sum()]
            }).round(4)
            trainer_stats.columns = ['管理数', '勝利数']
            trainer_stats['実勝率'] = trainer_stats['勝利数'] / trainer_stats['管理数']
            
            self.trainer_stats = {
                '実勝率': dict(zip(trainer_stats.index, trainer_stats['実勝率'])),
                '管理数': dict(zip(trainer_stats.index, trainer_stats['管理数']))
            }
            
            enhanced_df['調教師実勝率'] = df['調教師'].map(trainer_stats['実勝率']).fillna(0.06)
            selected_features.append('調教師実勝率')
        
        # 枠番分析
        if '枠' in df.columns:
            enhanced_df['枠番'] = df['枠'].fillna(4)
            enhanced_df['内枠'] = (enhanced_df['枠番'] <= 3).astype(int)
            enhanced_df['外枠'] = (enhanced_df['枠番'] >= 7).astype(int)
            selected_features.extend(['枠番', '内枠', '外枠'])
        
        # 人気のみの市場評価
        if '人気' in df.columns:
            enhanced_df['本命'] = (enhanced_df['人気'] <= 3).astype(int)
            enhanced_df['大穴'] = (enhanced_df['人気'] >= 9).astype(int)
            enhanced_df['人気逆数'] = 1.0 / enhanced_df['人気'].fillna(9)
            selected_features.extend(['本命', '大穴', '人気逆数'])
        
        # 最終特徴量マトリックス
        final_features = [col for col in selected_features if col in enhanced_df.columns]
        X = enhanced_df[final_features].copy()
        
        # データクリーニング
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        X = X.select_dtypes(include=[np.number])
        
        self.feature_columns = list(X.columns)
        
        print(f"✅ 特徴量エンジニアリング完了")
        print(f"   最終特徴量数: {len(self.feature_columns)}個")
        print(f"   オッズ関連特徴量: 除外済み")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, optimize=True):
        """モデル訓練"""
        print("🤖 クリーンモデル訓練開始")
        
        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"   訓練データ: {X_train.shape[0]:,}件")
        print(f"   検証データ: {X_test.shape[0]:,}件")
        print(f"   特徴量数: {X_train.shape[1]}個")
        print(f"   正例の割合: {y_train.mean():.3f}")
        
        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize:
            # ハイパーパラメータ最適化
            print("   🔍 ハイパーパラメータ最適化中...")
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 20, 25],
                'min_samples_split': [50, 80, 100],
                'min_samples_leaf': [25, 40, 50],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                oob_score=True
            )
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"   最適パラメータ: {grid_search.best_params_}")
            print(f"   CV AUC: {grid_search.best_score_:.3f}")
        else:
            # デフォルトパラメータ
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=80,
                min_samples_leaf=40,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                oob_score=True
            )
            self.model.fit(X_train_scaled, y_train)
        
        # 評価
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)
        oob_score = self.model.oob_score_
        
        self.model_metrics = {
            'model_name': 'CleanRandomForest',
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'oob_score': oob_score,
            'feature_count': len(self.feature_columns),
            'training_samples': len(X_train)
        }
        
        # 特徴量重要度
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ モデル訓練完了")
        print(f"   検証精度: {test_accuracy:.3f}")
        print(f"   検証AUC: {test_auc:.3f}")
        print(f"   OOB精度: {oob_score:.3f}")
        
        print(f"\n📊 特徴量重要度 Top 10:")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def save_model(self, model_dir="models"):
        """モデル保存"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics,
            'jockey_stats': self.jockey_stats,
            'trainer_stats': self.trainer_stats
        }
        
        filepath = f"{model_dir}/clean_model_{timestamp}.pkl"
        joblib.dump(model_data, filepath)
        
        # 最新モデルとしても保存
        latest_filepath = f"{model_dir}/clean_model_latest.pkl"
        joblib.dump(model_data, latest_filepath)
        
        print(f"💾 モデル保存完了:")
        print(f"   {filepath}")
        print(f"   {latest_filepath}")
        
        return filepath
    
    def load_model(self, filepath):
        """モデル読み込み"""
        print(f"📂 モデル読み込み: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.model_metrics = model_data['model_metrics']
            self.jockey_stats = model_data.get('jockey_stats', {})
            self.trainer_stats = model_data.get('trainer_stats', {})
            
            print(f"✅ モデル読み込み完了")
            print(f"   AUC: {self.model_metrics.get('test_auc', 'N/A')}")
            print(f"   特徴量数: {len(self.feature_columns)}個")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False
    
    def run_training_pipeline(self, data_path="encoded/2020_2025encoded_data_v2.csv", optimize=True):
        """完全な訓練パイプライン実行"""
        print("🚀 クリーン機械学習訓練パイプライン開始")
        print("💡 オッズを使わない真の機械学習モデル")
        
        # 1. データ読み込み
        df = self.load_training_data(data_path)
        if df is None:
            return None
        
        # 2. 特徴量エンジニアリング
        X, y = self.create_clean_features(df)
        if X is None:
            return None
        
        # 3. モデル訓練
        model = self.train_model(X, y, optimize=optimize)
        if model is None:
            return None
        
        # 4. モデル保存
        model_path = self.save_model()
        
        print(f"\n📊 訓練完了サマリー:")
        print(f"   モデル: {self.model_metrics['model_name']}")
        print(f"   AUC: {self.model_metrics['test_auc']:.3f}")
        print(f"   精度: {self.model_metrics['test_accuracy']:.3f}")
        print(f"   OOB精度: {self.model_metrics['oob_score']:.3f}")
        print(f"   特徴量数: {self.model_metrics['feature_count']}個")
        print(f"   訓練サンプル: {self.model_metrics['training_samples']:,}件")
        print(f"   ⚡ オッズ非依存の真の機械学習")
        
        print(f"\n✅ クリーン機械学習訓練パイプライン完了")
        return model_path


def main():
    """メイン実行"""
    trainer = CleanModelTrainer()
    model_path = trainer.run_training_pipeline(optimize=False)  # 高速実行
    
    if model_path:
        print(f"\n🎉 訓練が完了しました！")
        print(f"モデルファイル: {model_path}")


if __name__ == "__main__":
    main()