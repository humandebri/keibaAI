#!/usr/bin/env python3
"""
クリーンな全19万件データ活用システム（リファクタリング版）
オッズを使わない真の機械学習システム

設計原則:
- Single Responsibility Principle
- 高い拡張性と保守性
- 型安全性
- パフォーマンス最適化
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MLConfig:
    """機械学習設定"""
    random_state: int = 42
    test_size: float = 0.15
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = 'sqrt'
    class_weight: str = 'balanced'
    default_agari: float = 36.0
    default_jockey_rate: float = 0.08
    default_weight: float = 480.0


class DataProcessor:
    """データ処理専門クラス"""
    
    def __init__(self, config: MLConfig):
        self.config = config
    
    def load_training_data(self) -> Optional[pd.DataFrame]:
        """訓練データ読み込み"""
        try:
            df = pd.read_csv("encoded/2020_2025encoded_data_v2.csv")
            print(f"📊 データ読み込み完了: {len(df):,}件")
            print(f"   勝利ケース: {(df['着順'] == 1).sum():,}件")
            print(f"   期間: 2020-2025年")
            return df
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None
    
    def load_live_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """ライブデータ読み込み"""
        try:
            df = pd.read_csv(file_path)
            print(f"📊 ライブデータ読み込み: {len(df)}頭")
            return df
        except Exception as e:
            print(f"❌ ライブデータ読み込みエラー: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データクリーニング"""
        cleaned = df.copy()
        
        # 無限値・NaN処理
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.fillna(0)
        
        # 数値型変換
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').fillna(0)
        
        return cleaned


class JockeyTrainerStatsCalculator:
    """騎手・調教師統計計算専門クラス"""
    
    @staticmethod
    def calculate_jockey_stats(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """騎手統計計算"""
        print("🏇 騎手実績計算中...")
        
        jockey_stats = df.groupby('騎手').agg({
            '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
            '上がり': 'mean'
        }).round(4)
        
        # 列名を整理
        jockey_stats.columns = ['騎乗数', '勝利数', '複勝数', '平均上がり']
        jockey_stats['実勝率'] = jockey_stats['勝利数'] / jockey_stats['騎乗数']
        jockey_stats['実複勝率'] = jockey_stats['複勝数'] / jockey_stats['騎乗数']
        
        return {
            '騎乗数': jockey_stats['騎乗数'],
            '実勝率': jockey_stats['実勝率'],
            '実複勝率': jockey_stats['実複勝率'],
            '平均上がり': jockey_stats['平均上がり']
        }
    
    @staticmethod
    def calculate_trainer_stats(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """調教師統計計算"""
        print("👔 調教師実績計算中...")
        
        trainer_stats = df.groupby('調教師').agg({
            '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
            '体重': 'mean'
        }).round(4)
        
        trainer_stats.columns = ['管理頭数', '勝利数', '複勝数', '平均体重']
        trainer_stats['実勝率'] = trainer_stats['勝利数'] / trainer_stats['管理頭数']
        trainer_stats['実複勝率'] = trainer_stats['複勝数'] / trainer_stats['管理頭数']
        
        return {
            '管理頭数': trainer_stats['管理頭数'],
            '実勝率': trainer_stats['実勝率'],
            '実複勝率': trainer_stats['実複勝率'],
            '平均体重': trainer_stats['平均体重']
        }


class FeatureEngineer:
    """特徴量エンジニアリング専門クラス"""
    
    def __init__(self, config: MLConfig):
        self.config = config
    
    def get_base_features(self) -> List[str]:
        """基本特徴量リスト（オッズ・人気除外）"""
        return [
            '体重', '体重変化', '斤量', '上がり', '出走頭数', 
            '距離', 'クラス', '騎手の勝率', '性', '齢'
        ]
    
    def get_past_performance_features(self) -> List[str]:
        """過去成績特徴量"""
        features = []
        for i in range(1, 6):
            features.extend([f'着順{i}', f'距離{i}', f'通過順{i}', f'走破時間{i}'])
        return features
    
    def get_temporal_features(self) -> List[str]:
        """時系列特徴量"""
        return ['日付差1', '日付差2', '日付差3']
    
    def get_race_condition_features(self) -> List[str]:
        """レース条件特徴量"""
        return ['芝・ダート', '回り', '馬場', '天気', '場id']
    
    def create_enhanced_features(self, df: pd.DataFrame, 
                               jockey_stats: Dict[str, pd.Series],
                               trainer_stats: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, List[str]]:
        """拡張特徴量作成"""
        enhanced_df = df.copy()
        new_features = []
        
        # 過去成績分析
        new_features.extend(self._create_past_performance_features(enhanced_df))
        
        # 休養期間分析
        new_features.extend(self._create_rest_period_features(enhanced_df))
        
        # 距離適性分析
        new_features.extend(self._create_distance_aptitude_features(enhanced_df))
        
        # 騎手・調教師実績マッピング
        new_features.extend(self._map_jockey_trainer_stats(enhanced_df, jockey_stats, trainer_stats))
        
        # 枠番効果
        new_features.extend(self._create_gate_effect_features(enhanced_df))
        
        # 総合能力指標
        new_features.extend(self._create_composite_ability_features(enhanced_df))
        
        return enhanced_df, new_features
    
    def _create_past_performance_features(self, df: pd.DataFrame) -> List[str]:
        """過去成績特徴量作成"""
        features = []
        
        if all(f'着順{i}' in df.columns for i in range(1, 6)):
            past_positions = [df[f'着順{i}'].fillna(10) for i in range(1, 6)]
            past_pos_df = pd.concat(past_positions, axis=1)
            
            df['過去平均着順'] = past_pos_df.mean(axis=1)
            df['過去最高着順'] = past_pos_df.min(axis=1)
            df['勝利経験'] = (past_pos_df == 1).sum(axis=1)
            df['複勝経験'] = (past_pos_df <= 3).sum(axis=1)
            
            features.extend(['過去平均着順', '過去最高着順', '勝利経験', '複勝経験'])
        
        if all(f'走破時間{i}' in df.columns for i in range(1, 4)):
            past_times = [df[f'走破時間{i}'].fillna(120) for i in range(1, 4)]
            df['平均スピード'] = pd.concat(past_times, axis=1).mean(axis=1)
            features.append('平均スピード')
        
        # 着順改善傾向
        if all(f'着順{i}' in df.columns for i in range(1, 4)):
            df['着順改善傾向'] = (
                (df['着順2'].fillna(10) - df['着順1'].fillna(10)) + 
                (df['着順3'].fillna(10) - df['着順2'].fillna(10))
            ) / 2
            features.append('着順改善傾向')
        
        return features
    
    def _create_rest_period_features(self, df: pd.DataFrame) -> List[str]:
        """休養期間特徴量"""
        features = []
        
        if '日付差1' in df.columns:
            rest_days = df['日付差1'].fillna(30)
            df['休養適正'] = np.where(
                (rest_days >= 14) & (rest_days <= 90), 1, 0
            )
            df['長期休養'] = (rest_days > 90).astype(int)
            df['連闘'] = (rest_days < 14).astype(int)
            
            features.extend(['休養適正', '長期休養', '連闘'])
        
        return features
    
    def _create_distance_aptitude_features(self, df: pd.DataFrame) -> List[str]:
        """距離適性特徴量"""
        features = []
        
        if '距離' in df.columns:
            distance = df['距離'].fillna(1600)
            df['距離カテゴリ'] = pd.cut(
                distance, bins=[0, 1400, 1800, 2200, 3000], 
                labels=[1, 2, 3, 4]
            ).astype(float)
            
            # 同距離経験
            if all(f'距離{i}' in df.columns for i in range(1, 4)):
                same_dist_exp = sum(
                    (df[f'距離{i}'] == distance).astype(int).fillna(0) 
                    for i in range(1, 4)
                )
                df['同距離経験'] = same_dist_exp / 3
                features.append('同距離経験')
            
            features.append('距離カテゴリ')
        
        return features
    
    def _map_jockey_trainer_stats(self, df: pd.DataFrame,
                                 jockey_stats: Dict[str, pd.Series],
                                 trainer_stats: Dict[str, pd.Series]) -> List[str]:
        """騎手・調教師実績マッピング"""
        features = []
        
        # 騎手実績
        for stat_name, stat_series in jockey_stats.items():
            col_name = f'騎手{stat_name}'
            df[col_name] = df['騎手'].map(stat_series).fillna(
                stat_series.mean() if len(stat_series) > 0 else self.config.default_jockey_rate
            )
            features.append(col_name)
        
        # 調教師実績
        for stat_name, stat_series in trainer_stats.items():
            col_name = f'調教師{stat_name}'
            df[col_name] = df['調教師'].map(stat_series).fillna(
                stat_series.mean() if len(stat_series) > 0 else self.config.default_jockey_rate
            )
            features.append(col_name)
        
        return features
    
    def _create_gate_effect_features(self, df: pd.DataFrame) -> List[str]:
        """枠番効果特徴量"""
        features = []
        
        if '枠番' in df.columns:
            gate = df['枠番'].fillna(4)
            df['内枠'] = (gate <= 3).astype(int)
            df['外枠'] = (gate >= 7).astype(int)
            df['中枠'] = ((gate >= 4) & (gate <= 6)).astype(int)
            
            features.extend(['内枠', '外枠', '中枠'])
        
        return features
    
    def _create_composite_ability_features(self, df: pd.DataFrame) -> List[str]:
        """総合能力指標"""
        features = []
        
        ability_components = []
        
        if '過去平均着順' in df.columns:
            ability_components.append(10 - df['過去平均着順'])
        if '勝利経験' in df.columns:
            ability_components.append(df['勝利経験'])
        if '複勝経験' in df.columns:
            ability_components.append(df['複勝経験'] / 5)
        
        if len(ability_components) >= 3:
            df['総合能力指標'] = pd.concat(ability_components, axis=1).mean(axis=1)
            features.append('総合能力指標')
        
        return features


class LiveDataFeatureProcessor:
    """ライブデータ特徴量処理専門クラス"""
    
    def __init__(self, config: MLConfig):
        self.config = config
    
    def create_live_features(self, live_data: pd.DataFrame, 
                           feature_columns: List[str]) -> pd.DataFrame:
        """ライブデータの特徴量作成"""
        enhanced_df = self._preprocess_live_data(live_data)
        live_features = pd.DataFrame()
        
        feature_mappers = {
            '体重': lambda: enhanced_df['馬体重'].astype(float),
            '体重変化': lambda: self._process_weight_change(enhanced_df),
            '斤量': lambda: enhanced_df['斤量'].astype(float),
            '上がり': lambda: pd.Series([self.config.default_agari] * len(enhanced_df)),
            '出走頭数': lambda: pd.Series([len(enhanced_df)] * len(enhanced_df)),
            '距離': lambda: enhanced_df['distance'].astype(float),
            'クラス': lambda: self._map_class(enhanced_df['class']),
            '騎手の勝率': lambda: pd.Series([self.config.default_jockey_rate] * len(enhanced_df)),
            '性': lambda: self._map_sex(enhanced_df['性齢']),
            '齢': lambda: self._map_age(enhanced_df['性齢']),
            '芝・ダート': lambda: self._map_surface(enhanced_df['surface']),
            '枠番': lambda: enhanced_df['枠'].astype(int),
            '内枠': lambda: (enhanced_df['枠'].astype(int) <= 3).astype(int),
            '外枠': lambda: (enhanced_df['枠'].astype(int) >= 7).astype(int),
            '中枠': lambda: ((enhanced_df['枠'].astype(int) >= 4) & 
                           (enhanced_df['枠'].astype(int) <= 6)).astype(int)
        }
        
        for feature in feature_columns:
            if feature in feature_mappers:
                live_features[feature] = feature_mappers[feature]()
            else:
                live_features[feature] = self._get_default_value(feature, len(enhanced_df))
        
        return live_features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def _preprocess_live_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ライブデータ前処理"""
        enhanced = df.copy()
        return enhanced
    
    def _process_weight_change(self, df: pd.DataFrame) -> pd.Series:
        """体重変化処理"""
        weight_change = df['馬体重変化']
        if weight_change.dtype == 'object':
            return weight_change.astype(str).str.replace('+', '').astype(float)
        return weight_change.astype(float)
    
    def _map_class(self, class_series: pd.Series) -> pd.Series:
        """クラス情報マッピング"""
        mapping = {
            '新馬': 1, '未勝利': 2, '1勝クラス': 3, '2勝クラス': 4, 
            '3勝クラス': 5, 'オープン': 6, '4歳以上オープン': 6, 
            'G3': 7, 'G2': 8, 'G1': 9
        }
        return class_series.map(mapping).fillna(6)
    
    def _map_sex(self, sex_age_series: pd.Series) -> pd.Series:
        """性別マッピング"""
        mapping = {'牡': 1, '牝': 2, 'セ': 3}
        return sex_age_series.str[0].map(mapping).fillna(1)
    
    def _map_age(self, sex_age_series: pd.Series) -> pd.Series:
        """年齢マッピング"""
        return sex_age_series.str[1:].astype(int)
    
    def _map_surface(self, surface_series: pd.Series) -> pd.Series:
        """コース種別マッピング"""
        mapping = {'芝': 1, 'ダート': 2}
        return surface_series.map(mapping).fillna(1)
    
    def _get_default_value(self, feature: str, length: int) -> pd.Series:
        """デフォルト値設定"""
        defaults = {
            '着順': 5.5, '勝率': 0.08, '距離': 2000, '通過': 8.0,
            'タイム': 120.0, '体重': 480.0, '能力': 0.5, '適正': 1.0
        }
        
        for key, value in defaults.items():
            if key in feature:
                return pd.Series([value] * length)
        
        return pd.Series([0.5] * length)


class ModelTrainer:
    """モデル訓練専門クラス"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, float]:
        """モデル訓練"""
        print("🤖 モデル訓練開始")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        print(f"   訓練データ: {len(X_train):,}件")
        print(f"   検証データ: {len(X_test):,}件")
        
        # スケーリング
        print("   📊 特徴量スケーリング中...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # モデル訓練
        print("   🔄 RandomForest訓練中...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            oob_score=True,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 評価
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        oob_score = self.model.oob_score
        
        self.metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'oob_score': oob_score
        }
        
        print(f"✅ モデル訓練完了")
        print(f"   検証精度: {accuracy:.3f}")
        print(f"   検証AUC: {auc:.3f}")
        print(f"   OOB精度: {oob_score:.3f}")
        
        return accuracy, auc, oob_score
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.Series:
        """特徴量重要度取得"""
        if self.model is None:
            return pd.Series()
        
        importance = pd.Series(
            self.model.feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False)
        
        return importance


class Predictor:
    """予測専門クラス"""
    
    def __init__(self, model: RandomForestClassifier, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """予測実行"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def create_results_dataframe(self, live_data: pd.DataFrame, 
                                probabilities: np.ndarray) -> pd.DataFrame:
        """結果DataFrame作成"""
        results = live_data.copy()
        results['クリーンML勝利確率'] = probabilities
        
        # 期待値計算
        odds = results['単勝オッズ'].astype(float)
        results['クリーンML期待値'] = probabilities * odds
        
        # 着順予測
        results['クリーンML期待着順'] = (
            (1 - probabilities) * (len(results) + 1) / 2
        )
        results['クリーンML予測着順'] = results['クリーンML期待着順'].rank().astype(int)
        
        return results.sort_values('クリーンML勝利確率', ascending=False)


class ResultsDisplayer:
    """結果表示専門クラス"""
    
    @staticmethod
    def display_feature_importance(importance: pd.Series, top_n: int = 15):
        """特徴量重要度表示"""
        print(f"\n📊 特徴量重要度 Top {top_n}:")
        for feature, score in importance.head(top_n).items():
            print(f"      {feature}: {score:.4f}")
    
    @staticmethod
    def display_prediction_results(results: pd.DataFrame):
        """予測結果表示"""
        print("\n🎯 クリーンML予測結果（オッズ非使用）:")
        print("=" * 90)
        print(f"{'順位':>2} {'馬番':>3} {'馬名':>12} {'オッズ':>6} {'ML勝率':>7} {'期待値':>7} {'予測着順':>6}")
        print("=" * 90)
        
        for i, (_, horse) in enumerate(results.head(10).iterrows()):
            print(f"{i+1:2d}. {horse['馬番']:2d}番 {horse['馬名']:12s} "
                  f"{horse['単勝オッズ']:5.1f}倍 {horse['クリーンML勝利確率']*100:5.1f}% "
                  f"{horse['クリーンML期待値']:6.2f} {horse['クリーンML予測着順']:5d}着")
    
    @staticmethod
    def display_ranking_prediction(results: pd.DataFrame):
        """着順予測表示"""
        print(f"\n🏆 着順予測:")
        print("=" * 70)
        predicted_order = results.sort_values('クリーンML予測着順')
        
        for _, horse in predicted_order.head(8).iterrows():
            print(f"{horse['クリーンML予測着順']:2d}着予想: {horse['馬番']:2d}番 {horse['馬名']:12s} "
                  f"(勝率{horse['クリーンML勝利確率']*100:5.1f}% 期待値{horse['クリーンML期待値']:5.2f})")
    
    @staticmethod
    def display_investment_recommendation(results: pd.DataFrame):
        """投資推奨表示"""
        print(f"\n💰 投資推奨:")
        print("=" * 60)
        
        profitable = results[results['クリーンML期待値'] >= 1.0]
        
        if len(profitable) > 0:
            print(f"【期待値1.0以上】 {len(profitable)}頭")
            for _, horse in profitable.head(3).iterrows():
                confidence = ("超高" if horse['クリーンML期待値'] >= 1.4 
                            else "高" if horse['クリーンML期待値'] >= 1.2 else "中")
                print(f"  {horse['馬番']:2d}番 {horse['馬名']:12s} "
                      f"期待値{horse['クリーンML期待値']:5.2f} "
                      f"予測{horse['クリーンML予測着順']:2d}着 信頼度:{confidence}")
        else:
            print("期待値1.0以上の馬は見つかりませんでした")


class CleanMLSystem:
    """統合機械学習システム"""
    
    def __init__(self):
        self.config = MLConfig()
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.live_processor = LiveDataFeatureProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.stats_calculator = JockeyTrainerStatsCalculator()
        
        self.feature_columns = None
        self.jockey_stats = {}
        self.trainer_stats = {}
        self.is_trained = False
    
    def train_system(self) -> bool:
        """システム全体の訓練"""
        print("🚀 クリーンML訓練開始")
        print("💡 オッズを使わない真の機械学習")
        
        # 1. データ読み込み
        df = self.data_processor.load_training_data()
        if df is None:
            return False
        
        # 2. データクリーニング
        df = self.data_processor.clean_data(df)
        
        # 3. 統計計算
        self.jockey_stats = self.stats_calculator.calculate_jockey_stats(df)
        self.trainer_stats = self.stats_calculator.calculate_trainer_stats(df)
        
        # 4. 特徴量エンジニアリング
        print("🔧 特徴量エンジニアリング開始")
        
        # 基本特徴量選択
        base_features = self.feature_engineer.get_base_features()
        past_features = self.feature_engineer.get_past_performance_features()
        temporal_features = self.feature_engineer.get_temporal_features()
        race_features = self.feature_engineer.get_race_condition_features()
        
        # 存在する特徴量のみ選択
        available_features = [f for f in base_features + past_features + temporal_features + race_features 
                            if f in df.columns]
        
        # 拡張特徴量作成
        enhanced_df, new_features = self.feature_engineer.create_enhanced_features(
            df, self.jockey_stats, self.trainer_stats
        )
        
        # 最終特徴量選択
        self.feature_columns = [f for f in available_features + new_features 
                               if f in enhanced_df.columns and enhanced_df[f].dtype in ['int64', 'float64']]
        
        print(f"✅ 特徴量エンジニアリング完了")
        print(f"   最終特徴量数: {len(self.feature_columns)}個")
        print(f"   オッズ関連特徴量: 除外済み")
        
        # 5. モデル訓練
        X = enhanced_df[self.feature_columns]
        y = (enhanced_df['着順'] == 1).astype(int)
        
        accuracy, auc, oob_score = self.model_trainer.train(X, y)
        
        # 6. 特徴量重要度
        importance = self.model_trainer.get_feature_importance(self.feature_columns)
        ResultsDisplayer.display_feature_importance(importance)
        
        self.is_trained = True
        return True
    
    def predict_race(self, live_data_file: str) -> Optional[pd.DataFrame]:
        """レース予測"""
        if not self.is_trained:
            print("❌ モデルが訓練されていません")
            return None
        
        print("🏇 ライブ予測実行")
        
        # ライブデータ読み込み
        live_data = self.data_processor.load_live_data(live_data_file)
        if live_data is None:
            return None
        
        # 特徴量作成
        print("🎯 特徴量作成中...")
        live_features = self.live_processor.create_live_features(live_data, self.feature_columns)
        
        # 予測実行
        predictor = Predictor(self.model_trainer.model, self.model_trainer.scaler)
        predictions, probabilities = predictor.predict(live_features)
        
        # 結果作成
        results = predictor.create_results_dataframe(live_data, probabilities)
        
        print("✅ 予測完了")
        return results
    
    def run_complete_system(self, live_data_file: str = "live_race_data_202505021211.csv"):
        """完全システム実行"""
        print("🚀 クリーンML完全システム開始")
        
        # 訓練
        if not self.train_system():
            print("❌ 訓練失敗")
            return
        
        # 予測
        results = self.predict_race(live_data_file)
        if results is None:
            print("❌ 予測失敗")
            return
        
        # 結果表示
        ResultsDisplayer.display_prediction_results(results)
        ResultsDisplayer.display_ranking_prediction(results)
        ResultsDisplayer.display_investment_recommendation(results)
        
        # システム性能表示
        print(f"\n📊 システム性能:")
        print(f"   AUC: {self.model_trainer.metrics['auc']:.3f}")
        print(f"   精度: {self.model_trainer.metrics['accuracy']:.3f}")
        print(f"   特徴量数: {len(self.feature_columns)}個")
        print(f"   ⚡ 完全クリーンML")
        
        print("\n✅ クリーンML完全システム完了")


def main():
    """メイン実行"""
    system = CleanMLSystem()
    system.run_complete_system()


if __name__ == "__main__":
    main()