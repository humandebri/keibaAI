#!/usr/bin/env python3
"""
クリーンな全19万件データ活用システム
オッズを使わない真の機械学習システム
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CleanFullDataMLSystem:
    """オッズを使わない全データ活用機械学習システム"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.model_metrics = {}
        self.is_trained = False
        # 訓練時に学習した実績データを保存
        self.jockey_stats = {}
        self.trainer_stats = {}
        
    def load_full_training_data(self):
        """全19万件データ読み込み"""
        print("📊 全19万件データ読み込み開始")
        
        try:
            # 全データを読み込み
            df = pd.read_csv("encoded/2020_2025encoded_data_v2.csv")
            print(f"   読み込み完了: {len(df):,}件")
            print(f"   勝利ケース: {(df['着順'] == 1).sum():,}件 ({(df['着順'] == 1).mean():.3f})")
            print(f"   特徴量数: {len(df.columns)}個")
            print(f"   期間: 2020-2025年の全データ")
            
            return df
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None
    
    def create_clean_features(self, df: pd.DataFrame) -> tuple:
        """オッズを使わないクリーンな特徴量エンジニアリング"""
        print("🔧 クリーンな特徴量エンジニアリング開始")
        
        # ターゲット変数
        y = (df['着順'] == 1).astype(int)
        
        # 特徴量選択（オッズ関連を除外）
        selected_features = []
        
        # 1. 基本情報（オッズ除外）
        basic_features = [
            '体重', '体重変化', '斤量', '上がり', 
            '出走頭数', '距離', 'クラス', '騎手の勝率', '性', '齢'
        ]
        
        # 2. 過去成績（オッズ除外）
        past_features = []
        for i in range(1, 6):  # 過去5走
            past_features.extend([f'着順{i}', f'距離{i}', f'通過順{i}', f'走破時間{i}'])
        
        # 3. 時系列情報（休養期間）
        temporal_features = ['日付差1', '日付差2', '日付差3']
        
        # 4. 統計特徴量
        stat_features = ['平均スピード', '過去5走の合計賞金', '平均斤量']
        
        # 5. レース条件
        race_condition_features = ['芝・ダート', '回り', '馬場', '天気', '場id']
        
        # 実際に存在する特徴量のみ選択
        all_candidates = basic_features + past_features + temporal_features + stat_features + race_condition_features
        available_features = [col for col in all_candidates if col in df.columns]
        selected_features.extend(available_features)
        
        print(f"   基本特徴量: {len(available_features)}個")
        
        # 6. 新規計算特徴量（要求された特徴量を追加）
        enhanced_df = df.copy()
        
        # 馬の過去成績分析（オッズ除外）
        if all(f'着順{i}' in df.columns for i in range(1, 6)):
            past_positions = []
            past_times = []
            
            for i in range(1, 6):
                if f'着順{i}' in df.columns:
                    past_positions.append(df[f'着順{i}'].fillna(10))
                if f'走破時間{i}' in df.columns:
                    past_times.append(df[f'走破時間{i}'].fillna(120))
            
            if past_positions:
                past_pos_df = pd.concat(past_positions, axis=1)
                enhanced_df['過去平均着順'] = past_pos_df.mean(axis=1)
                enhanced_df['過去最高着順'] = past_pos_df.min(axis=1)
                enhanced_df['勝利経験'] = (past_pos_df == 1).sum(axis=1)
                enhanced_df['複勝経験'] = (past_pos_df <= 3).sum(axis=1)
                enhanced_df['着順安定性'] = past_pos_df.std(axis=1).fillna(5)
                enhanced_df['着順改善傾向'] = past_pos_df.iloc[:, 0] - past_pos_df.iloc[:, -1]  # 前走-最古走
                selected_features.extend(['過去平均着順', '過去最高着順', '勝利経験', '複勝経験', '着順安定性', '着順改善傾向'])
            
            if past_times:
                past_times_df = pd.concat(past_times, axis=1)
                enhanced_df['過去平均タイム'] = past_times_df.mean(axis=1)
                enhanced_df['過去最高タイム'] = past_times_df.min(axis=1)
                enhanced_df['タイム安定性'] = past_times_df.std(axis=1).fillna(10)
                selected_features.extend(['過去平均タイム', '過去最高タイム', 'タイム安定性'])
        
        # 前走からの休養期間（要求特徴量）
        if '日付差1' in df.columns:
            enhanced_df['休養期間'] = df['日付差1'].fillna(30)
            enhanced_df['休養適正'] = np.where(
                (enhanced_df['休養期間'] >= 14) & (enhanced_df['休養期間'] <= 60), 1.2,
                np.where(enhanced_df['休養期間'] < 14, 0.9, 0.8)
            )
            # 休養期間のカテゴリ化
            enhanced_df['休養カテゴリ'] = pd.cut(enhanced_df['休養期間'], 
                                          bins=[0, 7, 14, 30, 60, 180, 1000],
                                          labels=[1, 2, 3, 4, 5, 6]).astype(float)
            selected_features.extend(['休養期間', '休養適正', '休養カテゴリ'])
        
        # 距離適性分析（血統情報の代替）
        if '距離' in df.columns:
            current_distance = df['距離'].fillna(1600)
            enhanced_df['距離カテゴリ'] = pd.cut(current_distance, 
                                        bins=[0, 1400, 1800, 2200, 3000], 
                                        labels=[1, 2, 3, 4]).astype(float)
            
            # 同距離経験（血統の代替指標）
            if all(f'距離{i}' in df.columns for i in range(1, 4)):
                same_dist_exp = 0
                for i in range(1, 4):
                    same_dist_exp += (df[f'距離{i}'] == current_distance).astype(int).fillna(0)
                enhanced_df['同距離経験'] = same_dist_exp / 3
                
                # 距離変化の分析
                if '距離1' in df.columns:
                    enhanced_df['距離変化'] = current_distance - df['距離1'].fillna(current_distance)
                    enhanced_df['距離延長'] = (enhanced_df['距離変化'] > 200).astype(int)
                    enhanced_df['距離短縮'] = (enhanced_df['距離変化'] < -200).astype(int)
                    selected_features.extend(['距離変化', '距離延長', '距離短縮'])
                
                selected_features.extend(['距離カテゴリ', '同距離経験'])
        
        # 騎手・調教師詳細成績（全データから実際に計算）
        if '騎手' in df.columns:
            print("   🏇 全データから騎手実績計算中...")
            jockey_stats = df.groupby('騎手').agg({
                '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
                '上がり': 'mean'
            }).round(4)
            jockey_stats.columns = ['騎乗数', '勝利数', '複勝数', '平均上がり']
            jockey_stats['実勝率'] = jockey_stats['勝利数'] / jockey_stats['騎乗数']
            jockey_stats['実複勝率'] = jockey_stats['複勝数'] / jockey_stats['騎乗数']
            
            # 実績データをクラス変数に保存（予測時に使用）
            self.jockey_stats = {
                '実勝率': dict(zip(jockey_stats.index, jockey_stats['実勝率'])),
                '実複勝率': dict(zip(jockey_stats.index, jockey_stats['実複勝率'])),
                '騎乗数': dict(zip(jockey_stats.index, jockey_stats['騎乗数'])),
                '平均上がり': dict(zip(jockey_stats.index, jockey_stats['平均上がり']))
            }
            
            enhanced_df['騎手実勝率'] = df['騎手'].map(jockey_stats['実勝率']).fillna(0.08)
            enhanced_df['騎手実複勝率'] = df['騎手'].map(jockey_stats['実複勝率']).fillna(0.25)
            enhanced_df['騎手騎乗数'] = df['騎手'].map(jockey_stats['騎乗数']).fillna(50)
            enhanced_df['騎手平均上がり'] = df['騎手'].map(jockey_stats['平均上がり']).fillna(35.0)
            selected_features.extend(['騎手実勝率', '騎手実複勝率', '騎手騎乗数', '騎手平均上がり'])
        
        if '調教師' in df.columns:
            print("   👔 全データから調教師実績計算中...")
            trainer_stats = df.groupby('調教師').agg({
                '着順': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
                '体重': 'mean'
            }).round(4)
            trainer_stats.columns = ['管理数', '勝利数', '複勝数', '平均体重']
            trainer_stats['実勝率'] = trainer_stats['勝利数'] / trainer_stats['管理数']
            trainer_stats['実複勝率'] = trainer_stats['複勝数'] / trainer_stats['管理数']
            
            # 実績データをクラス変数に保存（予測時に使用）
            self.trainer_stats = {
                '実勝率': dict(zip(trainer_stats.index, trainer_stats['実勝率'])),
                '実複勝率': dict(zip(trainer_stats.index, trainer_stats['実複勝率'])),
                '管理数': dict(zip(trainer_stats.index, trainer_stats['管理数'])),
                '平均体重': dict(zip(trainer_stats.index, trainer_stats['平均体重']))
            }
            
            enhanced_df['調教師実勝率'] = df['調教師'].map(trainer_stats['実勝率']).fillna(0.06)
            enhanced_df['調教師実複勝率'] = df['調教師'].map(trainer_stats['実複勝率']).fillna(0.20)
            enhanced_df['調教師管理数'] = df['調教師'].map(trainer_stats['管理数']).fillna(100)
            enhanced_df['調教師平均体重'] = df['調教師'].map(trainer_stats['平均体重']).fillna(480)
            selected_features.extend(['調教師実勝率', '調教師実複勝率', '調教師管理数', '調教師平均体重'])
        
        # コーナー通過順位分析（要求特徴量）
        if any(f'通過順{i}' in df.columns for i in range(1, 4)):
            corner_positions = []
            for i in range(1, 4):
                if f'通過順{i}' in df.columns:
                    corner_positions.append(df[f'通過順{i}'].fillna(8))
            
            if corner_positions:
                corner_df = pd.concat(corner_positions, axis=1)
                enhanced_df['平均通過順'] = corner_df.mean(axis=1)
                enhanced_df['4角位置'] = corner_positions[0] if corner_positions else 8
                enhanced_df['位置取り能力'] = 10 - enhanced_df['平均通過順']  # 高いほど良い
                enhanced_df['通過順安定性'] = corner_df.std(axis=1).fillna(3)
                enhanced_df['前半後半差'] = corner_df.iloc[:, -1] - corner_df.iloc[:, 0] if len(corner_positions) >= 2 else 0
                selected_features.extend(['平均通過順', '4角位置', '位置取り能力', '通過順安定性', '前半後半差'])
        
        # 枠番詳細分析（要求特徴量）
        if '枠' in df.columns:
            enhanced_df['枠番'] = df['枠'].fillna(4)
            enhanced_df['内枠'] = (enhanced_df['枠番'] <= 3).astype(int)
            enhanced_df['外枠'] = (enhanced_df['枠番'] >= 7).astype(int)
            enhanced_df['中枠'] = ((enhanced_df['枠番'] >= 4) & (enhanced_df['枠番'] <= 6)).astype(int)
            
            # 距離別枠有利度
            if '距離' in df.columns:
                distance = df['距離'].fillna(1600)
                enhanced_df['枠距離適性'] = np.where(
                    distance <= 1400,
                    np.where(enhanced_df['枠番'] <= 4, 1.1, 0.9),  # 短距離は内枠有利
                    np.where(distance >= 2000,
                             np.where(enhanced_df['枠番'] >= 5, 1.05, 0.95),  # 長距離は外枠やや有利
                             1.0)  # 中距離は中立
                )
            selected_features.extend(['枠番', '内枠', '外枠', '中枠', '枠距離適性'])
        
        # 体重・体調分析
        if '体重' in df.columns and '体重変化' in df.columns:
            enhanced_df['体重適正'] = ((enhanced_df['体重'] >= 450) & (enhanced_df['体重'] <= 520)).astype(int)
            enhanced_df['体重変化絶対値'] = abs(enhanced_df['体重変化'].fillna(0))
            enhanced_df['体重増加'] = (enhanced_df['体重変化'] > 5).astype(int)
            enhanced_df['体重減少'] = (enhanced_df['体重変化'] < -5).astype(int)
            selected_features.extend(['体重適正', '体重変化絶対値', '体重増加', '体重減少'])
        
        # 年齢・性別分析
        if '齢' in df.columns:
            # 数値型に変換
            age = pd.to_numeric(enhanced_df['齢'], errors='coerce').fillna(4)
            enhanced_df['年齢ピーク'] = ((age >= 4) & (age <= 5)).astype(int)
            enhanced_df['若馬'] = (age == 3).astype(int)
            enhanced_df['古馬'] = (age >= 6).astype(int)
            selected_features.extend(['年齢ピーク', '若馬', '古馬'])
        
        # 総合指標（オッズ使わない）
        ability_components = []
        if '過去平均着順' in enhanced_df.columns:
            ability_components.append((10 - enhanced_df['過去平均着順']) / 10)
        if '騎手実勝率' in enhanced_df.columns:
            ability_components.append(enhanced_df['騎手実勝率'] * 5)
        if '位置取り能力' in enhanced_df.columns:
            ability_components.append(enhanced_df['位置取り能力'] / 10)
        if '調教師実勝率' in enhanced_df.columns:
            ability_components.append(enhanced_df['調教師実勝率'] * 5)
        if '過去最高タイム' in enhanced_df.columns:
            ability_components.append((130 - enhanced_df['過去最高タイム']) / 20)  # タイムを能力指標に
        
        if len(ability_components) >= 3:
            enhanced_df['総合能力指標'] = pd.concat(ability_components, axis=1).mean(axis=1)
            selected_features.append('総合能力指標')
        
        # 人気関連特徴量を除外（純粋な機械学習のため）
        # オッズも人気も使わない完全クリーンなML
        
        # 最終特徴量マトリックス作成
        final_features = []
        for col in selected_features:
            if col in enhanced_df.columns:
                final_features.append(col)
        
        X = enhanced_df[final_features].copy()
        
        # データクリーニング
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # 特徴量の標準化前チェック
        X = X.select_dtypes(include=[np.number])  # 数値型のみ
        
        self.feature_columns = list(X.columns)
        
        print(f"✅ クリーンな特徴量エンジニアリング完了")
        print(f"   最終特徴量数: {len(self.feature_columns)}個")
        print(f"   主要特徴量: {self.feature_columns[:10]}")
        print(f"   オッズ関連特徴量: 除外済み")
        
        return X, y
    
    def train_clean_model(self, X: pd.DataFrame, y: pd.Series):
        """クリーンなモデル訓練"""
        print("🤖 クリーンなモデル訓練開始")
        
        # 訓練・検証分割（大規模データ用）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"   訓練データ: {X_train.shape[0]:,}件")
        print(f"   検証データ: {X_test.shape[0]:,}件") 
        print(f"   特徴量数: {X_train.shape[1]}個")
        print(f"   正例の割合: {y_train.mean():.3f}")
        
        # スケーリング
        print("   📊 特徴量スケーリング中...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # クリーンなRandomForest（オッズに依存しない）
        self.model = RandomForestClassifier(
            n_estimators=300,  # オッズなしなので木を増やす
            max_depth=20,      # より深い学習が必要
            min_samples_split=80,
            min_samples_leaf=40,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            oob_score=True
        )
        
        # 訓練
        print("   🔄 クリーンなRandomForest訓練中...")
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
            'training_samples': len(X_train),
            'total_data_used': len(X)
        }
        
        self.is_trained = True
        
        # 特徴量重要度
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ クリーンなモデル訓練完了")
        print(f"   検証精度: {test_accuracy:.3f}")
        print(f"   検証AUC: {test_auc:.3f}")
        print(f"   OOB精度: {oob_score:.3f}")
        
        print(f"\n📊 クリーンな特徴量重要度 Top 15:")
        for _, row in self.feature_importance.head(15).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def predict_with_clean_ml(self, live_race_data: pd.DataFrame):
        """クリーンなML予測"""
        if not self.is_trained:
            print("❌ モデルが訓練されていません")
            return None
        
        print("🎯 クリーンなML予測実行")
        
        # ライブデータ特徴量作成
        live_features = self._create_clean_live_features(live_race_data)
        
        if live_features is None:
            return None
        
        # 予測
        X_live_scaled = self.scaler.transform(live_features)
        win_probabilities = self.model.predict_proba(X_live_scaled)[:, 1]
        
        # 確率正規化
        win_probabilities = win_probabilities / win_probabilities.sum()
        
        # 結果作成（オッズは表示のみ使用）
        results = live_race_data.copy()
        results['クリーンML勝利確率'] = win_probabilities
        results['クリーンML期待値'] = win_probabilities * results['単勝オッズ'].astype(float)
        
        # 着順予測
        results['クリーンML期待着順'] = (
            win_probabilities * 1 +
            (1 - win_probabilities) * (len(results) + 1) / 2
        )
        results['クリーンML予測着順'] = results['クリーンML期待着順'].rank().astype(int)
        
        # ソート（確率順）
        results = results.sort_values('クリーンML勝利確率', ascending=False)
        
        print("✅ クリーンなML予測完了")
        return results
    
    def _create_clean_live_features(self, live_race_data: pd.DataFrame):
        """クリーンなライブ特徴量作成（オッズ使わない）"""
        # 基本変換（CSVの実際の列名を使用）
        enhanced_df = live_race_data.copy()
        enhanced_df['馬体重_数値'] = enhanced_df['馬体重'].astype(float)
        
        if enhanced_df['馬体重変化'].dtype == 'object':
            enhanced_df['馬体重変化_数値'] = enhanced_df['馬体重変化'].astype(str).str.replace('+', '').astype(float)
        else:
            enhanced_df['馬体重変化_数値'] = enhanced_df['馬体重変化'].astype(float)
        
        # CSVの列名を標準化
        enhanced_df['距離_数値'] = enhanced_df['distance'].astype(float)
        enhanced_df['クラス_数値'] = enhanced_df['class']
        enhanced_df['芝ダート_数値'] = enhanced_df['surface']
        enhanced_df['斤量_数値'] = enhanced_df['斤量'].astype(float)
        
        # 訓練時特徴量に合わせたマッピング（オッズ・人気除外）
        live_features = pd.DataFrame()
        
        for feature in self.feature_columns:
            if feature == '体重':
                live_features[feature] = enhanced_df['馬体重_数値']
            elif feature == '体重変化':
                live_features[feature] = enhanced_df['馬体重変化_数値']
            elif feature == '出走頭数':
                live_features[feature] = len(enhanced_df)
            elif feature == '距離':
                live_features[feature] = enhanced_df['距離_数値']
            elif feature == 'クラス':
                # クラス情報を数値化
                class_mapping = {
                    '新馬': 1, '未勝利': 2, '1勝クラス': 3, '2勝クラス': 4, '3勝クラス': 5,
                    'オープン': 6, '4歳以上オープン': 6, 'G3': 7, 'G2': 8, 'G1': 9
                }
                live_features[feature] = enhanced_df['クラス_数値'].map(class_mapping).fillna(6)
            elif feature == '斤量':
                live_features[feature] = enhanced_df['斤量_数値']
            elif feature == '上がり':
                # 上がりタイムは通常36.0前後なのでデフォルト設定
                live_features[feature] = 36.0
            elif feature == '騎手の勝率':
                # 騎手の一般的な勝率デフォルト
                live_features[feature] = 0.08
            elif feature == '枠番':
                live_features[feature] = enhanced_df['枠'].astype(int)
            elif feature == '内枠':
                live_features[feature] = (enhanced_df['枠'].astype(int) <= 3).astype(int)
            elif feature == '外枠':
                live_features[feature] = (enhanced_df['枠'].astype(int) >= 7).astype(int)
            elif feature == '中枠':
                live_features[feature] = ((enhanced_df['枠'].astype(int) >= 4) & (enhanced_df['枠'].astype(int) <= 6)).astype(int)
            # 訓練時に学習した実際の騎手・調教師実績を使用
            elif feature == '騎手実勝率':
                if self.jockey_stats and '実勝率' in self.jockey_stats:
                    live_features[feature] = enhanced_df['騎手'].map(self.jockey_stats['実勝率']).fillna(0.08)
                else:
                    live_features[feature] = 0.08
            elif feature == '騎手実複勝率':
                if self.jockey_stats and '実複勝率' in self.jockey_stats:
                    live_features[feature] = enhanced_df['騎手'].map(self.jockey_stats['実複勝率']).fillna(0.25)
                else:
                    live_features[feature] = 0.25
            elif feature == '騎手騎乗数':
                if self.jockey_stats and '騎乗数' in self.jockey_stats:
                    live_features[feature] = enhanced_df['騎手'].map(self.jockey_stats['騎乗数']).fillna(50)
                else:
                    live_features[feature] = 50
            elif feature == '騎手平均上がり':
                if self.jockey_stats and '平均上がり' in self.jockey_stats:
                    live_features[feature] = enhanced_df['騎手'].map(self.jockey_stats['平均上がり']).fillna(35.0)
                else:
                    live_features[feature] = 35.0
            elif feature == '調教師実勝率':
                if self.trainer_stats and '実勝率' in self.trainer_stats:
                    live_features[feature] = enhanced_df['調教師'].map(self.trainer_stats['実勝率']).fillna(0.06)
                else:
                    live_features[feature] = 0.06
            elif feature == '調教師実複勝率':
                if self.trainer_stats and '実複勝率' in self.trainer_stats:
                    live_features[feature] = enhanced_df['調教師'].map(self.trainer_stats['実複勝率']).fillna(0.20)
                else:
                    live_features[feature] = 0.20
            elif feature == '調教師管理数':
                if self.trainer_stats and '管理数' in self.trainer_stats:
                    live_features[feature] = enhanced_df['調教師'].map(self.trainer_stats['管理数']).fillna(100)
                else:
                    live_features[feature] = 100
            elif feature == '調教師平均体重':
                if self.trainer_stats and '平均体重' in self.trainer_stats:
                    live_features[feature] = enhanced_df['調教師'].map(self.trainer_stats['平均体重']).fillna(480)
                else:
                    live_features[feature] = 480
            elif feature == '性':
                # 性別を数値化（牡=1, 牝=2, セ=3）
                if '性齢' in enhanced_df.columns:
                    sex_map = {'牡': 1, '牝': 2, 'セ': 3}
                    enhanced_df['性_数値'] = enhanced_df['性齢'].str[0].map(sex_map).fillna(1)
                    live_features[feature] = enhanced_df['性_数値']
                else:
                    live_features[feature] = 1
            elif feature == '齢':
                # 年齢を数値化
                if '性齢' in enhanced_df.columns:
                    enhanced_df['齢_数値'] = enhanced_df['性齢'].str[1:].astype(int)
                    live_features[feature] = enhanced_df['齢_数値']
                else:
                    live_features[feature] = 4
            elif feature == '芝・ダート':
                # 芝=1, ダート=2
                surface_map = {'芝': 1, 'ダート': 2}
                live_features[feature] = enhanced_df['芝ダート_数値'].map(surface_map).fillna(1)
            else:
                # その他のデフォルト値（統計的に妥当な値）
                if '着順' in feature:
                    live_features[feature] = 5.5
                elif '勝率' in feature:
                    live_features[feature] = 0.08
                elif '距離' in feature:
                    live_features[feature] = 2000
                elif '通過' in feature:
                    live_features[feature] = 8.0
                elif 'タイム' in feature:
                    live_features[feature] = 120.0
                elif '体重' in feature:
                    live_features[feature] = 480.0
                elif '能力' in feature:
                    live_features[feature] = 0.5
                elif '適正' in feature:
                    live_features[feature] = 1.0
                else:
                    live_features[feature] = 0.5
        
        # データクリーニング
        live_features = live_features.fillna(0)
        live_features = live_features.replace([np.inf, -np.inf], 0)
        
        return live_features
    
    def run_clean_system(self, race_data_file: str):
        """クリーンなシステム実行"""
        print("🚀 クリーンな19万件データシステム実行開始")
        print("💡 オッズを使わない真の機械学習による予測")
        
        # 1. 全データ読み込み
        df = self.load_full_training_data()
        if df is None:
            return None
        
        # 2. クリーンな特徴量エンジニアリング
        X, y = self.create_clean_features(df)
        if X is None:
            return None
        
        # 3. クリーンなモデル訓練
        model = self.train_clean_model(X, y)
        if model is None:
            return None
        
        # 4. ライブ予測
        print(f"\n🏇 クリーンなライブ予測実行")
        race_data = pd.read_csv(race_data_file)
        print(f"📊 レースデータ: {len(race_data)}頭")
        
        results = self.predict_with_clean_ml(race_data)
        if results is None:
            return None
        
        # 5. 結果表示
        print("\n🎯 クリーンML予測結果（オッズ非使用）:")
        print("="*130)
        print(f"{'順位':>2} {'馬番':>3} {'馬名':>12} {'オッズ':>6} {'クリーンML勝率':>11} {'クリーンML期待値':>13} {'ML着順':>6}")
        print("="*110)
        
        for i, (_, horse) in enumerate(results.head(10).iterrows()):
            print(f"{i+1:2d}. {horse['馬番']:2d}番 {horse['馬名']:12s} "
                  f"{horse['単勝オッズ']:5.1f}倍 {horse['クリーンML勝利確率']*100:8.1f}% "
                  f"{horse['クリーンML期待値']:10.2f} {horse['クリーンML予測着順']:5d}着")
        
        # 着順予測
        print(f"\n🏆 クリーンML着順予測:")
        print("="*90)
        predicted_order = results.sort_values('クリーンML予測着順')
        for _, horse in predicted_order.head(8).iterrows():
            print(f"{horse['クリーンML予測着順']:2d}着予想: {horse['馬番']:2d}番 {horse['馬名']:12s} "
                  f"(クリーンML勝率{horse['クリーンML勝利確率']*100:5.1f}% 期待値{horse['クリーンML期待値']:5.2f})")
        
        # 投資推奨
        print(f"\n💰 クリーンML投資推奨:")
        print("="*80)
        
        profitable = results[results['クリーンML期待値'] >= 1.0]
        
        if len(profitable) > 0:
            print(f"【期待値1.0以上】 {len(profitable)}頭")
            for _, horse in profitable.head(3).iterrows():
                confidence = "超高" if horse['クリーンML期待値'] >= 1.4 else "高" if horse['クリーンML期待値'] >= 1.2 else "中"
                print(f"  {horse['馬番']:2d}番 {horse['馬名']:12s} クリーンML期待値{horse['クリーンML期待値']:5.2f} "
                      f"予測{horse['クリーンML予測着順']:2d}着 信頼度:{confidence}")
            
            best = profitable.iloc[0]
            print(f"\n💡 クリーンML最推奨: {best['馬番']}番{best['馬名']} (期待値{best['クリーンML期待値']:.2f})")
        else:
            print("❌ 期待値1.0以上の馬なし")
            top_predicted = predicted_order.iloc[0]
            print(f"💡 着順重視推奨: {top_predicted['馬番']}番{top_predicted['馬名']} "
                  f"(1着予想、クリーンML勝率{top_predicted['クリーンML勝利確率']*100:.1f}%)")
        
        # システム性能
        print(f"\n📊 クリーンシステム性能:")
        print(f"   AUC: {self.model_metrics['test_auc']:.3f}")
        print(f"   精度: {self.model_metrics['test_accuracy']:.3f}")
        print(f"   OOB精度: {self.model_metrics['oob_score']:.3f}")
        print(f"   特徴量数: {self.model_metrics['feature_count']}個")
        print(f"   訓練サンプル: {self.model_metrics['training_samples']:,}件")
        print(f"   総データ活用: {self.model_metrics['total_data_used']:,}件")
        print(f"   ⚡ オッズ非依存の真の機械学習")
        
        print(f"\n✅ クリーンな19万件データMLシステム完了")
        return results


def main():
    """実行"""
    system = CleanFullDataMLSystem()
    results = system.run_clean_system("live_race_data_202505021211.csv")


if __name__ == "__main__":
    main()