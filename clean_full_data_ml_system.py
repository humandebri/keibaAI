#!/usr/bin/env python3
"""
改善版クリーンMLシステム
効率的な過去データ活用
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, CatBoostRanker, Pool
import joblib
import warnings
import os
import pickle
warnings.filterwarnings('ignore')


@dataclass
class MLConfig:
    """機械学習設定"""
    random_state: int = 42
    test_size: float = 0.15
    n_folds: int = 5
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0
    od_type: str = 'Iter'
    od_wait: int = 300
    class_weight_ratio: float = 8.0
    default_agari: float = 40.0
    default_jockey_rate: float = 0.05
    default_weight: float = 480.0
    missing_penalty_std: float = 2.0


class HorseDatabase:
    """馬の過去成績データベース（効率的な実装）"""
    
    def __init__(self):
        self.db_file = "cache/horse_database.pkl"
        self.horse_data = {}
        self.jockey_stats = {}
        self.jockey_context_stats = {}  # コンテキスト別騎手統計
        self.jockey_time_stats = {}     # 時系列騎手統計
        self.jockey_synergy_stats = {}  # シナジー統計
        self.trainer_stats = {}
        
    def build_database(self, years: List[int] = [2020, 2021, 2022, 2023, 2024, 2025]):
        """データベース構築（キャッシュ付き）- 6年分のデータを使用"""
        # キャッシュが存在する場合は読み込み
        if os.path.exists(self.db_file):
            print("   📂 キャッシュからデータベース読み込み中...")
            try:
                with open(self.db_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # キャッシュのバージョン確認（5年分のデータか確認）
                    if cache_data.get('years', []) == years and cache_data.get('version', 1) >= 2:
                        self.horse_data = cache_data['horse_data']
                        self.jockey_stats = cache_data['jockey_stats']
                        self.jockey_context_stats = cache_data.get('jockey_context_stats', {})
                        self.jockey_time_stats = cache_data.get('jockey_time_stats', {})
                        self.jockey_synergy_stats = cache_data.get('jockey_synergy_stats', {})
                        self.trainer_stats = cache_data['trainer_stats']
                        print(f"   ✅ {len(self.horse_data)}頭のデータを読み込み（6年分・拡張騎手統計付き）")
                        return
                    else:
                        print("   ⚠️ キャッシュが古いため再構築します")
            except Exception as e:
                print(f"   ⚠️ キャッシュ読み込みエラー: {e}")
        
        print("   🔨 6年分のデータベース構築中...")
        all_data = []
        
        # 6年分のデータを読み込み（data_with_payoutを優先）
        for year in years:
            # data_with_payout を優先的に読み込む
            file_loaded = False
            # data_with_payoutのファイル名パターン
            file_patterns = [
                (f'data_with_payout/{year}_with_payout.xlsx', 'payout'),
                (f'data/{year}.xlsx', 'regular')
            ]
            
            for file_path, file_type in file_patterns:
                if os.path.exists(file_path):
                    try:
                        print(f"   読み込み中: {file_path}")
                        df = pd.read_excel(file_path)
                        all_data.append(df)
                        print(f"   ✅ {year}年: {len(df)}件 ({file_type})")
                        file_loaded = True
                        break
                    except Exception as e:
                        print(f"   ⚠️ {year}年読み込みエラー: {e}")
            
            if not file_loaded:
                print(f"   ⚠️ {year}年のデータが見つかりません")
        
        if not all_data:
            print("   ⚠️ データが見つかりません")
            return
        
        # データ結合と集計
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # データクリーニング
        if '着順' in combined_df.columns:
            combined_df['着順'] = pd.to_numeric(combined_df['着順'], errors='coerce')
            combined_df = combined_df.dropna(subset=['着順'])
        
        # 日付処理
        if '日付' in combined_df.columns:
            combined_df['日付'] = pd.to_datetime(combined_df['日付'], errors='coerce')
        
        # 拡張騎手統計の計算
        self._calculate_jockey_stats(combined_df)
        self._calculate_jockey_context_stats(combined_df)
        self._calculate_jockey_time_stats(combined_df)
        self._calculate_jockey_synergy_stats(combined_df)
        
        # 調教師統計
        if '調教師' in combined_df.columns and '着順' in combined_df.columns:
            trainer_group = combined_df.groupby('調教師')['着順']
            self.trainer_stats = {
                trainer: {
                    'win_rate': (group == 1).sum() / len(group) if len(group) > 0 else 0.08,
                    'place_rate': (group <= 3).sum() / len(group) if len(group) > 0 else 0.25,
                    'count': len(group)
                }
                for trainer, group in trainer_group
            }
        
        # 馬ごとのデータ（最新20レースを保持して精度向上）
        print("   🐴 馬ごとの詳細成績を集計中...")
        horse_count = 0
        for horse_name in combined_df['馬'].unique():
            if pd.notna(horse_name):
                horse_races = combined_df[combined_df['馬'] == horse_name].copy()
                if '日付' in horse_races.columns:
                    horse_races['日付'] = pd.to_datetime(horse_races['日付'], errors='coerce')
                    horse_races = horse_races.sort_values('日付', ascending=False).head(20)
                
                # 中間日数の計算
                recent_dates = horse_races['日付'].dropna().tolist()
                days_between_races = []
                if len(recent_dates) >= 2:
                    for i in range(len(recent_dates) - 1):
                        days_diff = (recent_dates[i] - recent_dates[i+1]).days
                        days_between_races.append(days_diff)
                
                # より多くの情報を保持して精度向上
                self.horse_data[horse_name] = {
                    'recent_positions': horse_races['着順'].tolist()[:10],  # 10レース分に拡張
                    'recent_agari': horse_races['上がり'].dropna().tolist()[:10],
                    'recent_distances': horse_races['距離'].tolist()[:10],
                    'recent_times': horse_races['走破時間'].tolist()[:10],
                    'recent_classes': horse_races['クラス'].tolist()[:5] if 'クラス' in horse_races.columns else [],
                    'recent_surfaces': horse_races['芝・ダート'].tolist()[:5] if '芝・ダート' in horse_races.columns else [],
                    'recent_dates': recent_dates[:10],  # 日付情報を追加
                    'days_between_races': days_between_races[:9],  # 中間日数（最大9個）
                    'avg_position': horse_races['着順'].mean(),
                    'best_position': horse_races['着順'].min(),
                    'win_count': (horse_races['着順'] == 1).sum(),
                    'place_count': (horse_races['着順'] <= 3).sum(),
                    'race_count': len(horse_races),
                    'avg_agari': horse_races['上がり'].dropna().mean() if len(horse_races['上がり'].dropna()) > 0 else None,
                    'best_agari': horse_races['上がり'].dropna().min() if len(horse_races['上がり'].dropna()) > 0 else None,
                    'raw_data': horse_races  # 元データも保持
                }
                horse_count += 1
                if horse_count % 1000 == 0:
                    print(f"      {horse_count}頭処理完了...")
        
        # キャッシュ保存
        os.makedirs('cache', exist_ok=True)
        cache_data = {
            'horse_data': self.horse_data,
            'jockey_stats': self.jockey_stats,
            'jockey_context_stats': self.jockey_context_stats,
            'jockey_time_stats': self.jockey_time_stats,
            'jockey_synergy_stats': self.jockey_synergy_stats,
            'trainer_stats': self.trainer_stats,
            'years': years,
            'version': 2  # 拡張騎手統計を含むバージョン
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"   ✅ データベース構築完了: {len(self.horse_data)}頭（6年分）")
    
    def get_horse_features(self, horse_name: str, current_date=None) -> Dict[str, Any]:
        """馬の特徴量を取得（詳細版）"""
        if horse_name not in self.horse_data:
            return {}
        
        data = self.horse_data[horse_name]
        features = {
            '過去平均着順': data['avg_position'],
            '過去最高着順': data['best_position'],
            '勝利経験': data['win_count'],
            '複勝経験': data['place_count'],
            '過去レース数': data['race_count']
        }
        
        # 最近の着順（5走分）
        for i, pos in enumerate(data['recent_positions'][:5], 1):
            features[f'着順{i}'] = pos
        
        # 最近の距離（5走分）
        for i, dist in enumerate(data['recent_distances'][:5], 1):
            features[f'距離{i}'] = dist
            
        # 最近の走破時間（5走分）
        for i, time_str in enumerate(data['recent_times'][:5], 1):
            features[f'走破時間{i}'] = self._parse_time(time_str)
            
        # 上がり関連
        if data.get('avg_agari') is not None:
            features['平均上がり'] = data['avg_agari']
        if data.get('best_agari') is not None:
            features['最高上がり'] = data['best_agari']
        
        # 前走からの中間日数
        if current_date and data.get('recent_dates'):
            recent_dates = data['recent_dates']
            if recent_dates:
                last_race_date = recent_dates[0]
                if pd.notna(last_race_date):
                    days_since_last = (current_date - last_race_date).days
                    features['前走からの日数'] = days_since_last
                    
                    # 放牧区分（休み明けカテゴリ）
                    if days_since_last <= 14:
                        features['放牧区分'] = 0  # 連闘～2週
                    elif days_since_last <= 28:
                        features['放牧区分'] = 1  # 3-4週（通常）
                    elif days_since_last <= 56:
                        features['放牧区分'] = 2  # 5-8週（中間隔）
                    elif days_since_last <= 84:
                        features['放牧区分'] = 3  # 9-12週（やや長期）
                    else:
                        features['放牧区分'] = 4  # 13週以上（長期休養）
                else:
                    features['前走からの日数'] = 60
                    features['放牧区分'] = 3
            else:
                features['前走からの日数'] = 180  # 初出走
                features['放牧区分'] = 5  # 初出走カテゴリ
        
        # 中間日数の統計
        if data.get('days_between_races'):
            days_between = data['days_between_races']
            if days_between:
                features['平均中間日数'] = np.mean(days_between)
                features['中間日数標準偏差'] = np.std(days_between) if len(days_between) > 1 else 0
                # 最近3走の中間日数
                for i, days in enumerate(days_between[:3], 1):
                    features[f'中間日数{i}'] = days
            else:
                features['平均中間日数'] = 30
                features['中間日数標準偏差'] = 0
                
        # 元データも返す（詳細分析用）
        features['raw_data'] = data.get('raw_data', pd.DataFrame())
        
        return features
    
    def _parse_time(self, time_str):
        """走破時間を秒に変換"""
        if pd.isna(time_str) or time_str == '':
            return 120.0
        try:
            if isinstance(time_str, (int, float)):
                return float(time_str)
            # "2:05.3" 形式を秒に変換
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except:
            return 120.0
    
    def get_jockey_stats(self, jockey_name: str) -> Dict[str, float]:
        """騎手統計を取得（基本統計）"""
        if jockey_name in self.jockey_stats:
            return self.jockey_stats[jockey_name]
        return {
            'win_rate': 0.08, 
            'place_rate': 0.25, 
            'count': 100,
            'avg_position': 8.0,
            'best_position': 3,
            'roi': 1.0
        }
    
    def get_jockey_context_stats(self, jockey_name: str, context_type: str, context_value: Any) -> Dict[str, float]:
        """コンテキスト別騎手統計を取得"""
        key = f"{jockey_name}_{context_value}"
        if context_type in self.jockey_context_stats and key in self.jockey_context_stats[context_type]:
            return self.jockey_context_stats[context_type][key]
        # デフォルト値として基本統計を返す
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_jockey_time_stats(self, jockey_name: str, window: int = 30) -> Dict[str, float]:
        """時系列騎手統計を取得"""
        key = f"{jockey_name}_{window}d"
        if key in self.jockey_time_stats:
            return self.jockey_time_stats[key]
        # デフォルト値
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_jockey_streak_stats(self, jockey_name: str) -> Dict[str, float]:
        """騎手の連続成績統計を取得"""
        key = f"{jockey_name}_streak"
        if key in self.jockey_time_stats:
            return self.jockey_time_stats[key]
        return {
            'cold_streak': 0,
            'last_win_days': 30
        }
    
    def get_jockey_synergy_stats(self, jockey_name: str, trainer_name: str) -> Dict[str, float]:
        """騎手×調教師のシナジー統計を取得"""
        key = f"{jockey_name}_{trainer_name}"
        if key in self.jockey_synergy_stats:
            return self.jockey_synergy_stats[key]
        # デフォルト値
        return {
            'win_rate': self.jockey_stats.get(jockey_name, {}).get('win_rate', 0.08),
            'place_rate': self.jockey_stats.get(jockey_name, {}).get('place_rate', 0.25),
            'count': 0
        }
    
    def get_trainer_stats(self, trainer_name: str) -> Dict[str, float]:
        """調教師統計を取得"""
        if trainer_name in self.trainer_stats:
            return self.trainer_stats[trainer_name]
        return {'win_rate': 0.08, 'place_rate': 0.25, 'count': 50}
    
    def _calculate_jockey_stats(self, df: pd.DataFrame):
        """基本騎手統計の計算（実力系）"""
        if '騎手' not in df.columns or '着順' not in df.columns:
            return
        
        print("   🏇 騎手統計計算中...")
        
        # オッズ情報があれば ROI も計算
        has_odds = 'オッズ' in df.columns
        
        for jockey, group in df.groupby('騎手'):
            stats = {
                'win_rate': (group['着順'] == 1).mean(),
                'place_rate': (group['着順'] <= 3).mean(),
                'count': len(group),
                'avg_position': group['着順'].mean(),
                'best_position': group['着順'].min()
            }
            
            # ROI計算（オッズがある場合）
            if has_odds:
                win_rows = group[group['着順'] == 1]
                if len(win_rows) > 0:
                    # オッズを数値に変換
                    odds_numeric = pd.to_numeric(win_rows['オッズ'], errors='coerce')
                    odds_numeric = odds_numeric.dropna()
                    if len(odds_numeric) > 0:
                        # ROI = (平均オッズ × 勝率) / 100
                        stats['roi'] = (odds_numeric.mean() * stats['win_rate'])
                    else:
                        stats['roi'] = 1.0
                else:
                    stats['roi'] = 0.8  # 勝利なしの場合は低めのROI
            else:
                stats['roi'] = 1.0
            
            self.jockey_stats[jockey] = stats
        
        print(f"      ✅ {len(self.jockey_stats)}人の騎手統計完了")
        
        # 上位騎手の表示
        top_jockeys = sorted(self.jockey_stats.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)[:5]
        print("      上位騎手（騎乗数順）:")
        for jockey, stats in top_jockeys:
            print(f"        {jockey}: 勝率{stats['win_rate']:.3f}, "
                  f"複勝率{stats['place_rate']:.3f}, "
                  f"騎乗数{stats['count']}")
    
    def _calculate_jockey_context_stats(self, df: pd.DataFrame):
        """コンテキスト別騎手統計の計算"""
        if '騎手' not in df.columns:
            return
        
        print("   📊 コンテキスト別騎手統計計算中...")
        self.jockey_context_stats = {
            'course': {},      # コース別
            'distance': {},    # 距離別
            'surface': {},     # 芝/ダート別
            'condition': {}    # 馬場状態別
        }
        
        # コース別（場idで分類）
        if '場id' in df.columns:
            for (jockey, course), group in df.groupby(['騎手', '場id']):
                key = f"{jockey}_{course}"
                self.jockey_context_stats['course'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        # 距離カテゴリ別
        if '距離' in df.columns:
            df['距離カテゴリ'] = pd.cut(df['距離'], 
                                    bins=[0, 1400, 1800, 2200, 4000], 
                                    labels=['短距離', '中距離', '中長距離', '長距離'])
            
            for (jockey, dist_cat), group in df.groupby(['騎手', '距離カテゴリ']):
                key = f"{jockey}_{dist_cat}"
                self.jockey_context_stats['distance'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        # 芝/ダート別
        if '芝・ダート' in df.columns:
            for (jockey, surface), group in df.groupby(['騎手', '芝・ダート']):
                key = f"{jockey}_{surface}"
                self.jockey_context_stats['surface'][key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
    
    def _calculate_jockey_time_stats(self, df: pd.DataFrame):
        """時系列騎手統計の計算"""
        if '騎手' not in df.columns or '日付' not in df.columns:
            return
        
        print("   📅 時系列騎手統計計算中...")
        
        # 最新日付を基準に
        latest_date = df['日付'].max()
        
        # 30日、60日の成績
        for window_days in [30, 60]:
            cutoff_date = latest_date - pd.Timedelta(days=window_days)
            recent_df = df[df['日付'] >= cutoff_date]
            
            for jockey, group in recent_df.groupby('騎手'):
                key = f"{jockey}_{window_days}d"
                self.jockey_time_stats[key] = {
                    'win_rate': (group['着順'] == 1).mean(),
                    'place_rate': (group['着順'] <= 3).mean(),
                    'count': len(group)
                }
        
        # 連続不勝記録、最後の勝利からの日数
        for jockey, group in df.groupby('騎手'):
            group = group.sort_values('日付', ascending=False)
            
            # 連続不勝
            cold_streak = 0
            for _, row in group.iterrows():
                if row['着順'] == 1:
                    break
                cold_streak += 1
            
            # 最後の勝利からの日数
            win_dates = group[group['着順'] == 1]['日付']
            if len(win_dates) > 0:
                last_win_days = (latest_date - win_dates.iloc[0]).days
            else:
                last_win_days = 365  # デフォルト値
            
            self.jockey_time_stats[f"{jockey}_streak"] = {
                'cold_streak': cold_streak,
                'last_win_days': last_win_days
            }
    
    def _calculate_jockey_synergy_stats(self, df: pd.DataFrame):
        """シナジー統計の計算（騎手×調教師など）"""
        print("   🤝 シナジー統計計算中...")
        
        # 騎手×調教師
        if '騎手' in df.columns and '調教師' in df.columns:
            for (jockey, trainer), group in df.groupby(['騎手', '調教師']):
                if len(group) >= 3:  # 最低3回以上の組み合わせ
                    key = f"{jockey}_{trainer}"
                    self.jockey_synergy_stats[key] = {
                        'win_rate': (group['着順'] == 1).mean(),
                        'place_rate': (group['着順'] <= 3).mean(),
                        'count': len(group)
                    }


class ImprovedMLSystem:
    """改善版MLシステム"""
    
    def __init__(self):
        self.config = MLConfig()
        self.horse_db = HorseDatabase()
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_training_data(self):
        """訓練データ準備"""
        print("📊 訓練データ読み込み中...")
        df = pd.read_csv("encoded/2020_2025encoded_data_v2.csv")
        
        # データクリーニング
        df = df.replace(['?', '---'], np.nan)
        
        # 基本特徴量
        feature_cols = [
            '体重', '体重変化', '斤量', '上がり', '出走頭数', 
            '距離', 'クラス', '性',
            '芝・ダート', '回り', '馬場', '天気', '場id', '枠番',
            # 騎手基本統計
            '騎手の勝率', '騎手の複勝率', '騎手の騎乗数', '騎手の平均着順', '騎手のROI',
            # 騎手時系列統計
            '騎手の勝率_30日', '騎手の複勝率_30日', '騎手の勝率_60日',
            '騎手の連続不勝', '騎手の最後勝利日数',
            # 騎手コンテキスト統計
            '騎手の勝率_芝', '騎手の勝率_ダート',
            '騎手の勝率_短距離', '騎手の勝率_中距離', '騎手の勝率_長距離',
            # シナジー統計
            '騎手調教師相性',
            # 中間日数関連
            '前走からの日数', '放牧区分', '平均中間日数', '中間日数標準偏差',
            '中間日数1', '中間日数2', '中間日数3'
        ]
        
        # 過去成績特徴量
        for i in range(1, 6):
            feature_cols.extend([f'着順{i}', f'距離{i}', f'通過順{i}', f'走破時間{i}'])
        
        # 利用可能な特徴量
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_columns].copy()
        y = (df['着順'] == 1).astype(int)
        
        # 欠損値処理
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # レースID（GroupKFold用）
        groups = None
        if 'race_id' in df.columns:
            groups = df['race_id']
        
        return X, y, groups
    
    def train(self):
        """モデル訓練"""
        print("🚀 モデル訓練開始")
        
        # データ準備
        X, y, groups = self.prepare_training_data()
        
        # データ分割
        from sklearn.model_selection import train_test_split
        if groups is not None:
            # GroupKFoldの最後のfoldを使用
            gkf = GroupKFold(n_splits=5)
            train_idx, test_idx = list(gkf.split(X, y, groups))[-1]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # グループ情報も分割
            groups_train = groups.iloc[train_idx]
            groups_test = groups.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            groups_train = None
            groups_test = None
        
        # スケーリング
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if numeric_cols:
            X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        # CatBoost訓練（改善版）
        print("   🔄 改善版CatBoost訓練中...")
        
        # カテゴリ特徴量の指定
        cat_features = []
        
        # クラス重みを調整（正例1:負例17の比率に近づける）
        avg_horses_per_race = 18  # 平均出走頭数
        scale_pos_weight = avg_horses_per_race - 1  # 17
        
        self.model = CatBoostClassifier(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            random_seed=self.config.random_state,
            cat_features=cat_features if cat_features else None,
            verbose=False,
            scale_pos_weight=scale_pos_weight  # 正例の重みを調整（1:17の比率）
        )
        
        # グループ情報を含めて訓練
        if groups_train is not None:
            train_pool = Pool(
                data=X_train_scaled,
                label=y_train,
                group_id=groups_train
            )
            eval_pool = Pool(
                data=X_test_scaled,
                label=y_test,
                group_id=groups_test
            )
            self.model.fit(train_pool, eval_set=eval_pool, verbose=False)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # 評価
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"   ✅ AUC: {auc:.3f}")
        
        # レース単位での確率合計チェック
        if groups_test is not None:
            prob_sum_check = pd.DataFrame({
                'race_id': groups_test,
                'probability': y_proba
            })
            race_prob_sums = prob_sum_check.groupby('race_id')['probability'].sum()
            print(f"   📊 レース単位の確率合計: 平均{race_prob_sums.mean():.3f}, 標準偏差{race_prob_sums.std():.3f}")
        
        print("✅ 訓練完了")
        
        # 特徴量重要度を取得して騎手関連を確認
        if hasattr(self.model, 'get_feature_importance'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            print("\n📊 騎手関連特徴量の重要度:")
            jockey_features = importance[importance['feature'].str.contains('騎手')]
            for _, row in jockey_features.head(10).iterrows():
                print(f"   {row['feature']:25s}: {row['importance']:.3f}")
        
    def prepare_live_features(self, live_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ライブデータの特徴量準備"""
        print("\n📊 ライブデータ処理中...")
        
        # ライブデータ読み込み
        live_df = pd.read_csv(live_data_path)
        
        # 馬データベース構築（まだ構築されていない場合）
        if not self.horse_db.horse_data:
            self.horse_db.build_database()
        
        # 特徴量作成
        features_list = []
        
        # 現在の日付を取得（レース日付）
        if 'date' in live_df.columns:
            date_str = live_df['date'].iloc[0]
            try:
                # 日本語の日付フォーマットをパース
                current_date = pd.to_datetime(date_str, format='%Y年%m月%d日')
            except:
                try:
                    # その他のフォーマットを試す
                    current_date = pd.to_datetime(date_str)
                except:
                    current_date = pd.Timestamp.now()
        else:
            current_date = pd.Timestamp.now()
        
        for _, row in live_df.iterrows():
            features = {}
            
            # 基本特徴量
            features['体重'] = float(row['馬体重'])
            features['体重変化'] = float(row['馬体重変化'])
            features['斤量'] = float(row['斤量'])
            features['出走頭数'] = len(live_df)
            features['距離'] = float(row['distance'])
            features['クラス'] = 6  # オープン
            features['性'] = 0 if row['性齢'][0] == '牡' else 1
            features['芝・ダート'] = 0  # 芝
            features['枠番'] = int(row['枠'])
            features['場id'] = 5  # 東京
            features['回り'] = 1  # 左
            features['馬場'] = 0  # 良
            features['天気'] = 1  # 曇
            
            # カテゴリカル特徴量は除外（エンコード済みデータとの互換性のため）
            
            # 馬の過去成績（詳細版） - 現在の日付を渡す
            horse_features = self.horse_db.get_horse_features(row['馬名'], current_date)
            if horse_features:
                # 上がり
                features['上がり'] = horse_features.get('平均上がり', 35.0)
                
                # 過去5走の成績
                for i in range(1, 6):
                    features[f'着順{i}'] = horse_features.get(f'着順{i}', 8)
                    features[f'距離{i}'] = horse_features.get(f'距離{i}', 2000)
                    features[f'走破時間{i}'] = horse_features.get(f'走破時間{i}', 120)
                
                # 元データから通過順を取得
                raw_data = horse_features.get('raw_data', pd.DataFrame())
                if not raw_data.empty and '通過順' in raw_data.columns:
                    for i in range(1, 6):
                        if i <= len(raw_data):
                            passing = raw_data.iloc[i-1]['通過順']
                            features[f'通過順{i}'] = self._parse_passing_order(passing)
                        else:
                            features[f'通過順{i}'] = 8
                else:
                    for i in range(1, 6):
                        features[f'通過順{i}'] = 8
                
                # 中間日数関連の特徴量
                features['前走からの日数'] = horse_features.get('前走からの日数', 30)
                features['放牧区分'] = horse_features.get('放牧区分', 1)
                features['平均中間日数'] = horse_features.get('平均中間日数', 30)
                features['中間日数標準偏差'] = horse_features.get('中間日数標準偏差', 0)
                
                # 最近3走の中間日数
                for i in range(1, 4):
                    features[f'中間日数{i}'] = horse_features.get(f'中間日数{i}', 30)
                        
            else:
                # デフォルト値
                features['上がり'] = 35.0
                for i in range(1, 6):
                    features[f'着順{i}'] = 8
                    features[f'距離{i}'] = 2000
                    features[f'通過順{i}'] = 8
                    features[f'走破時間{i}'] = 120
                
                # 中間日数のデフォルト値
                features['前走からの日数'] = 180  # 初出走として扱う
                features['放牧区分'] = 5  # 初出走カテゴリ
                features['平均中間日数'] = 30
                features['中間日数標準偏差'] = 0
                for i in range(1, 4):
                    features[f'中間日数{i}'] = 30
            
            # 騎手統計（拡張版）
            jockey_name = row['騎手']
            jockey_stats = self.horse_db.get_jockey_stats(jockey_name)
            
            # 基本統計
            features['騎手の勝率'] = jockey_stats['win_rate']
            features['騎手の複勝率'] = jockey_stats['place_rate']
            features['騎手の騎乗数'] = np.log1p(jockey_stats['count'])  # 対数変換
            features['騎手の平均着順'] = jockey_stats['avg_position']
            features['騎手のROI'] = jockey_stats['roi']
            
            # 時系列統計
            time30_stats = self.horse_db.get_jockey_time_stats(jockey_name, 30)
            time60_stats = self.horse_db.get_jockey_time_stats(jockey_name, 60)
            streak_stats = self.horse_db.get_jockey_streak_stats(jockey_name)
            
            features['騎手の勝率_30日'] = time30_stats['win_rate']
            features['騎手の複勝率_30日'] = time30_stats['place_rate']
            features['騎手の勝率_60日'] = time60_stats['win_rate']
            features['騎手の連続不勝'] = streak_stats['cold_streak']
            features['騎手の最後勝利日数'] = np.exp(-streak_stats['last_win_days'] / 30)  # 指数減衰
            
            # コンテキスト統計
            # 芝/ダート
            features['騎手の勝率_芝'] = self.horse_db.get_jockey_context_stats(
                jockey_name, 'surface', '芝'
            )['win_rate']
            features['騎手の勝率_ダート'] = self.horse_db.get_jockey_context_stats(
                jockey_name, 'surface', 'ダ'
            )['win_rate']
            
            # 距離カテゴリ（現在のレース距離に基づく）
            current_distance = features['距離']
            if current_distance <= 1400:
                dist_cat = '短距離'
            elif current_distance <= 1800:
                dist_cat = '中距離'
            elif current_distance <= 2200:
                dist_cat = '中長距離'
            else:
                dist_cat = '長距離'
            
            features['騎手の勝率_短距離'] = self.horse_db.get_jockey_context_stats(
                jockey_name, 'distance', '短距離'
            )['win_rate']
            features['騎手の勝率_中距離'] = self.horse_db.get_jockey_context_stats(
                jockey_name, 'distance', '中距離'
            )['win_rate']
            features['騎手の勝率_長距離'] = self.horse_db.get_jockey_context_stats(
                jockey_name, 'distance', '長距離'
            )['win_rate']
            
            # シナジー統計
            trainer_name = row['調教師']
            synergy_stats = self.horse_db.get_jockey_synergy_stats(jockey_name, trainer_name)
            features['騎手調教師相性'] = synergy_stats['win_rate']
            
            features_list.append(features)
        
        # DataFrame作成
        features_df = pd.DataFrame(features_list)
        
        # 訓練時と同じ列に揃える
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.feature_columns]
        
        return features_df, live_df
    
    def _parse_passing_order(self, passing_order) -> float:
        """通過順をパースして平均値を返す"""
        if pd.isna(passing_order) or passing_order == '':
            return 8.0
        
        try:
            if isinstance(passing_order, (int, float)):
                return float(passing_order)
            
            # "5-6-7" のような形式を平均値に変換
            if '-' in str(passing_order):
                positions = [float(x) for x in str(passing_order).split('-') if x.strip()]
                return np.mean(positions) if positions else 8.0
            
            return float(passing_order)
        except:
            return 8.0
    
    def predict_race(self, live_data_path: str):
        """レース予測"""
        # 特徴量準備
        features_df, live_df = self.prepare_live_features(live_data_path)
        
        # スケーリング
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        features_scaled = features_df.copy()
        if numeric_cols:
            features_scaled[numeric_cols] = self.scaler.transform(features_df[numeric_cols])
        
        # 予測
        raw_probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Softmax正規化（レース内で確率の合計が1になるように）
        # exp(logit)を計算してからソフトマックス
        exp_probs = np.exp(np.log(raw_probabilities + 1e-10))  # 0除算を避ける
        probabilities = exp_probs / exp_probs.sum()
        
        # 結果作成
        results = live_df.copy()
        results['勝率'] = probabilities
        results['期待値'] = results['勝率'] * results['単勝オッズ'].astype(float)
        
        # 結果表示
        print("\n🎯 予測結果:")
        print("=" * 80)
        print(f"{'順位':>3} {'馬番':>3} {'馬名':>14} {'オッズ':>7} {'勝率':>7} {'期待値':>7}")
        print("=" * 80)
        
        sorted_results = results.sort_values('勝率', ascending=False)
        for i, (_, row) in enumerate(sorted_results.head(10).iterrows(), 1):
            print(f"{i:3d}. {row['馬番']:2d}番 {row['馬名']:14s} "
                  f"{row['単勝オッズ']:6.1f}倍 {row['勝率']*100:5.1f}% "
                  f"{row['期待値']:6.2f}")
        
        # 統計表示
        print(f"\n📊 予測統計:")
        print(f"   最小勝率: {probabilities.min()*100:.1f}%")
        print(f"   最大勝率: {probabilities.max()*100:.1f}%")
        print(f"   平均勝率: {probabilities.mean()*100:.1f}%")
        print(f"   勝率合計: {probabilities.sum()*100:.1f}%")
        
        # 投資推奨
        profitable = results[results['期待値'] >= 1.0]
        if len(profitable) > 0:
            print(f"\n💰 期待値1.0以上: {len(profitable)}頭")
            for _, horse in profitable.iterrows():
                print(f"   {horse['馬番']:2d}番 {horse['馬名']} 期待値{horse['期待値']:.2f}")


def main():
    """メイン実行"""
    system = ImprovedMLSystem()
    system.train()
    system.predict_race("live_race_data_202505021211.csv")


if __name__ == "__main__":
    main()