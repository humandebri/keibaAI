#!/usr/bin/env python3
"""
競馬AI改良システム - オッズ依存を減らし、ROIを重視した実運用可能モデル（修正版）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
import warnings
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from datetime import datetime
import logging
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')


class ImprovedKeibaAISystem:
    """改良版競馬AI予測システム（修正版）"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.segment_models = {}  # セグメント別モデル
        self.feature_cols = []
        self.non_odds_feature_cols = []  # オッズ系を除いた特徴量
        self.data = None
        self.feature_importance = {}
        self.calibrators = {}  # 確率校正用
        self.logger = self._setup_logger()
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'data_dir': 'data',
            'output_dir': 'results_improved',
            'model_params': {
                'lightgbm': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 31,  # 浅くして汎化性能向上
                    'max_depth': 4,    # 深さ制限
                    'learning_rate': 0.03,
                    'feature_fraction': 0.6,  # より少ない特徴量で汎化
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 30,  # より大きくして過学習防止
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'verbose': -1,
                    'seed': 42
                },
                'xgboost': {
                    'n_estimators': 300,
                    'max_depth': 4,      # 浅くする
                    'learning_rate': 0.04,  # eta調整
                    'subsample': 0.7,
                    'colsample_bytree': 0.6,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbosity': 0,
                    'early_stopping_rounds': 50
                }
            },
            'segments': {
                'track_type': ['芝', 'ダート'],
                'distance_category': ['sprint', 'mile', 'intermediate', 'long']
            },
            'betting': {
                'jra_takeout_rate': 0.25,  # JRA控除率
                'min_roi_threshold': 1.05,  # 最低ROI閾値
                'kelly_fraction': 0.05,     # Kelly基準の使用率（より保守的に）
                'max_bet_fraction': 0.01    # 最大ベット比率（1%に制限）
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('ImprovedKeibaAI')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            Path('logs').mkdir(exist_ok=True)
            fh = logging.FileHandler(f'logs/improved_keiba_ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """改良版の特徴量エンジニアリング（オッズ依存を減らす）"""
        result = df.copy()
        
        # 1. 馬場指数の計算（仮想的な含水率とクッション値）
        if '馬場' in df.columns:
            track_map = {'良': 1.0, '稍': 0.8, '稍重': 0.8, '重': 0.6, '不': 0.4, '不良': 0.4}
            result['track_moisture_index'] = result['馬場'].map(track_map).fillna(0.7)
            result['track_cushion_value'] = result['track_moisture_index'] * 0.5 + 0.5
        
        # 2. ラップタイムの区間別指数（前半3F・後半3F比）
        if '上がり' in df.columns:
            # 上がりタイムから推定される前後半バランス
            result['last_3f_index'] = 36.0 / (result['上がり'] + 0.1)  # 36秒を基準
            result['pace_balance'] = result['last_3f_index'] * 0.5  # ペースバランス推定
        
        # 3. 厩舎＆調教師 win% (最近3ヶ月) - 仮想データ
        if '調教師' in df.columns:
            # 実際はDBから取得すべきだが、ここでは仮想的に計算
            trainer_win_rate = result.groupby('調教師')['着順'].apply(
                lambda x: (x == 1).sum() / len(x)
            ).to_dict()
            result['trainer_win_rate_3m'] = result['調教師'].map(trainer_win_rate).fillna(0.1)
            result['trainer_place_rate_3m'] = result['調教師'].map(
                lambda t: trainer_win_rate.get(t, 0.1) * 2.5
            ).fillna(0.25)
        
        # 4. 枠順×馬場状態 交互作用
        if '馬番' in df.columns and 'track_moisture_index' in result.columns:
            result['draw_track_interaction'] = (
                result['馬番'] * result['track_moisture_index']
            )
            # 内枠有利/不利の指標
            result['inside_draw_advantage'] = np.where(
                (result['馬番'] <= 4) & (result['track_moisture_index'] < 0.8),
                1.2,  # 重馬場で内枠有利
                1.0
            )
        
        # 5. 馬体重変化の影響度
        if '体重変化' in df.columns:
            # 体重を数値化
            if '体重' in df.columns:
                weight_values = []
                for w in result['体重']:
                    try:
                        weight = int(str(w).split('(')[0]) if pd.notna(w) else 480
                    except:
                        weight = 480
                    weight_values.append(weight)
                result['体重_numeric'] = weight_values
            else:
                result['体重_numeric'] = 480
            
            result['weight_change_impact'] = (
                result['体重変化'].abs() / result['体重_numeric']
            )
            result['weight_stability'] = 1 / (1 + result['weight_change_impact'])
        
        # 6. 年齢×クラス×性別の複合指標
        if all(col in df.columns for col in ['齢', '性', '出走頭数']):
            result['age_competitiveness'] = result['齢'].map({
                2: 0.7, 3: 0.9, 4: 1.0, 5: 0.95, 6: 0.85, 7: 0.75
            }).fillna(0.7)
            
            result['gender_factor'] = result['性'].map({
                '牡': 1.0, '牝': 0.9, '騸': 0.95
            }).fillna(0.9)
            
            result['competitive_index'] = (
                result['age_competitiveness'] * 
                result['gender_factor'] * 
                (18 / result['出走頭数'])  # フィールドサイズ補正
            )
        
        # 7. 距離適性の詳細化
        if '距離' in df.columns:
            result['distance_category'] = pd.cut(
                result['距離'],
                bins=[0, 1400, 1800, 2200, 3600],
                labels=['sprint', 'mile', 'intermediate', 'long']
            )
            
            # 距離変化への適応性
            result['is_distance_specialist'] = 0  # 実際は過去データから計算すべき
        
        # 8. 血統的な距離適性（仮想）
        result['bloodline_distance_affinity'] = np.random.uniform(0.8, 1.2, len(result))
        
        # 9. 調子の波（仮想的に前走からの変化）
        result['form_cycle'] = np.sin(result.index * 0.1) * 0.1 + 1.0
        
        # 10. レース内の相対指標（オッズを使わない）
        if '斤量' in df.columns:
            result['weight_carried_relative'] = result.groupby('race_id')['斤量'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5)
            )
        
        if '体重_numeric' in result.columns:
            result['horse_weight_relative'] = result.groupby('race_id')['体重_numeric'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5)
            )
        
        # 11. 天候の影響
        if '天気' in df.columns:
            weather_map = {'晴': 1.0, '曇': 0.9, '雨': 0.7, '小雨': 0.8, '雪': 0.5}
            result['weather_factor'] = result['天気'].map(weather_map).fillna(0.85)
        
        # 12. 競馬場特性
        if '場名' in df.columns:
            # 各競馬場の特性（直線の長さ、高低差など）
            track_characteristics = {
                '東京': {'straight': 525, 'elevation': 2.7},
                '中山': {'straight': 310, 'elevation': 5.3},
                '阪神': {'straight': 474, 'elevation': 2.2},
                '京都': {'straight': 404, 'elevation': 4.3},
                '新潟': {'straight': 359, 'elevation': 0.8},
                '中京': {'straight': 412, 'elevation': 3.5},
                '札幌': {'straight': 266, 'elevation': 1.0},
                '函館': {'straight': 262, 'elevation': 1.5},
                '福島': {'straight': 310, 'elevation': 2.5},
                '小倉': {'straight': 326, 'elevation': 1.8}
            }
            
            result['track_straight_length'] = result['場名'].map(
                lambda x: track_characteristics.get(x, {}).get('straight', 350)
            )
            result['track_elevation'] = result['場名'].map(
                lambda x: track_characteristics.get(x, {}).get('elevation', 2.0)
            )
        
        return result
    
    def split_features_by_type(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """特徴量をオッズ系とそれ以外に分割"""
        odds_related = ['オッズ', 'オッズ_numeric', '人気', 'odds_rank', 
                       'popularity_odds_ratio', 'popularity_relative', 
                       'popularity_mean', 'popularity_std']
        
        all_features = []
        non_odds_features = []
        
        exclude_cols = ['race_id', '着順', '着順_numeric', 'year', '馬', 
                       '騎手', '調教師', 'レース名', '場名', '走破時間', 
                       '通過順', '日付', '開催', 'クラス', '芝・ダート', 
                       '回り', '馬場', '天気', 'distance_category']
        
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if df[col].notna().sum() / len(df) > 0.5:
                    all_features.append(col)
                    if col not in odds_related:
                        non_odds_features.append(col)
        
        return all_features, non_odds_features
    
    def train_segment_models(self, train_years: List[int], val_years: List[int]) -> None:
        """セグメント別モデルの訓練"""
        self.logger.info("セグメント別モデルを訓練中...")
        
        train_data = self.data[self.data['year'].isin(train_years)]
        val_data = self.data[self.data['year'].isin(val_years)]
        
        # セグメントの組み合わせ
        segments = []
        if '芝・ダート' in train_data.columns:
            for track in ['芝', 'ダート']:
                for dist_cat in ['sprint', 'mile', 'intermediate', 'long']:
                    segments.append((track, dist_cat))
        
        for track_type, distance_cat in segments:
            segment_key = f"{track_type}_{distance_cat}"
            self.logger.info(f"  セグメント: {segment_key}")
            
            # セグメントデータの抽出
            train_segment = train_data[
                (train_data['芝・ダート'].str.contains(track_type)) &
                (train_data['distance_category'] == distance_cat)
            ]
            val_segment = val_data[
                (val_data['芝・ダート'].str.contains(track_type)) &
                (val_data['distance_category'] == distance_cat)
            ]
            
            if len(train_segment) < 100 or len(val_segment) < 50:
                self.logger.warning(f"    データ不足でスキップ")
                continue
            
            # 特徴量準備（オッズ系を除く）
            X_train = train_segment[self.non_odds_feature_cols].fillna(0).values
            y_train = train_segment['着順_numeric'].values
            
            X_val = val_segment[self.non_odds_feature_cols].fillna(0).values
            y_val = val_segment['着順_numeric'].values
            
            # LightGBMでセグメントモデル訓練
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            params = self.config['model_params']['lightgbm'].copy()
            params['num_leaves'] = 15  # セグメントモデルはさらに浅く
            
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            
            self.segment_models[segment_key] = model
    
    def calculate_roi_simulation(self, test_data: pd.DataFrame, 
                                model: lgb.Booster, 
                                feature_cols: List[str]) -> Dict:
        """ROIシミュレーション"""
        self.logger.info("ROIシミュレーションを実行中...")
        
        # 予測スコアを計算
        all_predictions = []
        all_results = []
        
        unique_races = test_data['race_id'].unique()
        
        for race_id in unique_races[:1000]:  # 最初の1000レース
            race_data = test_data[test_data['race_id'] == race_id]
            
            if len(race_data) < 5:
                continue
            
            X = race_data[feature_cols].fillna(0).values
            predictions = model.predict(X, num_iteration=model.best_iteration)
            
            # 予測順位を計算（低い値ほど上位）
            pred_ranks = rankdata(predictions)
            
            for i, (_, horse) in enumerate(race_data.iterrows()):
                all_predictions.append({
                    'race_id': race_id,
                    'horse_num': horse['馬番'],
                    'pred_score': predictions[i],
                    'pred_rank': pred_ranks[i],
                    'actual_rank': horse['着順_numeric'],
                    'odds': horse.get('オッズ_numeric', 10),
                    'popularity': horse.get('人気', 10)
                })
        
        # 閾値別のROI計算
        roi_results = {}
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]  # 上位X%のみ賭ける
        
        for threshold in thresholds:
            pred_df = pd.DataFrame(all_predictions)
            
            # レースごとに上位X%を選択
            selected_bets = []
            for race_id in pred_df['race_id'].unique():
                race_pred = pred_df[pred_df['race_id'] == race_id]
                n_select = max(1, int(len(race_pred) * threshold))
                top_horses = race_pred.nsmallest(n_select, 'pred_rank')
                selected_bets.extend(top_horses.to_dict('records'))
            
            # ROI計算
            total_bet = len(selected_bets) * 100  # 各100円賭け
            total_return = 0
            wins = 0
            
            for bet in selected_bets:
                if bet['actual_rank'] == 1:  # 単勝的中
                    wins += 1
                    # 簡易的な払戻計算（実際のオッズ × 100円）
                    # ただし、極端なオッズは制限
                    payout_odds = min(bet['odds'], 100)  # 最大100倍に制限
                    payout = payout_odds * 100
                    total_return += payout
            
            roi = total_return / total_bet if total_bet > 0 else 0
            win_rate = wins / len(selected_bets) if selected_bets else 0
            
            roi_results[f'top_{int(threshold*100)}pct'] = {
                'threshold': threshold,
                'total_bets': len(selected_bets),
                'wins': wins,
                'win_rate': win_rate,
                'total_bet_amount': total_bet,
                'total_return': total_return,
                'roi': roi,
                'profit': total_return - total_bet
            }
        
        return roi_results
    
    def train_models_with_validation(self, train_years: List[int], val_years: List[int]) -> None:
        """改良版モデル訓練（年度別Hold-Out検証付き）"""
        self.logger.info("改良版モデルを訓練中...")
        
        # データ分割
        train_data = self.data[self.data['year'].isin(train_years)]
        val_data = self.data[self.data['year'].isin(val_years)]
        
        # 特徴量の分離
        self.feature_cols, self.non_odds_feature_cols = self.split_features_by_type(train_data)
        
        self.logger.info(f"全特徴量数: {len(self.feature_cols)}")
        self.logger.info(f"非オッズ特徴量数: {len(self.non_odds_feature_cols)}")
        
        # 1. オッズ依存モデル（比較用）
        self._train_model_variant('with_odds', train_data, val_data, self.feature_cols)
        
        # 2. 純粋な実力モデル（オッズ系除外）
        self._train_model_variant('pure_ability', train_data, val_data, self.non_odds_feature_cols)
        
        # 3. セグメント別モデル
        self.train_segment_models(train_years, val_years)
        
        # 4. アンサンブルモデル（Rank Averaging）
        self._create_ensemble_model(val_data)
    
    def _train_model_variant(self, name: str, train_data: pd.DataFrame, 
                           val_data: pd.DataFrame, feature_cols: List[str]) -> None:
        """モデルバリアントの訓練"""
        self.logger.info(f"  {name}モデルを訓練中...")
        
        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data['着順_numeric'].values
        
        X_val = val_data[feature_cols].fillna(0).values
        y_val = val_data['着順_numeric'].values
        
        # LightGBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        lgb_model = lgb.train(
            self.config['model_params']['lightgbm'],
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**self.config['model_params']['xgboost'])
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # モデル保存
        self.models[f'{name}_lgb'] = lgb_model
        self.models[f'{name}_xgb'] = xgb_model
        
        # 評価
        lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        xgb_pred = xgb_model.predict(X_val)
        
        # 順位相関
        lgb_corr = self._calculate_rank_correlation(y_val, lgb_pred)
        xgb_corr = self._calculate_rank_correlation(y_val, xgb_pred)
        
        self.logger.info(f"    LightGBM順位相関: {lgb_corr:.3f}")
        self.logger.info(f"    XGBoost順位相関: {xgb_corr:.3f}")
        
        # ROIシミュレーション
        if name == 'pure_ability':
            roi_results = self.calculate_roi_simulation(val_data, lgb_model, feature_cols)
            self.logger.info("    ROIシミュレーション結果:")
            for key, result in roi_results.items():
                self.logger.info(f"      {key}: ROI={result['roi']:.3f}, "
                               f"勝率={result['win_rate']:.3f}")
    
    def _calculate_rank_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """順位相関の計算"""
        pred_ranks = rankdata(y_pred)
        true_ranks = rankdata(y_true)
        return np.corrcoef(pred_ranks, true_ranks)[0, 1]
    
    def _create_ensemble_model(self, val_data: pd.DataFrame) -> None:
        """アンサンブルモデルの作成（Rank Averaging）"""
        self.logger.info("アンサンブルモデルを作成中...")
        
        # 各モデルの予測を収集
        ensemble_predictions = {}
        
        for name, model in self.models.items():
            if 'lgb' in name:
                features = self.feature_cols if 'with_odds' in name else self.non_odds_feature_cols
                X_val = val_data[features].fillna(0).values
                pred = model.predict(X_val, num_iteration=model.best_iteration)
            elif 'xgb' in name:
                features = self.feature_cols if 'with_odds' in name else self.non_odds_feature_cols
                X_val = val_data[features].fillna(0).values
                pred = model.predict(X_val)
            else:
                continue
            
            # ランクに変換
            pred_ranks = rankdata(pred)
            ensemble_predictions[name] = pred_ranks
        
        # Borda count（順位の平均）
        if ensemble_predictions:
            ensemble_rank = np.mean(list(ensemble_predictions.values()), axis=0)
            
            # 評価
            y_val = val_data['着順_numeric'].values
            ensemble_corr = self._calculate_rank_correlation(y_val, -ensemble_rank)  # 低い順位が良い
            
            self.logger.info(f"  アンサンブル順位相関: {ensemble_corr:.3f}")
    
    def run_final_backtest(self, test_years: List[int], initial_capital: float = 1_000_000) -> Dict:
        """最終的なバックテスト（ROI重視、修正版）"""
        self.logger.info(f"最終バックテスト実行中 ({test_years}年)...")
        
        test_data = self.data[self.data['year'].isin(test_years)]
        
        # 純粋実力モデルを使用
        model = self.models.get('pure_ability_lgb')
        if not model:
            raise ValueError("純粋実力モデルが訓練されていません")
        
        capital = initial_capital
        all_trades = []
        monthly_results = {}
        
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races[:2000]):
            if i % 200 == 0:
                self.logger.debug(f"  処理中: {i}/{min(len(unique_races), 2000)} レース")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            if len(race_data) < 8:
                continue
            
            # 月を記録
            month = pd.to_datetime(race_data.iloc[0]['日付']).strftime('%Y-%m')
            if month not in monthly_results:
                monthly_results[month] = {'bets': 0, 'wins': 0, 'profit': 0}
            
            # 予測
            X = race_data[self.non_odds_feature_cols].fillna(0).values
            predictions = model.predict(X, num_iteration=model.best_iteration)
            
            # 予測順位
            pred_ranks = rankdata(predictions)
            
            # 上位20%のみ選択
            threshold_rank = int(len(race_data) * 0.2)
            
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                if pred_ranks[idx] <= threshold_rank:
                    # 期待値計算（簡易版）
                    # 予測順位に基づく勝率推定（より保守的に）
                    if pred_ranks[idx] == 1:
                        win_prob = 0.20  # 予測1位でも20%程度
                    elif pred_ranks[idx] == 2:
                        win_prob = 0.12
                    elif pred_ranks[idx] <= 3:
                        win_prob = 0.08
                    elif pred_ranks[idx] <= 5:
                        win_prob = 0.04
                    else:
                        win_prob = 0.02
                    
                    odds = horse.get('オッズ_numeric', 10)
                    # オッズの上限を設定（現実的な範囲）
                    odds = min(odds, 50)
                    
                    expected_value = win_prob * odds
                    
                    # ROI閾値チェック
                    if expected_value < self.config['betting']['min_roi_threshold']:
                        continue
                    
                    # Kelly基準でベット額計算
                    kelly_full = (win_prob * (odds - 1) - (1 - win_prob)) / (odds - 1)
                    kelly = max(0, kelly_full * self.config['betting']['kelly_fraction'])
                    
                    bet_fraction = min(kelly, self.config['betting']['max_bet_fraction'])
                    bet_amount = int(capital * bet_fraction / 100) * 100
                    
                    # 最小・最大ベット額制限
                    bet_amount = max(100, min(bet_amount, 10000))
                    
                    if bet_amount < 100 or bet_amount > capital * 0.05:  # 資金の5%まで
                        continue
                    
                    # 結果判定
                    is_win = (horse['着順_numeric'] == 1)
                    
                    if is_win:
                        # 実際の払戻額（100円あたり）
                        profit = bet_amount * odds - bet_amount
                        monthly_results[month]['wins'] += 1
                    else:
                        profit = -bet_amount
                    
                    capital += profit
                    monthly_results[month]['bets'] += 1
                    monthly_results[month]['profit'] += profit
                    
                    all_trades.append({
                        'race_id': race_id,
                        'horse_num': horse['馬番'],
                        'bet_amount': bet_amount,
                        'odds': odds,
                        'profit': profit,
                        'capital': capital,
                        'is_win': is_win,
                        'expected_value': expected_value
                    })
                    
                    if capital <= 10000:
                        self.logger.warning("資金不足で終了")
                        break
            
            if capital <= 10000:
                break
        
        # 結果集計
        total_return = (capital - initial_capital) / initial_capital
        winning_trades = [t for t in all_trades if t['is_win']]
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(all_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(all_trades) if all_trades else 0,
            'avg_expected_value': np.mean([t['expected_value'] for t in all_trades]) if all_trades else 0,
            'monthly_results': monthly_results,
            'roi': capital / initial_capital if all_trades else 1.0,
            'all_trades': all_trades[:100]  # 最初の100件のみ保存
        }
        
        return results
    
    def load_data(self, start_year: int = 2020, end_year: int = 2025) -> bool:
        """データの読み込みと前処理"""
        self.logger.info(f"{start_year}年から{end_year}年のデータを読み込み中...")
        
        all_data = []
        data_dir = Path(self.config['data_dir'])
        
        for year in range(start_year, end_year + 1):
            try:
                file_path = data_dir / f'{year}.xlsx'
                df = pd.read_excel(file_path)
                df['year'] = year
                
                # 着順を数値に変換
                df['着順_numeric'] = pd.to_numeric(df['着順'], errors='coerce')
                df = df.dropna(subset=['着順_numeric'])
                
                # オッズの数値化
                if 'オッズ' in df.columns:
                    df['オッズ_numeric'] = pd.to_numeric(df['オッズ'], errors='coerce').fillna(99.9)
                
                # 高度な特徴量を追加
                df = self.create_advanced_features(df)
                
                all_data.append(df)
                self.logger.info(f"  {year}年: {len(df)}行のデータを処理")
                
            except Exception as e:
                self.logger.warning(f"  {year}年: エラー ({e})")
                continue
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"合計 {len(self.data)} 行のデータ")
            return True
        else:
            self.logger.error("データの読み込みに失敗しました")
            return False
    
    def display_results(self, results: Dict) -> None:
        """結果の表示"""
        print("\n" + "="*60)
        print("改良版バックテスト結果")
        print("="*60)
        print(f"初期資金: ¥{results['initial_capital']:,.0f}")
        print(f"最終資金: ¥{results['final_capital']:,.0f}")
        print(f"総収益率: {results['total_return']*100:.1f}%")
        print(f"ROI: {results['roi']:.3f}")
        print(f"総取引数: {results['total_trades']}")
        print(f"勝利数: {results['winning_trades']}")
        print(f"勝率: {results['win_rate']*100:.1f}%")
        print(f"平均期待値: {results['avg_expected_value']:.3f}")
        
        # 月別パフォーマンス
        print("\n月別パフォーマンス（上位5ヶ月）:")
        print("-" * 50)
        monthly_sorted = sorted(
            results['monthly_results'].items(),
            key=lambda x: x[1]['profit'],
            reverse=True
        )[:5]
        
        for month, stats in monthly_sorted:
            if stats['bets'] > 0:
                win_rate = stats['wins'] / stats['bets'] * 100
                print(f"{month}: 収益 {stats['profit']:+8,.0f}円, "
                      f"勝率 {win_rate:4.1f}% ({stats['wins']}/{stats['bets']})")
        
        # 取引例
        print("\n取引例（最初の10件）:")
        print("-" * 50)
        for i, trade in enumerate(results.get('all_trades', [])[:10], 1):
            result = "的中" if trade['is_win'] else "外れ"
            print(f"{i}. レース{trade['race_id']}: "
                  f"馬番{trade['horse_num']}, "
                  f"賭け金¥{trade['bet_amount']:,}, "
                  f"オッズ{trade['odds']:.1f}, "
                  f"{result}, 損益{trade['profit']:+,.0f}円")


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("競馬AI改良システム - ROI重視の実運用可能モデル（修正版）")
    print("=" * 60)
    
    # システム初期化
    system = ImprovedKeibaAISystem()
    
    # データ読み込み
    if not system.load_data(start_year=2020, end_year=2025):
        return False
    
    # モデル訓練（改良版）
    system.train_models_with_validation(
        train_years=[2020, 2021, 2022],
        val_years=[2023]
    )
    
    # 最終バックテスト
    results = system.run_final_backtest(
        test_years=[2024, 2025],
        initial_capital=1_000_000
    )
    
    # 結果表示
    system.display_results(results)
    
    # 結果保存
    output_dir = Path(system.config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 取引履歴を除いて保存（ファイルサイズ削減）
    save_results = results.copy()
    save_results.pop('all_trades', None)
    
    with open(output_dir / 'improved_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    if results['roi'] > 1.0:
        print("\n✅ ROI > 1.0 を達成！実運用の可能性があります。")
        print("ただし、実際の運用では以下に注意してください：")
        print("- JRA控除率（約25%）を考慮")
        print("- スリッページや約定の問題")
        print("- 過去データでの検証と実運用の差")
        return True
    else:
        print("\n📊 ROI < 1.0 です。さらなる改善が必要です。")
        print("改善案：")
        print("- セグメント特化（芝・短距離など）")
        print("- より高度な特徴量エンジニアリング")
        print("- 賭け方の最適化（ワイド・馬連など）")
        return False


if __name__ == "__main__":
    success = main()