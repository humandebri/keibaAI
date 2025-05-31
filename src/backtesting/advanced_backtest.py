#!/usr/bin/env python3
"""
高度なバックテストシステム
1. 高配当狙い戦略
2. 選択的ベッティング
3. アンサンブルモデル
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import os
from pathlib import Path

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class AdvancedBacktest:
    def __init__(self, betting_fraction=0.02, min_odds=5.0, confidence_threshold=0.65):
        self.betting_fraction = betting_fraction
        self.min_odds = min_odds  # 最低オッズ（人気薄狙い）
        self.confidence_threshold = confidence_threshold  # 賭けるための最低確信度
        self.initial_capital = 1000000
        
    def load_data(self):
        """データの読み込み"""
        print("Loading data...")
        dfs = []
        for year in range(2019, 2024):  # テスト用にデータを減らす
            try:
                df = pd.read_excel(f'data/{year}.xlsx')
                df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                df = df.dropna(subset=['着順'])
                df['year'] = year
                print(f"Loaded {year}.xlsx: {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {year}.xlsx - {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total rows: {len(self.data)}")
        
        # 特徴量の準備
        self.prepare_features()
        
    def prepare_features(self):
        """高度な特徴量エンジニアリング"""
        print("Preparing advanced features...")
        
        # 基本的なエンコーディング
        categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        # 数値変数の処理
        numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # 追加の特徴量作成
        # 1. オッズランク（高配当を識別）
        self.data['odds_rank'] = pd.qcut(self.data['オッズ'], q=10, labels=False)
        
        # 2. 体重変化率
        self.data['weight_change_rate'] = self.data['体重変化'] / self.data['体重']
        
        # 3. 人気と斤量の交互作用
        self.data['popularity_weight_interaction'] = self.data['人気'] * self.data['斤量']
        
        # 4. 馬番の有利不利（内枠・外枠）
        self.data['is_inner'] = (self.data['馬番'] <= 4).astype(int)
        self.data['is_outer'] = (self.data['馬番'] >= 13).astype(int)
        
        # ターゲット変数
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
        
        # 高配当馬の複勝率を確認
        high_odds_place_rate = self.data[self.data['オッズ'] >= self.min_odds]['is_place'].mean()
        print(f"High odds (>={self.min_odds}) place rate: {high_odds_place_rate:.1%}")
        
    def get_features(self, data):
        """特徴量の取得"""
        feature_columns = [
            '馬番', '斤量', 'オッズ', '人気', '体重', '体重変化',
            '性', '馬場', '天気', '芝・ダート', '場名',
            'odds_rank', 'weight_change_rate', 'popularity_weight_interaction',
            'is_inner', 'is_outer'
        ]
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                features.append(data[col].values)
        
        if len(features) == 0:
            return None
            
        return np.column_stack(features)
    
    def train_ensemble_models(self, train_data):
        """アンサンブルモデルの訓練"""
        print("Training ensemble models...")
        
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
            
        target = train_data['is_place']
        
        # 1. LightGBMモデル
        lgb_train_data = lgb.Dataset(features, target)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 2. ランダムフォレストモデル
        rf_model = RandomForestClassifier(
            n_estimators=50,  # 高速化のため減らす
            max_depth=10,
            min_samples_split=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(features, target)
        
        # 3. ロジスティック回帰（正則化あり）
        lr_model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(features, target)
        
        return {
            'lgb': lgb_model,
            'rf': rf_model,
            'lr': lr_model
        }
    
    def predict_ensemble(self, models, features):
        """アンサンブル予測"""
        # 各モデルの予測を取得
        lgb_pred = models['lgb'].predict(features)
        rf_pred = models['rf'].predict_proba(features)[:, 1]
        lr_pred = models['lr'].predict_proba(features)[:, 1]
        
        # 重み付き平均（LightGBMを重視）
        ensemble_pred = 0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * lr_pred
        
        # 予測の信頼度（各モデルの予測の一致度）
        predictions = np.column_stack([lgb_pred, rf_pred, lr_pred])
        confidence = 1 - np.std(predictions, axis=1)
        
        return ensemble_pred, confidence
    
    def calculate_expected_return(self, win_odds, place_prob):
        """期待リターンの計算"""
        # 複勝配当の推定（オッズが高いほど配当も高い）
        if win_odds <= 5:
            place_return = 1.5  # 150円
        elif win_odds <= 10:
            place_return = 2.0  # 200円
        elif win_odds <= 20:
            place_return = 2.5  # 250円
        elif win_odds <= 50:
            place_return = 3.5  # 350円
        else:
            place_return = 5.0  # 500円
        
        # 期待値 = (的中確率 × 配当) - 1
        expected_value = (place_prob * place_return) - 1
        
        return expected_value, place_return
    
    def run_backtest(self):
        """高度なバックテスト実行"""
        # データ分割
        train_years = range(2019, 2021)
        test_years = range(2021, 2024)
        
        train_data = self.data[self.data['year'].isin(train_years)]
        test_data = self.data[self.data['year'].isin(test_years)]
        
        print(f"\nTrain data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # アンサンブルモデルの訓練
        models = self.train_ensemble_models(train_data)
        
        # バックテスト実行
        print("\nRunning advanced backtest...")
        capital = self.initial_capital
        total_bets = 0
        total_wins = 0
        high_value_bets = 0
        monthly_results = []
        
        # レースごとに処理
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if i % 1000 == 0:
                print(f"Processing race {i}/{len(unique_races)}...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            # 特徴量取得
            features = self.get_features(race_data)
            if features is None:
                continue
            
            # アンサンブル予測
            predictions, confidence = self.predict_ensemble(models, features)
            
            # ベッティング候補の選択
            betting_candidates = []
            
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                win_odds = horse['オッズ']
                
                # 高配当馬のみを対象
                if win_odds < self.min_odds:
                    continue
                
                place_prob = predictions[idx]
                conf = confidence[idx]
                
                # 期待値計算
                ev, place_return = self.calculate_expected_return(win_odds, place_prob)
                
                # 高い確信度と正の期待値を持つ馬を選択
                if conf >= self.confidence_threshold and ev > 0.2:  # 期待値20%以上
                    betting_candidates.append({
                        'idx': idx,
                        'horse': horse,
                        'prob': place_prob,
                        'confidence': conf,
                        'expected_value': ev,
                        'place_return': place_return
                    })
            
            # 最も期待値の高い馬に賭ける
            if betting_candidates:
                # 期待値でソート
                betting_candidates.sort(key=lambda x: x['expected_value'], reverse=True)
                best_bet = betting_candidates[0]
                
                # Kelly基準でベット額を決定
                kelly_fraction = best_bet['expected_value'] / (best_bet['place_return'] - 1)
                kelly_fraction = min(kelly_fraction, 0.25)  # 最大25%
                bet_amount = capital * self.betting_fraction * kelly_fraction
                
                total_bets += 1
                high_value_bets += 1
                
                horse = best_bet['horse']
                if horse['着順'] <= 3:
                    # 的中
                    payout = bet_amount * best_bet['place_return']
                    profit = payout - bet_amount
                    total_wins += 1
                else:
                    # 外れ
                    profit = -bet_amount
                
                capital += profit
                
                # 破産チェック
                if capital <= 0:
                    print(f"Bankrupt at race {i}")
                    break
        
        # 結果を返す
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0,
            'high_value_bets': high_value_bets,
            'avg_bets_per_race': total_bets / len(unique_races)
        }

def main():
    """メイン実行関数"""
    print("=== 高度なバックテストシステム ===")
    print("戦略:")
    print("1. 高配当馬（5倍以上）に焦点")
    print("2. アンサンブルモデルによる予測")
    print("3. 高期待値レースのみ選択的ベット")
    
    # バックテストシステムの初期化
    backtest = AdvancedBacktest(
        betting_fraction=0.02,
        min_odds=5.0,
        confidence_threshold=0.65
    )
    
    # データの読み込み
    backtest.load_data()
    
    # バックテスト実行
    results = backtest.run_backtest()
    
    # 結果表示
    print("\n=== 結果 ===")
    print(f"初期資産: ¥{results['initial_capital']:,}")
    print(f"最終資産: ¥{results['final_capital']:,.0f}")
    print(f"総リターン: {results['total_return']:.2%}")
    print(f"総ベット数: {results['total_bets']}")
    print(f"勝利数: {results['total_wins']}")
    print(f"勝率: {results['win_rate']:.1%}")
    print(f"レースあたり平均ベット数: {results['avg_bets_per_race']:.2f}")
    
    print("\n=== 戦略の効果 ===")
    if results['total_return'] > 0:
        print("✓ プラスリターンを達成！")
        print("成功要因:")
        print("- 高配当馬への選択的ベット")
        print("- アンサンブルモデルによる予測精度向上")
        print("- 期待値の高いレースのみ参加")
    else:
        print("改善が必要な点:")
        print("- さらなる特徴量エンジニアリング")
        print("- ベッティング基準の調整")

if __name__ == "__main__":
    main()