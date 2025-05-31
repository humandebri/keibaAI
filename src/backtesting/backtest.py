#!/usr/bin/env python3
"""
シンプルなバックテストシステム
日付を使わずにrace_idベースで処理
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pathlib import Path

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SimpleBacktest:
    def __init__(self, betting_fraction=0.005, ev_threshold=1.2):
        self.betting_fraction = betting_fraction
        self.ev_threshold = ev_threshold
        self.initial_capital = 1000000
        
    def load_data(self):
        """データの読み込み"""
        print("Loading data...")
        dfs = []
        for year in range(2014, 2024):
            try:
                df = pd.read_excel(f'data/{year}.xlsx')
                
                # 着順を数値に変換
                df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                
                # 着順がNaN（中止・除外など）の行を削除
                df = df.dropna(subset=['着順'])
                
                # 年の情報を追加（日付の代わりに使用）
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
        """特徴量エンジニアリング"""
        print("Preparing features...")
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['性', '馬場', '天気', '芝・ダート', '場名']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        # 数値変数の欠損値処理
        numeric_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # ターゲット変数：複勝（3着以内）
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
        
    def get_features(self, data):
        """特徴量の取得"""
        feature_columns = ['馬番', '斤量', 'オッズ', '人気', '体重', '体重変化',
                          '性', '馬場', '天気', '芝・ダート', '場名']
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                features.append(data[col].values)
        
        if len(features) == 0:
            return None
            
        return np.column_stack(features)
    
    def train_model(self, train_data):
        """モデルの訓練"""
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
            
        target = train_data['is_place']
        
        # クラス重み調整
        pos_weight = len(target[target == 0]) / len(target[target == 1])
        
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': pos_weight
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def calculate_place_odds(self, win_odds):
        """複勝オッズの推定"""
        if win_odds <= 2.0:
            return win_odds * 0.4
        elif win_odds <= 5.0:
            return win_odds * 0.35
        elif win_odds <= 10.0:
            return win_odds * 0.3
        else:
            return win_odds * 0.25
    
    def run_backtest(self):
        """シンプルなバックテスト実行"""
        # 訓練データとテストデータの分割（年ベース）
        train_years = range(2014, 2021)  # 2014-2020
        test_years = range(2021, 2024)   # 2021-2023
        
        train_data = self.data[self.data['year'].isin(train_years)]
        test_data = self.data[self.data['year'].isin(test_years)]
        
        print(f"\nTrain data: {len(train_data)} rows (2014-2020)")
        print(f"Test data: {len(test_data)} rows (2021-2023)")
        
        # モデル訓練
        print("\nTraining model...")
        model = self.train_model(train_data)
        
        # テストデータでバックテスト
        print("\nRunning backtest...")
        capital = self.initial_capital
        total_bets = 0
        total_wins = 0
        
        # レースごとに処理
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if i % 1000 == 0:
                print(f"Processing race {i}/{len(unique_races)}...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            # 予測
            features = self.get_features(race_data)
            if features is None:
                continue
                
            predictions = model.predict(features)
            
            # 期待値計算とベッティング
            best_horse_idx = None
            best_ev = 0
            
            for idx, (_, horse) in enumerate(race_data.iterrows()):
                win_odds = horse['オッズ']
                place_odds = self.calculate_place_odds(win_odds)
                place_prob = predictions[idx]
                
                # 期待値 = 確率 × オッズ
                ev = place_prob * place_odds
                
                if ev > self.ev_threshold and ev > best_ev:
                    best_ev = ev
                    best_horse_idx = idx
            
            # ベッティング実行
            if best_horse_idx is not None:
                bet_amount = capital * self.betting_fraction
                horse = race_data.iloc[best_horse_idx]
                
                total_bets += 1
                
                # 複勝の結果判定
                if horse['着順'] <= 3:
                    # 複勝的中
                    place_odds = self.calculate_place_odds(horse['オッズ'])
                    payout = bet_amount * place_odds
                    profit = payout - bet_amount
                    total_wins += 1
                else:
                    # 外れ
                    profit = -bet_amount
                
                capital += profit
        
        # 結果を返す
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0
        }

def main():
    """メイン実行関数"""
    print("=== シンプルバックテスト ===")
    
    # バックテストシステムの初期化
    backtest = SimpleBacktest()
    
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
    
    print("\n=== 改善効果 ===")
    print("改善前（単勝）: -100%の損失")
    print(f"改善後（複勝）: {results['total_return']:+.1%}のリターン")
    
    if results['total_return'] > 0:
        print("\n✓ プラスのリターンを達成しました！")

if __name__ == "__main__":
    main()