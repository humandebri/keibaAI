#!/usr/bin/env python3
"""
現実的なバックテストシステム
実際の競馬の仕組みを正確に反映
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pathlib import Path

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class RealisticBacktest:
    def __init__(self, betting_fraction=0.01):
        self.betting_fraction = betting_fraction
        self.initial_capital = 1000000
        self.jra_take_rate = 0.2  # JRA控除率20%
        
    def load_data(self):
        """データの読み込み"""
        print("Loading data...")
        dfs = []
        for year in range(2014, 2024):
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
        
        # 実際の複勝率を確認
        place_rate = self.data['is_place'].mean()
        print(f"Actual place rate: {place_rate:.1%}")
        
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
    
    def calculate_place_return(self, popularity):
        """人気順位から複勝配当を推定（100円に対する払戻）"""
        if popularity <= 3:
            return np.random.uniform(110, 150)  # 1-3番人気: 110-150円
        elif popularity <= 6:
            return np.random.uniform(130, 200)  # 4-6番人気: 130-200円
        elif popularity <= 10:
            return np.random.uniform(150, 300)  # 7-10番人気: 150-300円
        else:
            return np.random.uniform(200, 500)  # 11番人気以下: 200-500円
    
    def run_backtest(self):
        """現実的なバックテスト実行"""
        # 訓練データとテストデータの分割
        train_years = range(2014, 2021)
        test_years = range(2021, 2024)
        
        train_data = self.data[self.data['year'].isin(train_years)]
        test_data = self.data[self.data['year'].isin(test_years)]
        
        print(f"\nTrain data: {len(train_data)} rows (2014-2020)")
        print(f"Test data: {len(test_data)} rows (2021-2023)")
        
        # モデル訓練
        print("\nTraining model...")
        model = self.train_model(train_data)
        
        # バックテスト実行
        print("\nRunning realistic backtest...")
        capital = self.initial_capital
        total_bets = 0
        total_wins = 0
        monthly_results = []
        
        # レースごとに処理
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if i % 1000 == 0:
                print(f"Processing race {i}/{len(unique_races)}...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            
            # 最低出走頭数チェック
            if len(race_data) < 5:
                continue
            
            # 予測
            features = self.get_features(race_data)
            if features is None:
                continue
                
            predictions = model.predict(features)
            
            # 予測確率トップ3を選択（実際の複勝と同じ）
            top_3_indices = np.argsort(predictions)[-3:]
            
            # 各馬に均等にベット
            bet_per_horse = capital * self.betting_fraction / 3
            
            race_profit = 0
            for idx in top_3_indices:
                horse = race_data.iloc[idx]
                total_bets += 1
                
                if horse['着順'] <= 3:
                    # 複勝的中
                    popularity = int(horse['人気'])
                    place_return = self.calculate_place_return(popularity)
                    payout = bet_per_horse * place_return / 100
                    race_profit += (payout - bet_per_horse)
                    total_wins += 1
                else:
                    # 外れ
                    race_profit -= bet_per_horse
            
            capital += race_profit
            
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
            'win_rate': total_wins / total_bets if total_bets > 0 else 0
        }

def main():
    """メイン実行関数"""
    print("=== 現実的なバックテスト ===")
    
    # バックテストシステムの初期化
    backtest = RealisticBacktest()
    
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
    
    print("\n=== 分析 ===")
    if results['total_return'] < 0:
        print("マイナスリターンの主な要因:")
        print("1. JRA控除率20%の影響")
        print("2. 予測精度の限界")
        print("3. 人気馬への集中による低配当")
    else:
        print("プラスリターンを達成！")

if __name__ == "__main__":
    main()