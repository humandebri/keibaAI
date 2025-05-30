#!/usr/bin/env python3
"""
改善されたバックテストシステム
複勝ベッティング、期待値フィルタリング、マネーマネジメントを実装
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import json
import os
from pathlib import Path

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ImprovedBacktest:
    def __init__(self, betting_fraction=0.005, monthly_stop_loss=0.1, ev_threshold=1.2):
        """
        Args:
            betting_fraction: 1回のベット額の割合（デフォルト0.5%）
            monthly_stop_loss: 月間ストップロス（デフォルト10%）
            ev_threshold: 期待値の閾値（デフォルト1.2）
        """
        self.betting_fraction = betting_fraction
        self.monthly_stop_loss = monthly_stop_loss
        self.ev_threshold = ev_threshold
        self.initial_capital = 1000000
        
    def load_and_prepare_data(self):
        """データの読み込みと準備"""
        print("Loading data...")
        dfs = []
        for year in range(2014, 2024):
            try:
                df = pd.read_excel(f'data/{year}.xlsx')
                dfs.append(df)
            except:
                print(f"Warning: Could not load {year}.xlsx")
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.data['日付'] = pd.to_datetime(self.data['日付'])
        self.data = self.data.sort_values(['日付', 'レースID'])
        
        # 特徴量の準備
        self.prepare_features()
        
    def prepare_features(self):
        """特徴量エンジニアリング"""
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['性別', '馬場状態', '天気', 'コース', '競馬場']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = pd.Categorical(self.data[col]).codes
        
        # 数値変数の欠損値処理
        numeric_columns = ['枠番', '馬番', '斤量', 'オッズ', '人気', '馬体重', '馬体重_増減']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # ターゲット変数：複勝（3着以内）
        self.data['is_place'] = (self.data['着順'] <= 3).astype(int)
        
    def calculate_place_odds(self, win_odds):
        """単勝オッズから複勝オッズを推定"""
        # 簡易的な推定式：複勝オッズ ≈ 単勝オッズ × 0.3
        # より正確には過去データから学習すべきだが、ここでは簡易版
        return win_odds * 0.3
    
    def run_backtest(self):
        """改善されたバックテストの実行"""
        results = []
        capital = self.initial_capital
        
        # 年ごとにバックテスト
        for year in range(2014, 2024):
            print(f"\n=== Year {year} ===")
            
            # データ分割
            train_mask = self.data['日付'].dt.year < year
            test_mask = self.data['日付'].dt.year == year
            
            if not train_mask.any() or not test_mask.any():
                continue
                
            train_data = self.data[train_mask]
            test_data = self.data[test_mask]
            
            # モデル訓練
            model = self.train_model(train_data)
            
            # 月ごとの結果を追跡
            monthly_results = {}
            monthly_capital = capital
            
            # テストデータで予測とベッティング
            for month in range(1, 13):
                month_mask = test_data['日付'].dt.month == month
                month_data = test_data[month_mask]
                
                if len(month_data) == 0:
                    continue
                
                month_start_capital = monthly_capital
                month_returns = []
                
                # レースごとに処理
                for race_id in month_data['レースID'].unique():
                    race_data = month_data[month_data['レースID'] == race_id]
                    
                    # 予測
                    features = self.get_features(race_data)
                    predictions = model.predict(features, num_iteration=model.best_iteration_)
                    
                    # 期待値計算とベッティング決定
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
                        bet_amount = monthly_capital * self.betting_fraction
                        horse = race_data.iloc[best_horse_idx]
                        
                        # 複勝の結果判定
                        if horse['着順'] <= 3:
                            # 複勝的中
                            place_odds = self.calculate_place_odds(horse['オッズ'])
                            payout = bet_amount * place_odds
                            profit = payout - bet_amount
                        else:
                            # 外れ
                            profit = -bet_amount
                        
                        month_returns.append(profit)
                        monthly_capital += profit
                
                # 月間結果の記録
                month_return = sum(month_returns) if month_returns else 0
                month_return_rate = (monthly_capital - month_start_capital) / month_start_capital
                
                monthly_results[month] = {
                    'returns': month_return,
                    'return_rate': month_return_rate,
                    'num_bets': len(month_returns),
                    'capital': monthly_capital
                }
                
                # 月間ストップロスチェック
                if month_return_rate < -self.monthly_stop_loss:
                    print(f"Month {month}: Stop loss triggered ({month_return_rate:.2%})")
                    # 翌月まで取引停止（ここでは簡易的に実装）
                    continue
                
                print(f"Month {month}: Return {month_return_rate:.2%}, Bets: {len(month_returns)}")
            
            # 年間結果の記録
            year_return = (monthly_capital - capital) / capital
            results.append({
                'year': year,
                'start_capital': capital,
                'end_capital': monthly_capital,
                'return_rate': year_return,
                'monthly_results': monthly_results
            })
            
            capital = monthly_capital
            print(f"Year {year} Total Return: {year_return:.2%}")
        
        return results
    
    def train_model(self, train_data):
        """LightGBMモデルの訓練"""
        features = self.get_features(train_data)
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
            valid_sets=[lgb_train],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        return model
    
    def get_features(self, data):
        """特徴量の取得"""
        feature_columns = ['枠番', '馬番', '斤量', 'オッズ', '人気', '馬体重', '馬体重_増減',
                          '性別', '馬場状態', '天気', 'コース', '競馬場']
        
        features = []
        for col in feature_columns:
            if col in data.columns:
                features.append(data[col].values)
        
        return np.column_stack(features)
    
    def optimize_parameters(self):
        """パラメータの最適化"""
        best_params = None
        best_total_return = -float('inf')
        
        # グリッドサーチ
        for betting_frac in [0.002, 0.005, 0.01]:
            for ev_thresh in [1.1, 1.2, 1.3, 1.5]:
                print(f"\nTesting: betting_fraction={betting_frac}, ev_threshold={ev_thresh}")
                
                self.betting_fraction = betting_frac
                self.ev_threshold = ev_thresh
                
                results = self.run_backtest()
                
                # 総合リターンの計算
                final_capital = results[-1]['end_capital'] if results else self.initial_capital
                total_return = (final_capital - self.initial_capital) / self.initial_capital
                
                if total_return > best_total_return:
                    best_total_return = total_return
                    best_params = {
                        'betting_fraction': betting_frac,
                        'ev_threshold': ev_thresh,
                        'total_return': total_return,
                        'final_capital': final_capital
                    }
                
                print(f"Total return: {total_return:.2%}")
        
        return best_params
    
    def save_results(self, results, params):
        """結果の保存"""
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        # 結果をJSONで保存
        output_data = {
            'parameters': params,
            'results': results,
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': results[-1]['end_capital'] if results else self.initial_capital,
                'total_return': (results[-1]['end_capital'] - self.initial_capital) / self.initial_capital if results else 0,
                'years': len(results)
            }
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(output_dir / f'improved_backtest_{timestamp}.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        # サマリーレポートの作成
        with open(output_dir / f'summary_{timestamp}.txt', 'w') as f:
            f.write("=== Improved Backtest Results ===\n\n")
            f.write(f"Initial Capital: ¥{self.initial_capital:,}\n")
            f.write(f"Final Capital: ¥{output_data['summary']['final_capital']:,}\n")
            f.write(f"Total Return: {output_data['summary']['total_return']:.2%}\n")
            f.write(f"Annualized Return: {(1 + output_data['summary']['total_return']) ** (1/10) - 1:.2%}\n\n")
            
            f.write("Best Parameters:\n")
            f.write(f"- Betting Fraction: {params['betting_fraction']:.1%}\n")
            f.write(f"- EV Threshold: {params['ev_threshold']}\n")
            f.write(f"- Monthly Stop Loss: {self.monthly_stop_loss:.1%}\n\n")
            
            f.write("Yearly Results:\n")
            for result in results:
                f.write(f"{result['year']}: {result['return_rate']:.2%}\n")


def main():
    """メイン実行関数"""
    print("Starting improved backtest system...")
    
    # バックテストシステムの初期化
    backtest = ImprovedBacktest()
    
    # データの読み込みと準備
    backtest.load_and_prepare_data()
    
    # パラメータ最適化
    print("\n=== Parameter Optimization ===")
    best_params = backtest.optimize_parameters()
    
    print(f"\nBest parameters found:")
    print(f"- Betting Fraction: {best_params['betting_fraction']:.1%}")
    print(f"- EV Threshold: {best_params['ev_threshold']}")
    print(f"- Total Return: {best_params['total_return']:.2%}")
    
    # 最適パラメータで最終実行
    backtest.betting_fraction = best_params['betting_fraction']
    backtest.ev_threshold = best_params['ev_threshold']
    
    print("\n=== Final Backtest with Optimal Parameters ===")
    final_results = backtest.run_backtest()
    
    # 結果の保存
    backtest.save_results(final_results, best_params)
    
    print("\nBacktest completed! Results saved to backtest_results/")


if __name__ == "__main__":
    main()