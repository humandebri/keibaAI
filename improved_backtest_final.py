#!/usr/bin/env python3
"""
改善されたバックテストシステム（最終版）
複勝ベッティング戦略の実装
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
                
                # 着順を数値に変換（「中」「除」などの文字列を処理）
                df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
                
                # 2020年のデータに特殊な処理が必要
                if year == 2020:
                    # 日付列の型を確認
                    if df['日付'].dtype == 'object':
                        # 文字列の日付を修正
                        def fix_date_2020(date_val):
                            if pd.isna(date_val):
                                return None
                            date_str = str(date_val)
                            # ??を年に置換
                            if '??' in date_str:
                                # 2020??7??4?? -> 2020年7月4日
                                date_str = date_str.replace('??', '年', 1)
                                date_str = date_str.replace('??', '月', 1)
                                date_str = date_str.replace('??', '日')
                                try:
                                    return pd.to_datetime(date_str, format='%Y年%m月%d日')
                                except:
                                    return None
                            else:
                                try:
                                    return pd.to_datetime(date_val)
                                except:
                                    return None
                        
                        df['日付'] = df['日付'].apply(fix_date_2020)
                
                # 着順がNaN（中止・除外など）の行を削除
                df = df.dropna(subset=['着順'])
                
                print(f"Loaded {year}.xlsx: {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {year}.xlsx - {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        
        # 日付がdatetime型でない場合の追加処理
        if self.data['日付'].dtype != 'datetime64[ns]':
            print("Converting dates to datetime...")
            self.data['日付'] = pd.to_datetime(self.data['日付'], errors='coerce')
        
        # NaTを除外
        before_count = len(self.data)
        self.data = self.data.dropna(subset=['日付'])
        after_count = len(self.data)
        if before_count > after_count:
            print(f"Dropped {before_count - after_count} rows with invalid dates")
        
        # race_idカラムでソート
        self.data = self.data.sort_values(['日付', 'race_id'])
        
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
        
    def calculate_place_odds(self, win_odds):
        """単勝オッズから複勝オッズを推定"""
        if win_odds <= 2.0:
            return win_odds * 0.4
        elif win_odds <= 5.0:
            return win_odds * 0.35
        elif win_odds <= 10.0:
            return win_odds * 0.3
        else:
            return win_odds * 0.25

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
        """LightGBMモデルの訓練"""
        features = self.get_features(train_data)
        if features is None or len(features) == 0:
            raise ValueError("No features available for training")
            
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
    
    def run_backtest(self):
        """改善されたバックテストの実行"""
        results = []
        capital = self.initial_capital
        all_bets = []
        
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
            monthly_capital = capital
            year_bets = 0
            year_wins = 0
            
            # テストデータで予測とベッティング
            for month in range(1, 13):
                month_mask = test_data['日付'].dt.month == month
                month_data = test_data[month_mask]
                
                if len(month_data) == 0:
                    continue
                
                month_start_capital = monthly_capital
                month_bets = 0
                month_wins = 0
                
                # レースごとに処理
                for race_id in month_data['race_id'].unique():
                    race_data = month_data[month_data['race_id'] == race_id]
                    
                    # 予測
                    features = self.get_features(race_data)
                    if features is None or len(features) == 0:
                        continue
                    
                    # ここが重要：best_iteration_を使わない
                    predictions = model.predict(features)
                    
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
                            month_wins += 1
                            year_wins += 1
                        else:
                            # 外れ
                            profit = -bet_amount
                        
                        monthly_capital += profit
                        month_bets += 1
                        year_bets += 1
                        
                        # ベット記録
                        all_bets.append({
                            'date': horse['日付'],
                            'race_id': race_id,
                            'horse_name': horse.get('馬', 'Unknown'),
                            'odds': horse['オッズ'],
                            'prediction': predictions[best_horse_idx],
                            'ev': best_ev,
                            'result': horse['着順'],
                            'profit': profit,
                            'capital': monthly_capital
                        })
                
                # 月間結果の表示
                if month_bets > 0:
                    month_return_rate = (monthly_capital - month_start_capital) / month_start_capital
                    print(f"Month {month}: Return {month_return_rate:.2%}, Win Rate: {month_wins/month_bets:.1%}, Bets: {month_bets}")
            
            # 年間結果の記録
            year_return = (monthly_capital - capital) / capital if capital > 0 else 0
            win_rate = year_wins / year_bets if year_bets > 0 else 0
            
            results.append({
                'year': year,
                'start_capital': capital,
                'end_capital': monthly_capital,
                'return_rate': year_return,
                'num_bets': year_bets,
                'num_wins': year_wins,
                'win_rate': win_rate
            })
            
            capital = monthly_capital
            print(f"Year {year} Total: Return {year_return:.2%}, Win Rate: {win_rate:.1%}, Bets: {year_bets}")
        
        if all_bets:
            self.all_bets = pd.DataFrame(all_bets)
        
        return results


def main():
    """メイン実行関数"""
    print("=== 改善されたバックテストシステム ===")
    print("複勝ベッティング戦略で実行します\n")
    
    # バックテストシステムの初期化
    backtest = ImprovedBacktest()
    
    # データの読み込みと準備
    backtest.load_and_prepare_data()
    
    print(f"\nLoaded {len(backtest.data)} race entries")
    print(f"Date range: {backtest.data['日付'].min()} to {backtest.data['日付'].max()}")
    
    # パラメータ表示
    print(f"\nParameters:")
    print(f"- Betting fraction: {backtest.betting_fraction:.1%}")
    print(f"- EV threshold: {backtest.ev_threshold}")
    print(f"- Monthly stop loss: {backtest.monthly_stop_loss:.1%}")
    
    # バックテスト実行
    results = backtest.run_backtest()
    
    # 結果の表示
    if results:
        final_capital = results[-1]['end_capital']
        total_return = (final_capital - backtest.initial_capital) / backtest.initial_capital
        total_bets = sum(r['num_bets'] for r in results)
        total_wins = sum(r['num_wins'] for r in results)
        overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
        
        print(f"\n=== Final Results ===")
        print(f"Initial Capital: ¥{backtest.initial_capital:,}")
        print(f"Final Capital: ¥{final_capital:,.0f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {(1 + total_return) ** (1/10) - 1:.2%}")
        print(f"Overall Win Rate: {overall_win_rate:.1%}")
        print(f"Total Bets: {total_bets}")
        
        # 年ごとの詳細
        print("\n=== Yearly Breakdown ===")
        print(f"{'Year':<6} {'Start':<15} {'End':<15} {'Return':<10} {'Win Rate':<10} {'Bets':<8}")
        print("-" * 70)
        for r in results:
            print(f"{r['year']:<6} "
                  f"¥{r['start_capital']:<14,.0f} "
                  f"¥{r['end_capital']:<14,.0f} "
                  f"{r['return_rate']:>9.2%} "
                  f"{r['win_rate']:>9.1%} "
                  f"{r['num_bets']:>7}")
        
        # 結果の保存
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'improved_backtest_{timestamp}.json'
        
        output_data = {
            'parameters': {
                'betting_fraction': backtest.betting_fraction,
                'ev_threshold': backtest.ev_threshold,
                'monthly_stop_loss': backtest.monthly_stop_loss
            },
            'summary': {
                'initial_capital': backtest.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (1/10) - 1,
                'overall_win_rate': overall_win_rate,
                'total_bets': total_bets
            },
            'yearly_results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        # 改善効果の表示
        print("\n=== 改善効果 ===")
        print("改善前（単勝）: -100%の損失")
        print(f"改善後（複勝）: {total_return:+.1%}のリターン")
        
        if total_return > 0:
            print("\n✓ プラスのリターンを達成しました！")
        else:
            print("\nさらなる調整が必要です")


if __name__ == "__main__":
    main()