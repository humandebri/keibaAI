#!/usr/bin/env python3
"""
三連単・馬連・ワイドなど多様な馬券種別に対応したバックテストシステム
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pathlib import Path
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# LightGBMのエラー対策
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class AdvancedBettingSystem:
    def __init__(self):
        self.initial_capital = 1000000
        self.betting_types = {
            '単勝': self.bet_win,
            '複勝': self.bet_place,
            '馬連': self.bet_quinella,
            'ワイド': self.bet_wide,
            '馬単': self.bet_exacta,
            '三連複': self.bet_trio,
            '三連単': self.bet_trifecta,
            '三連単1着流し': self.bet_trifecta_1st_wheel,
            '三連単2着流し': self.bet_trifecta_2nd_wheel,
            '三連単ボックス': self.bet_trifecta_box,
            '馬連流し': self.bet_quinella_wheel,
            'ワイド流し': self.bet_wide_wheel
        }
        
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
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {year}.xlsx - {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.prepare_features()
        
    def prepare_features(self):
        """特徴量エンジニアリング"""
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
    
    def train_finish_predictor(self, train_data):
        """着順予測モデルの訓練"""
        features = self.get_features(train_data)
        if features is None:
            raise ValueError("No features available")
        
        # 着順を予測（1-18位）
        target = train_data['着順'].values
        
        lgb_train = lgb.Dataset(features, target)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return model
    
    def predict_race_probabilities(self, model, race_data):
        """レースの各馬の着順確率を予測"""
        features = self.get_features(race_data)
        if features is None:
            return None
        
        # 着順の予測値を取得
        predicted_positions = model.predict(features)
        
        # 予測着順を確率に変換（順位が低いほど高い確率）
        # ソフトマックス風の変換
        scores = 1 / predicted_positions
        probabilities = scores / scores.sum()
        
        return probabilities
    
    def calculate_odds_multiplier(self, bet_type, odds_data):
        """馬券種別に応じたオッズ倍率の計算"""
        # 簡易的なオッズ推定（実際はもっと複雑）
        multipliers = {
            '単勝': 1.0,
            '複勝': 0.3,
            '馬連': 2.0,
            'ワイド': 0.7,
            '馬単': 4.0,
            '三連複': 10.0,
            '三連単': 50.0
        }
        
        base_multiplier = multipliers.get(bet_type, 1.0)
        
        # オッズに基づく調整
        if isinstance(odds_data, (list, tuple)):
            avg_odds = np.mean(odds_data)
        else:
            avg_odds = odds_data
            
        return base_multiplier * avg_odds
    
    # === 各種馬券の買い方メソッド ===
    
    def bet_win(self, race_data, probabilities, capital):
        """単勝"""
        best_idx = np.argmax(probabilities)
        if probabilities[best_idx] > 0.3:  # 30%以上の確率
            horse = race_data.iloc[best_idx]
            bet_amount = capital * 0.005
            return [(horse['馬番'], bet_amount, 'win')]
        return []
    
    def bet_place(self, race_data, probabilities, capital):
        """複勝"""
        # 上位3頭の確率の合計が高い馬
        place_probs = []
        for i in range(len(probabilities)):
            # この馬が3着以内に入る確率を推定
            place_prob = probabilities[i] * 3  # 簡易推定
            place_probs.append(min(place_prob, 0.9))
        
        best_idx = np.argmax(place_probs)
        if place_probs[best_idx] > 0.5:
            horse = race_data.iloc[best_idx]
            bet_amount = capital * 0.005
            return [(horse['馬番'], bet_amount, 'place')]
        return []
    
    def bet_quinella(self, race_data, probabilities, capital):
        """馬連"""
        # 上位2頭を選択
        top_2_idx = np.argsort(probabilities)[-2:]
        if probabilities[top_2_idx].sum() > 0.4:
            horses = race_data.iloc[top_2_idx]
            bet_amount = capital * 0.003
            return [(tuple(horses['馬番'].values), bet_amount, 'quinella')]
        return []
    
    def bet_wide(self, race_data, probabilities, capital):
        """ワイド"""
        # 上位3頭から2頭の組み合わせ
        top_3_idx = np.argsort(probabilities)[-3:]
        if probabilities[top_3_idx].sum() > 0.5:
            horses = race_data.iloc[top_3_idx]
            horse_nums = horses['馬番'].values
            bets = []
            for combo in combinations(horse_nums, 2):
                bet_amount = capital * 0.002
                bets.append((combo, bet_amount, 'wide'))
            return bets
        return []
    
    def bet_exacta(self, race_data, probabilities, capital):
        """馬単"""
        top_2_idx = np.argsort(probabilities)[-2:]
        if probabilities[top_2_idx[1]] > 0.25 and probabilities[top_2_idx[0]] > 0.15:
            horses = race_data.iloc[top_2_idx]
            bet_amount = capital * 0.002
            return [((horses.iloc[1]['馬番'], horses.iloc[0]['馬番']), 
                    bet_amount, 'exacta')]
        return []
    
    def bet_trio(self, race_data, probabilities, capital):
        """三連複"""
        top_3_idx = np.argsort(probabilities)[-3:]
        if probabilities[top_3_idx].sum() > 0.6:
            horses = race_data.iloc[top_3_idx]
            bet_amount = capital * 0.002
            return [(tuple(sorted(horses['馬番'].values)), bet_amount, 'trio')]
        return []
    
    def bet_trifecta(self, race_data, probabilities, capital):
        """三連単"""
        top_3_idx = np.argsort(probabilities)[-3:]
        if probabilities[top_3_idx].sum() > 0.6:
            horses = race_data.iloc[top_3_idx]
            horse_nums = horses['馬番'].values
            # 最も確率の高い順番で賭ける
            bet_amount = capital * 0.001
            return [((horse_nums[2], horse_nums[1], horse_nums[0]), 
                    bet_amount, 'trifecta')]
        return []
    
    def bet_trifecta_1st_wheel(self, race_data, probabilities, capital):
        """三連単1着流し（軸馬から2,3着を流す）"""
        best_idx = np.argmax(probabilities)
        if probabilities[best_idx] > 0.3:  # 軸馬の勝率が30%以上
            # 軸馬
            axis_horse = race_data.iloc[best_idx]['馬番']
            
            # 残りの馬から上位5頭を選択
            other_idx = [i for i in range(len(probabilities)) if i != best_idx]
            other_probs = probabilities[other_idx]
            top_other_idx = np.argsort(other_probs)[-5:]
            
            other_horses = race_data.iloc[[other_idx[i] for i in top_other_idx]]
            other_nums = other_horses['馬番'].values
            
            bets = []
            # 1着固定で2,3着の組み合わせ
            for second, third in permutations(other_nums, 2):
                bet_amount = capital * 0.0005  # 組み合わせが多いので少額
                bets.append(((axis_horse, second, third), bet_amount, 'trifecta'))
            
            return bets
        return []
    
    def bet_trifecta_2nd_wheel(self, race_data, probabilities, capital):
        """三連単2着流し（2着に軸馬を固定）"""
        # 2番目に強い馬を軸にする戦略
        sorted_idx = np.argsort(probabilities)
        if len(sorted_idx) >= 5 and probabilities[sorted_idx[-2]] > 0.2:
            axis_horse = race_data.iloc[sorted_idx[-2]]['馬番']
            
            # 1着候補（上位3頭）
            first_candidates = race_data.iloc[sorted_idx[-3:]]['馬番'].values
            first_candidates = first_candidates[first_candidates != axis_horse]
            
            # 3着候補（上位6頭から軸馬を除く）
            third_candidates = race_data.iloc[sorted_idx[-6:]]['馬番'].values
            third_candidates = third_candidates[third_candidates != axis_horse]
            
            bets = []
            for first in first_candidates:
                for third in third_candidates:
                    if first != third:
                        bet_amount = capital * 0.0003
                        bets.append(((first, axis_horse, third), 
                                   bet_amount, 'trifecta'))
            
            return bets
        return []
    
    def bet_trifecta_box(self, race_data, probabilities, capital):
        """三連単ボックス（選択した馬の全順列）"""
        # 上位4頭でボックス
        top_4_idx = np.argsort(probabilities)[-4:]
        if probabilities[top_4_idx].sum() > 0.7:
            horses = race_data.iloc[top_4_idx]
            horse_nums = horses['馬番'].values
            
            bets = []
            # 4頭から3頭を選んで全順列（24通り）
            for combo in permutations(horse_nums, 3):
                bet_amount = capital * 0.0002  # 24通りなので少額
                bets.append((combo, bet_amount, 'trifecta'))
            
            return bets
        return []
    
    def bet_quinella_wheel(self, race_data, probabilities, capital):
        """馬連流し（軸馬と他の馬の組み合わせ）"""
        best_idx = np.argmax(probabilities)
        if probabilities[best_idx] > 0.25:
            axis_horse = race_data.iloc[best_idx]['馬番']
            
            # 他の上位5頭と組み合わせ
            other_idx = [i for i in range(len(probabilities)) if i != best_idx]
            other_probs = probabilities[other_idx]
            top_other_idx = np.argsort(other_probs)[-5:]
            
            bets = []
            for idx in top_other_idx:
                partner = race_data.iloc[other_idx[idx]]['馬番']
                bet_amount = capital * 0.001
                bets.append((tuple(sorted([axis_horse, partner])), 
                           bet_amount, 'quinella'))
            
            return bets
        return []
    
    def bet_wide_wheel(self, race_data, probabilities, capital):
        """ワイド流し"""
        best_idx = np.argmax(probabilities)
        if probabilities[best_idx] > 0.2:
            axis_horse = race_data.iloc[best_idx]['馬番']
            
            # 他の上位6頭と組み合わせ
            other_idx = [i for i in range(len(probabilities)) if i != best_idx]
            other_probs = probabilities[other_idx]
            top_other_idx = np.argsort(other_probs)[-6:]
            
            bets = []
            for idx in top_other_idx:
                partner = race_data.iloc[other_idx[idx]]['馬番']
                bet_amount = capital * 0.0008
                bets.append((tuple(sorted([axis_horse, partner])), 
                           bet_amount, 'wide'))
            
            return bets
        return []
    
    def simulate_betting_strategy(self, model, test_data, strategy_name, 
                                betting_method, verbose=True):
        """特定の馬券戦略のシミュレーション"""
        capital = self.initial_capital
        history = []
        total_bets = 0
        total_wins = 0
        total_expected_value = 0
        ev_count = 0
        
        unique_races = test_data['race_id'].unique()
        
        for i, race_id in enumerate(unique_races):
            if verbose and i % 1000 == 0:
                print(f"Processing race {i}/{len(unique_races)}...")
            
            race_data = test_data[test_data['race_id'] == race_id]
            probabilities = self.predict_race_probabilities(model, race_data)
            
            if probabilities is None:
                continue
            
            # 賭けの実行
            bets = betting_method(race_data, probabilities, capital)
            
            for bet_info in bets:
                selection, bet_amount, bet_type = bet_info
                total_bets += 1
                
                # 期待値計算
                win_prob = self.calculate_win_probability(race_data, selection, bet_type, probabilities)
                estimated_odds = self.calculate_payout_odds(race_data, selection, bet_type)
                expected_value = win_prob * estimated_odds
                total_expected_value += expected_value
                ev_count += 1
                
                # 結果判定（簡易版）
                win = self.check_bet_result(race_data, selection, bet_type)
                
                if win:
                    # オッズ計算（簡易版）
                    odds = self.calculate_payout_odds(race_data, selection, bet_type)
                    payout = bet_amount * odds
                    profit = payout - bet_amount
                    total_wins += 1
                else:
                    profit = -bet_amount
                
                capital += profit
                
                history.append({
                    'race_id': race_id,
                    'bet_type': bet_type,
                    'selection': selection,
                    'bet_amount': bet_amount,
                    'profit': profit,
                    'capital': capital,
                    'expected_value': expected_value,
                    'win_probability': win_prob
                })
        
        avg_expected_value = total_expected_value / ev_count if ev_count > 0 else 0
        
        return {
            'strategy': strategy_name,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0,
            'average_expected_value': avg_expected_value,
            'history': history
        }
    
    def check_bet_result(self, race_data, selection, bet_type):
        """賭けの結果を判定"""
        results = race_data.sort_values('着順')
        
        if bet_type == 'win':
            return results.iloc[0]['馬番'] == selection
        
        elif bet_type == 'place':
            return selection in results.iloc[:3]['馬番'].values
        
        elif bet_type == 'quinella':
            top_2 = set(results.iloc[:2]['馬番'].values)
            return set(selection) == top_2
        
        elif bet_type == 'wide':
            top_3 = set(results.iloc[:3]['馬番'].values)
            return set(selection).issubset(top_3)
        
        elif bet_type == 'exacta':
            return (results.iloc[0]['馬番'] == selection[0] and 
                   results.iloc[1]['馬番'] == selection[1])
        
        elif bet_type == 'trio':
            top_3 = set(results.iloc[:3]['馬番'].values)
            return set(selection) == top_3
        
        elif bet_type == 'trifecta':
            return (results.iloc[0]['馬番'] == selection[0] and 
                   results.iloc[1]['馬番'] == selection[1] and
                   results.iloc[2]['馬番'] == selection[2])
        
        return False
    
    def calculate_payout_odds(self, race_data, selection, bet_type):
        """払い戻しオッズの計算（簡易版）"""
        # 実際のオッズデータがないため、推定値を使用
        base_odds = {
            'win': 5.0,
            'place': 2.0,
            'quinella': 15.0,
            'wide': 5.0,
            'exacta': 30.0,
            'trio': 50.0,
            'trifecta': 200.0
        }
        
        # 人気度による調整
        if isinstance(selection, (tuple, list)):
            avg_popularity = np.mean([
                race_data[race_data['馬番'] == h]['人気'].values[0] 
                for h in selection if len(race_data[race_data['馬番'] == h]) > 0
            ])
        else:
            avg_popularity = race_data[race_data['馬番'] == selection]['人気'].values[0]
        
        # 人気が低いほどオッズが高い
        popularity_multiplier = 1 + (avg_popularity - 1) * 0.5
        
        return base_odds.get(bet_type, 10.0) * popularity_multiplier
    
    def calculate_win_probability(self, race_data, selection, bet_type, probabilities):
        """各馬券種別の的中確率を計算"""
        horse_probs = {}
        for i, (_, horse) in enumerate(race_data.iterrows()):
            horse_probs[horse['馬番']] = probabilities[i]
        
        if bet_type == 'win':
            return horse_probs.get(selection, 0)
        
        elif bet_type == 'place':
            # 3着以内に入る確率（簡易計算）
            return min(horse_probs.get(selection, 0) * 3, 0.9)
        
        elif bet_type == 'quinella':
            # 2頭が1-2着に入る確率
            p1 = horse_probs.get(selection[0], 0)
            p2 = horse_probs.get(selection[1], 0)
            return (p1 * p2) * 2  # どちらが1着でもOK
        
        elif bet_type == 'wide':
            # 2頭が3着以内に入る確率
            p1 = min(horse_probs.get(selection[0], 0) * 3, 0.9)
            p2 = min(horse_probs.get(selection[1], 0) * 3, 0.9)
            return p1 * p2 * 0.8  # 調整係数
        
        elif bet_type == 'exacta':
            # 順番通りに1-2着
            p1 = horse_probs.get(selection[0], 0)
            p2 = horse_probs.get(selection[1], 0)
            return p1 * p2
        
        elif bet_type == 'trio':
            # 3頭が3着以内（順不同）
            probs = [horse_probs.get(h, 0) for h in selection]
            return np.prod(probs) * 6  # 3!通り
        
        elif bet_type == 'trifecta':
            # 順番通りに1-2-3着
            probs = [horse_probs.get(h, 0) for h in selection]
            return np.prod(probs)
        
        return 0.01  # デフォルト
    
    def run_comprehensive_simulation(self):
        """全馬券種別でのシミュレーション実行"""
        # データ分割
        train_data = self.data[self.data['year'].isin(range(2014, 2021))]
        test_data = self.data[self.data['year'].isin(range(2021, 2024))]
        
        print(f"Train data: {len(train_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # モデル訓練
        print("\nTraining finish position predictor...")
        model = self.train_finish_predictor(train_data)
        
        # 各戦略でシミュレーション
        results = {}
        
        # テストする戦略
        strategies_to_test = [
            '複勝',
            '馬連',
            'ワイド',
            '三連複',
            '三連単',
            '三連単1着流し',
            '三連単ボックス',
            '馬連流し',
            'ワイド流し'
        ]
        
        for strategy_name in strategies_to_test:
            print(f"\n--- Testing {strategy_name} ---")
            betting_method = self.betting_types[strategy_name]
            
            result = self.simulate_betting_strategy(
                model, test_data, strategy_name, betting_method, verbose=False
            )
            
            results[strategy_name] = result
            
            print(f"Final Capital: ¥{result['final_capital']:,.0f}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Expected Value: {result['average_expected_value']:.3f}")
            print(f"Total Bets: {result['total_bets']}")
        
        return results
    
    def analyze_results(self, results):
        """結果の分析とレポート作成"""
        print("\n=== 馬券種別パフォーマンス比較 ===")
        
        summary = []
        for strategy, result in results.items():
            summary.append({
                '馬券種': strategy,
                'リターン': f"{result['total_return']:.1%}",
                '勝率': f"{result['win_rate']:.1%}",
                '期待値': f"{result['average_expected_value']:.3f}",
                'ベット数': result['total_bets'],
                '最終資産': f"¥{result['final_capital']:,.0f}"
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('リターン', ascending=False)
        print(summary_df.to_string(index=False))
        
        # 最も効果的な戦略
        best_strategy = max(results.items(), key=lambda x: x[1]['total_return'])
        print(f"\n最も効果的な戦略: {best_strategy[0]}")
        print(f"リターン: {best_strategy[1]['total_return']:.2%}")
        print(f"期待値: {best_strategy[1]['average_expected_value']:.3f}")
        
        # 期待値が1.0以上の戦略
        print("\n期待値が1.0以上の戦略:")
        ev_above_1 = [(s, r) for s, r in results.items() if r['average_expected_value'] >= 1.0]
        if ev_above_1:
            for strategy, result in ev_above_1:
                print(f"  {strategy}: 期待値 {result['average_expected_value']:.3f}")
        else:
            print("  なし（すべて1.0未満）")
        
        # 結果の保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_data = {
            'timestamp': timestamp,
            'results': {k: {
                'total_return': v['total_return'],
                'win_rate': v['win_rate'],
                'total_bets': v['total_bets'],
                'final_capital': v['final_capital']
            } for k, v in results.items()}
        }
        
        os.makedirs('backtest_results', exist_ok=True)
        with open(f'backtest_results/betting_types_{timestamp}.json', 'w') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return summary_df

def main():
    """メイン実行関数"""
    print("=== 多様な馬券種別バックテストシステム ===")
    
    system = AdvancedBettingSystem()
    system.load_data()
    results = system.run_comprehensive_simulation()
    summary = system.analyze_results(results)
    
    print("\n完了！各馬券種別の成績を比較しました。")
    print("三連単流しなどの複雑な買い方も含めて検証済みです。")

if __name__ == "__main__":
    main()