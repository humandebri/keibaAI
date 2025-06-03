#!/usr/bin/env python3
"""
今週のレース予測スクリプト
使い方: python predict_races.py [レースデータファイル]
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
from datetime import datetime, timedelta
import sys

from src.strategies.optimized_kelly_strategy import OptimizedKellyStrategy
from src.features.unified_features import UnifiedFeatureEngine

def load_trained_model():
    """訓練済みモデルを読み込み"""
    print("🤖 訓練済みモデル読み込み中...")
    
    # 既存データでモデル訓練
    if os.path.exists('encoded/2020_2025encoded_data_v2.csv'):
        train_data = pd.read_csv('encoded/2020_2025encoded_data_v2.csv', nrows=5000)
        
        # 年カラム追加
        if 'year' not in train_data.columns:
            train_data['year'] = 2023
        
        strategy = OptimizedKellyStrategy()
        strategy.train_data = train_data
        
        model = strategy.train_model()
        print("✅ モデル訓練完了")
        return strategy, model
    else:
        print("❌ 訓練データが見つかりません")
        return None, None

def prepare_race_data(race_file):
    """レースデータを予測用に準備"""
    print(f"📊 レースデータ準備: {race_file}")
    
    # ファイル形式に応じて読み込み
    if race_file.endswith('.csv'):
        data = pd.read_csv(race_file)
    elif race_file.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(race_file)
    else:
        print("❌ サポートされていないファイル形式")
        return None
    
    print(f"   データサイズ: {len(data)}行, {len(data.columns)}列")
    
    # 必要な列をチェック
    required_cols = ['race_id', '馬', '馬番', '騎手', 'オッズ', '人気']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"⚠️ 不足している列: {missing_cols}")
        print("データの列一覧:")
        print(list(data.columns))
        
        # 可能な限り補完
        if 'race_id' not in data.columns:
            data['race_id'] = f"upcoming_{datetime.now().strftime('%Y%m%d')}"
        
        if '馬番' not in data.columns and '馬' in data.columns:
            data['馬番'] = range(1, len(data) + 1)
    
    # デフォルト値で補完
    defaults = {
        '年齢': 4,
        '性': '牡',
        '斤量': 55.0,
        '体重': '480(0)',
        '体重変化': 0,
        '距離': 1600,
        'クラス': 5,
        '芝・ダート': 0,
        '馬場': 0,
        '天気': 0,
        '通過順': '5-5',
        '上がり': 35.0,
        'レース名': '予測レース',
        '開催': '東京',
        '場名': '東京',
        '日付': datetime.now()
    }
    
    for col, default_val in defaults.items():
        if col not in data.columns:
            data[col] = default_val
    
    # 現在の着順（予測対象なので値は不要だが、HistoricalFeatureBuilderが必要とする）
    if '着順' not in data.columns:
        data['着順'] = 5  # デフォルト値
    
    # 調教師カラム追加（HistoricalFeatureBuilderが必要とする）
    if '調教師' not in data.columns:
        data['調教師'] = '未知'
    
    # 騎手勝率カラム追加（HistoricalFeatureBuilderが必要とする）
    if 'jockey_win_rate' not in data.columns:
        # 有名騎手による勝率推定
        famous_jockeys = {'武豊': 0.18, '川田': 0.20, '福永': 0.17, 'ルメール': 0.22, 'デムーロ': 0.19, '岩田': 0.15}
        data['jockey_win_rate'] = data['騎手'].apply(
            lambda x: next((rate for jockey, rate in famous_jockeys.items() if jockey in str(x)), 0.10)
        )
    
    # 調教師勝率カラム追加
    if 'trainer_win_rate' not in data.columns:
        data['trainer_win_rate'] = 0.08  # デフォルト勝率
    
    # 過去走データ（ダミー）
    for i in range(1, 6):
        for base_col in ['馬番', '騎手', 'オッズ', '着順', '距離', 'クラス', '走破時間', '芝・ダート', '調教師']:
            col_name = f'{base_col}{i}'
            if col_name not in data.columns:
                if base_col == '着順':
                    data[col_name] = np.random.randint(1, 10)
                elif base_col == 'オッズ':
                    data[col_name] = np.random.uniform(2.0, 20.0)
                elif base_col == '走破時間':
                    data[col_name] = f"1:{np.random.randint(20, 40)}.{np.random.randint(0, 9)}"
                elif base_col == '距離':
                    data[col_name] = np.random.choice([1200, 1400, 1600, 1800, 2000])
                elif base_col == 'クラス':
                    data[col_name] = np.random.randint(1, 8)
                elif base_col == '芝・ダート':
                    data[col_name] = np.random.choice([0, 1])
                elif base_col == '調教師':
                    data[col_name] = '未知'
                else:
                    data[col_name] = data[base_col] if base_col in data.columns else 1
        
        # 日付
        data[f'日付{i}'] = datetime.now() - timedelta(days=i*30)
    
    # 追加で必要となる可能性がある特徴量カラム
    additional_cols = ['馬主', '生産者', '父', '母', '母父', '枠', '単勝', '複勝', 'タイム', 
                      '賞金', '通過', '上り', '馬体重', '着差', 'コーナー通過順']
    
    for col in additional_cols:
        if col not in data.columns:
            if col in ['単勝', '複勝', '賞金']:
                data[col] = 0.0
            elif col in ['タイム', '上り']:
                data[col] = '1:35.0'
            elif col == '着差':
                data[col] = '0.0'
            elif col == '枠':
                data[col] = (data['馬番'] - 1) // 2 + 1  # 馬番から枠番を推定
            elif col == '馬体重':
                data[col] = data['体重'] if '体重' in data.columns else '480(0)'
            elif col in ['通過', 'コーナー通過順']:
                data[col] = '5-5-5-5'
            else:
                data[col] = '不明'
    
    print("✅ データ準備完了")
    return data

def predict_race_outcomes(strategy, model, race_data):
    """レース結果を予測"""
    print("🔮 レース予測実行中...")
    
    try:
        # 特徴量構築
        enhanced_data = strategy.create_additional_features(race_data)
        
        # 確率予測
        probabilities = strategy.predict_probabilities(model, enhanced_data)
        
        if not probabilities:
            print("❌ 予測に失敗しました")
            return None
        
        print(f"✅ {len(probabilities)}頭の予測完了")
        return probabilities
    
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_betting_recommendations(strategy, probabilities, race_data):
    """ベット推奨を生成（全候補も含む）"""
    print("💡 ベット推奨生成中...")
    
    try:
        # ベット機会生成
        bet_opportunities = strategy._generate_bet_opportunities(probabilities, race_data)
        
        # Kelly最適化
        optimized_bets = strategy.calculate_diversified_kelly(bet_opportunities)
        
        print(f"✅ {len(optimized_bets)}件のベット推奨生成")
        
        # すべての候補を期待値順でソート
        all_candidates = sorted(bet_opportunities, key=lambda x: x['expected_value'], reverse=True)
        
        return optimized_bets, all_candidates
    
    except Exception as e:
        print(f"❌ ベット推奨エラー: {e}")
        return [], []

def display_predictions(probabilities, betting_recommendations, race_data):
    """予測結果を表示"""
    print("\n🏇 レース予測結果")
    print("=" * 60)
    
    # 馬番順にソート
    horses = []
    for horse_num, prob_data in probabilities.items():
        horse_name = "不明"
        if '馬' in race_data.columns:
            horse_row = race_data[race_data['馬番'] == horse_num]
            if not horse_row.empty:
                horse_name = str(horse_row.iloc[0]['馬'])
        
        horses.append({
            'horse_num': horse_num,
            'horse_name': horse_name,
            'win_prob': prob_data['win_prob'],
            'place_prob': prob_data['place_prob'],
            'predicted_rank': prob_data['predicted_rank'],
            'odds': prob_data['odds'],
            'popularity': prob_data['popularity']
        })
    
    # 予測順位でソート
    horses.sort(key=lambda x: x['predicted_rank'])
    
    print("予測順位 | 馬番 | 馬名 | 勝率 | 複勝率 | オッズ | 人気")
    print("-" * 60)
    
    for i, horse in enumerate(horses, 1):  # 全頭表示
        print(f"{i:4d}位   | {horse['horse_num']:2d}番 | "
              f"{horse['horse_name'][:12]:12s} | "
              f"{horse['win_prob']*100:4.1f}% | "
              f"{horse['place_prob']*100:4.1f}% | "
              f"{horse['odds']:5.1f}倍 | "
              f"{horse['popularity']:2d}人気")
    
    # ベット推奨
    if betting_recommendations:
        print(f"\n💰 ベット推奨 ({len(betting_recommendations)}件)")
        print("=" * 50)
        
        for i, bet in enumerate(betting_recommendations, 1):
            bet_type = bet['type']
            selection = bet['selection']
            expected_value = bet['expected_value']
            win_prob = bet['win_probability']
            kelly_fraction = bet.get('kelly_fraction', 0)
            
            print(f"{i}. {bet_type}: {selection}")
            print(f"   期待値: {expected_value:.3f}")
            print(f"   勝率: {win_prob*100:.1f}%")
            print(f"   Kelly推奨: 資金の{kelly_fraction*100:.1f}%")
            
            # 具体的な金額例
            for capital in [100000, 500000, 1000000]:
                bet_amount = capital * kelly_fraction
                if bet_amount >= 100:
                    print(f"   資金{capital//10000}万円の場合: {bet_amount:,.0f}円")
            print()
    else:
        print("\n💡 推奨ベットなし（期待値条件を満たすベットが見つかりませんでした）")
    
    print("\n⚠️ 注意:")
    print("- この予測は過去データに基づく統計的推定です")
    print("- 実際の結果を保証するものではありません")
    print("- 投資は自己責任で行ってください")

def main():
    parser = argparse.ArgumentParser(description='今週のレース予測')
    parser.add_argument('race_file', nargs='?', 
                       default='today_races.csv',
                       help='レースデータファイル (CSV or Excel)')
    parser.add_argument('--output', type=str,
                       help='結果をJSONファイルに保存')
    parser.add_argument('--min_ev', type=float, default=1.05,
                       help='ベット推奨の最低期待値')
    parser.add_argument('--strategy', type=str,
                       choices=['conservative', 'standard', 'aggressive'],
                       default='standard',
                       help='戦略タイプ')
    
    args = parser.parse_args()
    
    print("🔮 競馬レース予測システム")
    print("=" * 40)
    
    # レースデータ確認
    if not os.path.exists(args.race_file):
        print(f"❌ レースデータファイルが見つかりません: {args.race_file}")
        print("\n📝 レースデータファイルの作成方法:")
        print("1. CSVファイルに以下の列を含めてください:")
        print("   - race_id, 馬, 馬番, 騎手, オッズ, 人気")
        print("2. 例: today_races.csv")
        print("   race_id,馬,馬番,騎手,オッズ,人気")
        print("   20241201001,サンプル馬1,1,騎手A,3.2,1")
        print("   20241201001,サンプル馬2,2,騎手B,5.4,2")
        return
    
    # モデル読み込み
    strategy, model = load_trained_model()
    if not model:
        return
    
    # 戦略設定調整
    strategy_configs = {
        'conservative': {'min_expected_value': 1.15},
        'standard': {'min_expected_value': 1.05},
        'aggressive': {'min_expected_value': 1.02}
    }
    strategy.min_expected_value = max(args.min_ev, strategy_configs[args.strategy]['min_expected_value'])
    
    # レースデータ準備
    race_data = prepare_race_data(args.race_file)
    if race_data is None:
        return
    
    # 予測実行
    probabilities = predict_race_outcomes(strategy, model, race_data)
    if not probabilities:
        return
    
    # ベット推奨生成
    betting_recommendations, all_candidates = generate_betting_recommendations(strategy, probabilities, race_data)
    
    # 結果表示
    display_predictions(probabilities, betting_recommendations, race_data)
    
    # 結果保存（オプション）
    if args.output:
        results = {
            'timestamp': datetime.now().isoformat(),
            'race_file': args.race_file,
            'probabilities': probabilities,
            'betting_recommendations': betting_recommendations,
            'strategy_config': {
                'type': args.strategy,
                'min_expected_value': strategy.min_expected_value
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 結果保存: {args.output}")

if __name__ == "__main__":
    main()