#!/usr/bin/env python3
"""
リアルなデモオッズ生成システム
実際にスクレイピングした基本データに、騎手・厩舎の実績に基づく
リアルなオッズを生成してデモンストレーションを実行
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List


class RealisticOddsGenerator:
    """騎手・厩舎データに基づくリアルなオッズ生成"""
    
    # 実際の騎手実績データ（勝率を基にしたランキング）
    JOCKEY_RANKINGS = {
        'ルメール': 1,     # 最強騎手
        '川田': 2,
        'Ｍデムーロ': 3,
        '武豊': 4,
        '戸崎圭': 5,
        '岩田望': 6,
        '田辺': 7,
        '横山典': 8,
        '松山': 9,
        '北村友': 10,
        '北村宏': 11,
        '佐々木': 12,
        '坂井': 13,
        '池添': 14,
        '浜中': 15,
        '津村': 16,
        '丹内': 17,
    }
    
    # 厩舎実績データ
    TRAINER_RANKINGS = {
        '友道': 1,      # 最強厩舎
        '池江': 2,
        '杉山晴': 3,
        '矢作': 4,
        '中内田': 5,
        '高柳大': 6,
        '奥村武': 7,
        '西村': 8,
        '手塚久': 9,
        '斉藤崇': 10,
        '武幸': 11,
        '堀': 12,
        '藤野': 13,
        '昆': 14,
        '辻': 15,
        '笹田': 16,
        '千葉': 17,
    }
    
    def __init__(self):
        self.base_odds_range = (1.5, 50.0)
        
    def generate_realistic_odds(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """リアルな競馬オッズを生成"""
        print("🎲 リアルなオッズ生成中...")
        
        # 各馬の基礎評価点を計算
        evaluation_scores = []
        
        for _, horse in race_data.iterrows():
            score = self._calculate_horse_score(horse)
            evaluation_scores.append({
                'race_id': horse['race_id'],
                '枠': horse['枠'],
                '馬番': horse['馬番'],
                '馬名': horse['馬名'],
                '性齢': horse['性齢'],
                '騎手': horse['騎手'],
                '厩舎': horse['厩舎'],
                '斤量': horse['斤量'],
                '馬体重': horse['馬体重'],
                'score': score
            })
        
        # DataFrameに変換してソート
        df = pd.DataFrame(evaluation_scores)
        df = df.sort_values('score', ascending=False)
        
        # 人気順を決定
        df['人気'] = range(1, len(df) + 1)
        
        # オッズを生成（人気に基づく）
        df['オッズ'] = df['人気'].apply(self._popularity_to_odds)
        
        # 元の馬番順にソート
        df = df.sort_values('馬番')
        
        # 統計情報を表示
        print(f"✅ オッズ生成完了: 1番人気 {df[df['人気']==1]['オッズ'].iloc[0]:.1f}倍")
        print(f"   最低オッズ: {df['オッズ'].min():.1f}倍, 最高オッズ: {df['オッズ'].max():.1f}倍")
        
        return df
    
    def _calculate_horse_score(self, horse: Dict) -> float:
        """馬の総合評価スコアを計算"""
        score = 100.0  # ベーススコア
        
        # 騎手評価（最重要要素）
        jockey = horse['騎手']
        if jockey in self.JOCKEY_RANKINGS:
            jockey_bonus = (18 - self.JOCKEY_RANKINGS[jockey]) * 5
            score += jockey_bonus
        else:
            score += random.uniform(-10, 5)  # 未知騎手はややマイナス
        
        # 厩舎評価
        trainer = horse['厩舎'] 
        if trainer in self.TRAINER_RANKINGS:
            trainer_bonus = (18 - self.TRAINER_RANKINGS[trainer]) * 3
            score += trainer_bonus
        else:
            score += random.uniform(-5, 3)
        
        # 斤量評価（軽いほど有利）
        kinryo = horse['斤量']
        if kinryo <= 54.0:
            score += 15
        elif kinryo <= 56.0:
            score += 5
        elif kinryo >= 58.0:
            score -= 8
        
        # 枠番評価（中枠が有利傾向）
        waku = horse['枠']
        if waku in [3, 4, 5]:
            score += 8
        elif waku in [2, 6]:
            score += 3
        elif waku in [1, 8]:
            score -= 5
        
        # 馬体重評価
        weight_str = horse['馬体重']
        if isinstance(weight_str, str) and '(' in weight_str:
            try:
                weight = int(weight_str.split('(')[0])
                change_str = weight_str.split('(')[1].replace(')', '')
                change = int(change_str)
                
                # 理想体重範囲
                if 460 <= weight <= 500:
                    score += 10
                elif weight < 440 or weight > 520:
                    score -= 10
                
                # 体重変化
                if -2 <= change <= 4:
                    score += 5  # 微増は好調
                elif change >= 8:
                    score -= 8  # 大幅増は不調
                elif change <= -6:
                    score -= 5  # 大幅減も不安
                    
            except:
                pass
        
        # 年齢・性別評価
        sei_rei = horse['性齢']
        if isinstance(sei_rei, str):
            if '牡3' in sei_rei or '牝3' in sei_rei:
                score += 5  # 3歳は成長力
            elif '牡4' in sei_rei or '牝4' in sei_rei:
                score += 8  # 4歳は最盛期
            elif '牡5' in sei_rei or '牝5' in sei_rei:
                score += 3  # 5歳は経験豊富
            elif '牡6' in sei_rei or '牝6' in sei_rei or '牡7' in sei_rei:
                score -= 5  # 高齢は衰え
        
        # ランダム要素（競馬の不確実性）
        score += random.uniform(-15, 15)
        
        return max(score, 10)  # 最低スコア保証
    
    def _popularity_to_odds(self, popularity: int) -> float:
        """人気順からリアルなオッズを生成"""
        # 実際の競馬オッズ分布に近い計算
        base_odds_map = {
            1: random.uniform(1.8, 4.5),   # 1番人気: 1.8-4.5倍
            2: random.uniform(3.2, 7.8),   # 2番人気: 3.2-7.8倍  
            3: random.uniform(5.5, 12.0),  # 3番人気: 5.5-12.0倍
            4: random.uniform(8.0, 18.0),  # 4番人気: 8.0-18.0倍
            5: random.uniform(12.0, 25.0), # 5番人気: 12.0-25.0倍
        }
        
        if popularity in base_odds_map:
            return round(base_odds_map[popularity], 1)
        elif popularity <= 8:
            return round(random.uniform(15.0, 40.0), 1)
        elif popularity <= 12:
            return round(random.uniform(25.0, 80.0), 1)
        else:
            return round(random.uniform(50.0, 150.0), 1)


def main():
    """デモ実行"""
    print("🏇 リアルなオッズ生成デモシステム")
    print("="*50)
    
    # スクレイピング済みデータを読み込み
    try:
        race_data = pd.read_csv('/Users/0xhude/Desktop/Keiba_AI/corrected_race_data.csv')
        print(f"✅ レースデータ読み込み成功: {len(race_data)}頭")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # オッズ生成器を初期化
    generator = RealisticOddsGenerator()
    
    # リアルなオッズを生成
    complete_data = generator.generate_realistic_odds(race_data)
    
    # 結果を保存
    output_file = '/Users/0xhude/Desktop/Keiba_AI/demo_race_with_realistic_odds.csv'
    complete_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 完全データ保存: {output_file}")
    
    # 結果を表示
    print(f"\n📊 完全レースデータ: {len(complete_data)}頭")
    print("\n🏇 人気順出馬表:")
    display_data = complete_data.sort_values('人気')
    
    for _, horse in display_data.iterrows():
        print(f"  {horse['人気']:2d}人気 {horse['枠']}枠{horse['馬番']:2d}番 "
              f"{horse['馬名']:15s} {horse['騎手']:8s} {horse['厩舎']:8s} "
              f"{horse['馬体重']:10s} {horse['オッズ']:5.1f}倍")
    
    # 統計情報
    print(f"\n📈 オッズ統計:")
    print(f"   平均オッズ: {complete_data['オッズ'].mean():.1f}倍")
    print(f"   最低オッズ: {complete_data['オッズ'].min():.1f}倍 ({complete_data[complete_data['オッズ']==complete_data['オッズ'].min()]['馬名'].iloc[0]})")
    print(f"   最高オッズ: {complete_data['オッズ'].max():.1f}倍 ({complete_data[complete_data['オッズ']==complete_data['オッズ'].max()]['馬名'].iloc[0]})")
    
    print(f"\n✅ リアルなデモデータ生成完了！")
    print(f"このデータで予想システムのテストが可能です。")


if __name__ == "__main__":
    main()