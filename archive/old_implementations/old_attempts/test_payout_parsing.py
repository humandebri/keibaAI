#!/usr/bin/env python3
"""
払戻データのパース確認テスト
"""

import json
from src.strategies.advanced_betting import AdvancedBettingStrategy

# 実際の払戻データの例
sample_payout = {
    "win": {"5": 120}, 
    "place": {"52": 100240}, 
    "quinella": {}, 
    "exacta": {"5 → 2": 610}, 
    "wide": {"2 - 53 - 52 - 3": 190150440}, 
    "trio": {"2 - 3 - 5": 620}, 
    "trifecta": {"5 → 2 → 3": 1560}
}

def test_odds_parsing():
    """オッズパースのテスト"""
    strategy = AdvancedBettingStrategy()
    
    print("=== 払戻データパースのテスト ===")
    print(f"サンプルデータ: {json.dumps(sample_payout, ensure_ascii=False, indent=2)}")
    
    # 各馬券種のテスト
    test_cases = [
        # (bet_type, selection, expected_key)
        ('trifecta', (5, 2, 3), '5 → 2 → 3'),
        ('quinella', (2, 3), None),  # 空
        ('wide', (2, 3), '2 - 53 - 52 - 3'),  # 複合データ
        ('wide', (5, 3), '2 - 53 - 52 - 3'),  # 複合データ
        ('wide', (2, 5), '2 - 53 - 52 - 3'),  # 複合データ（存在しない組み合わせ）
    ]
    
    print("\n結果:")
    for bet_type, selection, expected_key in test_cases:
        odds = strategy._get_actual_odds(sample_payout, bet_type, selection)
        print(f"\n{bet_type} {selection}:")
        print(f"  期待するキー: {expected_key}")
        print(f"  取得したオッズ: {odds}")
        
        # ワイドの場合、複合データから正しく抽出できているか確認
        if bet_type == 'wide' and odds:
            print(f"  (元のデータ: {sample_payout['wide'].get(expected_key, 'N/A')})")


def test_wide_payout_extraction():
    """ワイド払戻の抽出テスト"""
    strategy = AdvancedBettingStrategy()
    
    print("\n\n=== ワイド払戻の抽出テスト ===")
    
    # 複合払戻データ
    test_data = [
        ("2 - 53 - 52 - 3", 190150440, ['2', '53', '52', '3']),
        ("1 - 2 - 3", 100200300, ['1', '2', '3']),
    ]
    
    for key, payout, horses in test_data:
        result = strategy._extract_wide_payout(payout, horses)
        print(f"\nキー: {key}")
        print(f"払戻: {payout}")
        print(f"馬番リスト: {horses}")
        print(f"抽出結果: {result}")
        
        # 組み合わせ数を確認
        from itertools import combinations
        pairs = list(combinations(horses, 2))
        print(f"組み合わせ数: {len(pairs)} → {pairs}")


def test_backtest_flow():
    """実際のバックテストフローのテスト"""
    strategy = AdvancedBettingStrategy(
        min_expected_value=1.0,
        use_actual_odds=True
    )
    
    print("\n\n=== バックテストフローのテスト ===")
    
    # テスト用の予測確率
    probs = {
        5: {'win_prob': 0.3, 'place_prob': 0.6, 'show_prob': 0.5, 'popularity': 1, 'predicted_rank': 1.2},
        2: {'win_prob': 0.2, 'place_prob': 0.5, 'show_prob': 0.4, 'popularity': 2, 'predicted_rank': 2.1},
        3: {'win_prob': 0.15, 'place_prob': 0.4, 'show_prob': 0.3, 'popularity': 3, 'predicted_rank': 3.0},
    }
    
    # 期待値計算のテスト
    print("\n1. 三連単 5-2-3:")
    actual_odds = sample_payout['trifecta'].get('5 → 2 → 3')
    ev, wp, odds = strategy.calculate_trifecta_ev(probs, 5, 2, 3, actual_odds)
    print(f"  実際のオッズ: {actual_odds}円 → {actual_odds/100:.1f}倍")
    print(f"  的中確率: {wp:.4f}")
    print(f"  期待値: {ev:.2f}")
    
    print("\n2. ワイド 2-3:")
    # ワイドの実際のオッズ取得
    actual_odds = strategy._get_actual_odds(sample_payout, 'wide', (2, 3))
    ev, wp, odds = strategy.calculate_wide_ev(probs, 2, 3, actual_odds)
    print(f"  実際のオッズ: {actual_odds}円 → {actual_odds/100:.1f}倍" if actual_odds else "  実際のオッズ: なし")
    print(f"  的中確率: {wp:.4f}")
    print(f"  期待値: {ev:.2f}")


if __name__ == "__main__":
    test_odds_parsing()
    test_wide_payout_extraction()
    test_backtest_flow()