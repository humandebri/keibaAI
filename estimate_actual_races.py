#!/usr/bin/env python3
"""
実際のレース開催数を推定
"""

def estimate_races():
    """年間のレース数を推定"""
    
    print("=== 中央競馬の開催パターン ===")
    
    # 各競馬場の年間開催日数（概算）
    venue_days = {
        "札幌": 12,    # 夏季のみ
        "函館": 12,    # 夏季のみ
        "福島": 24,    # 春・秋
        "新潟": 20,    # 春・夏・秋
        "東京": 40,    # 主要開催
        "中山": 40,    # 主要開催
        "中京": 30,    # 通年
        "京都": 32,    # 春・秋
        "阪神": 36,    # 通年
        "小倉": 24     # 夏・冬
    }
    
    total_days = sum(venue_days.values())
    races_per_day = 12
    total_races = total_days * races_per_day
    
    print(f"年間開催日数（全場合計）: 約{total_days}日")
    print(f"1日あたりレース数: {races_per_day}レース")
    print(f"年間総レース数: 約{total_races:,}レース")
    
    print("\n各競馬場の開催日数:")
    for venue, days in venue_days.items():
        print(f"  {venue}: 約{days}日 → 約{days * races_per_day}レース")
    
    print(f"\n全組み合わせ: 10,920通り")
    print(f"実際のレース: 約{total_races:,}レース（{total_races/10920*100:.1f}%）")
    print(f"空振り率: 約{100 - total_races/10920*100:.1f}%")
    
    print("\n→ 「No data found」が大量に出るのは正常です！")

if __name__ == "__main__":
    estimate_races()