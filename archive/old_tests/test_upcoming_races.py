#!/usr/bin/env python3
"""
今後のレース取得機能のテストスクリプト
"""

from jra_realtime_system import JRARealTimeSystem, NetkeibaDataCollector
from datetime import datetime, timedelta

def test_jra_upcoming_races():
    """JRAの今後のレース取得をテスト"""
    print("=" * 60)
    print("JRA 今後のレース取得テスト")
    print("=" * 60)
    
    jra_system = JRARealTimeSystem()
    
    # 本日のレース
    print("\n【本日のレース】")
    today_races = jra_system.get_today_races()
    print(f"取得件数: {len(today_races)}件")
    for i, race in enumerate(today_races[:3], 1):
        print(f"{i}. {race.get('time', 'N/A')} {race.get('racecourse', 'N/A')} "
              f"{race.get('race_number', 'N/A')}R {race.get('race_name', 'N/A')}")
    
    # 明日から3日間のレース
    print("\n【明日から3日間のレース】")
    upcoming_races = jra_system.get_upcoming_races(days_ahead=1, max_days=3)
    print(f"取得件数: {len(upcoming_races)}件")
    
    # 日付ごとに表示
    if upcoming_races:
        from itertools import groupby
        from operator import itemgetter
        
        grouped = groupby(upcoming_races, key=itemgetter('date'))
        for date, races in grouped:
            races_list = list(races)
            print(f"\n{date}: {len(races_list)}レース")
            for race in races_list[:2]:
                print(f"  - {race.get('time', 'N/A')} {race.get('racecourse', 'N/A')} "
                      f"{race.get('race_number', 'N/A')}R {race.get('race_name', 'N/A')}")

def test_netkeiba_upcoming_races():
    """netkeibaの今後のレース取得をテスト"""
    print("\n" + "=" * 60)
    print("netkeiba 今後のレース取得テスト")
    print("=" * 60)
    
    netkeiba = NetkeibaDataCollector()
    
    # 明日から2日間のレース
    print("\n【明日から2日間のレース】")
    upcoming_races = netkeiba.get_upcoming_race_list(days_ahead=1, max_days=2)
    print(f"取得件数: {len(upcoming_races)}件")
    
    for i, race in enumerate(upcoming_races[:5], 1):
        print(f"{i}. {race.get('date', 'N/A')} ID:{race.get('race_id', 'N/A')} "
              f"{race.get('race_name', 'N/A')}")

def main():
    """メイン処理"""
    print("今後のレース取得機能テスト")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        test_jra_upcoming_races()
    except Exception as e:
        print(f"\n❌ JRAテストエラー: {e}")
    
    try:
        test_netkeiba_upcoming_races()
    except Exception as e:
        print(f"\n❌ netkeibaテストエラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    main()