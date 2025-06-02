#!/usr/bin/env python3
"""
競馬予測システム モニタリングツール
システムの状態とパフォーマンスを監視
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import numpy as np


class SystemMonitor:
    """システムモニタリングクラス"""
    
    def __init__(self):
        self.logs_dir = Path("logs")
        self.demo_logs_dir = Path("demo_logs")
        self.betting_logs = Path("logs/betting_opportunities.json")
        self.daily_stats = Path("logs/daily_stats.json")
        
        # 日本語フォント設定（matplotlib用）
        self.font_prop = FontProperties(fname='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc')
        
    def get_latest_stats(self) -> Dict:
        """最新の統計情報を取得"""
        stats = {
            'current_time': datetime.now(),
            'system_status': self._check_system_status(),
            'today_stats': self._get_today_stats(),
            'recent_opportunities': self._get_recent_opportunities(),
            'system_health': self._check_system_health()
        }
        return stats
    
    def _check_system_status(self) -> str:
        """システム稼働状態をチェック"""
        # 最新のログファイルをチェック
        latest_log = None
        latest_time = 0
        
        for log_file in self.logs_dir.glob("jra_realtime_*.log"):
            if log_file.stat().st_mtime > latest_time:
                latest_time = log_file.stat().st_mtime
                latest_log = log_file
        
        if latest_log and (time.time() - latest_time) < 600:  # 10分以内
            return "🟢 稼働中"
        elif latest_log:
            return "🟡 停止中"
        else:
            return "🔴 未起動"
    
    def _get_today_stats(self) -> Dict:
        """本日の統計を取得"""
        if self.daily_stats.exists():
            with open(self.daily_stats, 'r') as f:
                all_stats = json.load(f)
                
            # 本日のデータを探す
            today = datetime.now().date().isoformat()
            for stat in reversed(all_stats):
                if stat.get('date') == today:
                    return stat
        
        # デフォルト値
        return {
            'date': datetime.now().date().isoformat(),
            'total_bets': 0,
            'total_amount': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0,
            'races_analyzed': 0
        }
    
    def _get_recent_opportunities(self, limit: int = 10) -> List[Dict]:
        """最近のベッティング機会を取得"""
        if self.betting_logs.exists():
            with open(self.betting_logs, 'r') as f:
                opportunities = json.load(f)
            
            # 最新のものから返す
            return opportunities[-limit:]
        
        return []
    
    def _check_system_health(self) -> Dict:
        """システムヘルスチェック"""
        health = {
            'cpu_usage': self._get_cpu_usage(),
            'memory_usage': self._get_memory_usage(),
            'disk_space': self._get_disk_space(),
            'api_status': self._check_api_status()
        }
        return health
    
    def _get_cpu_usage(self) -> float:
        """CPU使用率を取得（ダミー実装）"""
        import random
        return round(random.uniform(10, 30), 1)
    
    def _get_memory_usage(self) -> float:
        """メモリ使用率を取得（ダミー実装）"""
        import random
        return round(random.uniform(20, 40), 1)
    
    def _get_disk_space(self) -> float:
        """ディスク空き容量を取得"""
        import shutil
        stat = shutil.disk_usage("/")
        return round((stat.free / stat.total) * 100, 1)
    
    def _check_api_status(self) -> str:
        """API接続状態をチェック"""
        # 実際にはAPIをチェックする
        return "正常"
    
    def display_dashboard(self):
        """ダッシュボード表示"""
        stats = self.get_latest_stats()
        
        print("=" * 70)
        print(f"🏇 競馬予測システム モニタリングダッシュボード")
        print(f"   更新時刻: {stats['current_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # システム状態
        print(f"\n📊 システム状態: {stats['system_status']}")
        
        # 本日の統計
        today = stats['today_stats']
        print(f"\n📈 本日の統計 ({today['date']}):")
        print(f"  - 分析レース数: {today['races_analyzed']}件")
        print(f"  - ベット数: {today['total_bets']}件")
        print(f"  - ベット総額: ¥{today['total_amount']:,}")
        print(f"  - 勝利: {today['wins']}件 / 敗北: {today['losses']}件")
        print(f"  - 損益: ¥{today['profit_loss']:+,}")
        
        # システムヘルス
        health = stats['system_health']
        print(f"\n🔧 システムヘルス:")
        print(f"  - CPU使用率: {health['cpu_usage']}%")
        print(f"  - メモリ使用率: {health['memory_usage']}%")
        print(f"  - ディスク空き容量: {health['disk_space']}%")
        print(f"  - API状態: {health['api_status']}")
        
        # 最近のベッティング機会
        opportunities = stats['recent_opportunities']
        if opportunities:
            print(f"\n💡 最近のベッティング機会:")
            for i, opp in enumerate(opportunities[-5:], 1):
                print(f"  {i}. {opp['timestamp'][:16]} - "
                      f"レース{opp['race_id']} "
                      f"馬番{opp['horse_number']} "
                      f"EV={opp['expected_value']:.2f} "
                      f"¥{opp['suggested_bet']:,}")
        else:
            print(f"\n💡 最近のベッティング機会: なし")
        
        print("\n" + "=" * 70)
    
    def generate_performance_report(self, days: int = 7):
        """パフォーマンスレポート生成"""
        print(f"\n📊 過去{days}日間のパフォーマンスレポート")
        print("=" * 60)
        
        # 日次統計を読み込み
        if self.daily_stats.exists():
            with open(self.daily_stats, 'r') as f:
                all_stats = json.load(f)
            
            # 過去N日分のデータをフィルタ
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            recent_stats = [
                stat for stat in all_stats
                if datetime.fromisoformat(stat['date']).date() >= cutoff_date
            ]
            
            if recent_stats:
                # 集計
                total_bets = sum(s['total_bets'] for s in recent_stats)
                total_amount = sum(s['total_amount'] for s in recent_stats)
                total_wins = sum(s['wins'] for s in recent_stats)
                total_profit = sum(s['profit_loss'] for s in recent_stats)
                
                print(f"期間: {recent_stats[0]['date']} ～ {recent_stats[-1]['date']}")
                print(f"\n📈 サマリー:")
                print(f"  - 総ベット数: {total_bets}件")
                print(f"  - 総ベット額: ¥{total_amount:,}")
                print(f"  - 勝利数: {total_wins}件")
                print(f"  - 勝率: {(total_wins/total_bets*100) if total_bets > 0 else 0:.1f}%")
                print(f"  - 総損益: ¥{total_profit:+,}")
                print(f"  - ROI: {(total_profit/total_amount*100) if total_amount > 0 else 0:+.1f}%")
                
                # 日別詳細
                print(f"\n📅 日別詳細:")
                for stat in recent_stats:
                    roi = (stat['profit_loss']/stat['total_amount']*100) if stat['total_amount'] > 0 else 0
                    print(f"  {stat['date']}: "
                          f"ベット{stat['total_bets']}件 "
                          f"損益¥{stat['profit_loss']:+,} "
                          f"ROI{roi:+.1f}%")
            else:
                print("データがありません")
        else:
            print("統計ファイルが見つかりません")
        
        print("=" * 60)
    
    def plot_performance(self, days: int = 30):
        """パフォーマンスグラフ作成"""
        if not self.daily_stats.exists():
            print("グラフ作成用のデータがありません")
            return
        
        with open(self.daily_stats, 'r') as f:
            all_stats = json.load(f)
        
        # データをDataFrameに変換
        df = pd.DataFrame(all_stats)
        df['date'] = pd.to_datetime(df['date'])
        
        # 過去N日分をフィルタ
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
        
        if df.empty:
            print("グラフ作成用のデータが不足しています")
            return
        
        # 累積損益を計算
        df['cumulative_profit'] = df['profit_loss'].cumsum()
        
        # グラフ作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 累積損益グラフ
        ax1.plot(df['date'], df['cumulative_profit'], 'b-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('累積損益推移', fontproperties=self.font_prop)
        ax1.set_ylabel('損益 (円)', fontproperties=self.font_prop)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # 日別ベット数と勝率
        ax2.bar(df['date'], df['total_bets'], alpha=0.6, label='ベット数')
        
        # 勝率を計算して追加
        df['win_rate'] = df.apply(
            lambda row: (row['wins'] / row['total_bets'] * 100) if row['total_bets'] > 0 else 0,
            axis=1
        )
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['date'], df['win_rate'], 'r-', marker='o', label='勝率')
        
        ax2.set_title('日別ベット数と勝率', fontproperties=self.font_prop)
        ax2.set_ylabel('ベット数', fontproperties=self.font_prop)
        ax2_twin.set_ylabel('勝率 (%)', fontproperties=self.font_prop)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        output_path = Path("logs/performance_chart.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 パフォーマンスグラフを保存しました: {output_path}")
        
        # 表示（オプション）
        # plt.show()


def main():
    """メイン実行"""
    monitor = SystemMonitor()
    
    while True:
        # 画面クリア（OSに応じて）
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # ダッシュボード表示
        monitor.display_dashboard()
        
        # オプション表示
        print("\n📋 オプション:")
        print("  1. パフォーマンスレポート（7日間）")
        print("  2. パフォーマンスレポート（30日間）")
        print("  3. パフォーマンスグラフ生成")
        print("  4. 更新（30秒後に自動更新）")
        print("  0. 終了")
        
        # ユーザー入力待ち（タイムアウト付き）
        import select
        import sys
        
        print("\n選択 (0-4): ", end='', flush=True)
        
        # 30秒のタイムアウト
        i, o, e = select.select([sys.stdin], [], [], 30)
        
        if i:
            choice = sys.stdin.readline().strip()
            
            if choice == '0':
                print("モニタリングを終了します")
                break
            elif choice == '1':
                monitor.generate_performance_report(days=7)
                input("\nEnterキーで続行...")
            elif choice == '2':
                monitor.generate_performance_report(days=30)
                input("\nEnterキーで続行...")
            elif choice == '3':
                monitor.plot_performance(days=30)
                input("\nEnterキーで続行...")
        else:
            # タイムアウト - 自動更新
            continue


if __name__ == "__main__":
    print("🏇 競馬予測システム モニター")
    print("Ctrl+C で終了します\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nモニタリングを終了しました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()