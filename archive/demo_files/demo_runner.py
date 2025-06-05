#!/usr/bin/env python3
"""
統合競馬予測システム デモランナー
安全にシステムをテストするためのスクリプト
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import json
import sys

from integrated_betting_system import IntegratedKeibaSystem
from keiba_ai_improved_system_fixed import ImprovedKeibaAISystem


class DemoRunner:
    """デモ実行管理クラス"""
    
    def __init__(self, mode='paper'):
        self.mode = mode  # 'paper' or 'live'
        self.config = self._get_demo_config()
        self.results = []
        
    def _get_demo_config(self):
        """デモ用設定"""
        if self.mode == 'paper':
            # ペーパートレードモード
            return {
                'model_path': 'models/improved_model.pkl',
                'max_bet_per_race': 5000,
                'max_daily_loss': 30000,
                'min_expected_value': 1.2,
                'kelly_fraction': 0.025,
                'data_refresh_interval': 300,  # 5分
                'enable_auto_betting': False,  # 常にFalse
                'notification': {
                    'email': None,
                    'slack_webhook': None
                }
            }
        else:
            # ライブモード（手動確認必須）
            return {
                'model_path': 'models/improved_model.pkl',
                'max_bet_per_race': 10000,
                'max_daily_loss': 50000,
                'min_expected_value': 1.1,
                'kelly_fraction': 0.05,
                'data_refresh_interval': 300,
                'enable_auto_betting': False,  # 安全のため常にFalse
                'notification': {
                    'email': None,
                    'slack_webhook': None
                }
            }
    
    async def run_demo(self, duration_hours=1):
        """デモ実行"""
        print("=" * 60)
        print(f"🏇 競馬予測システム - {self.mode.upper()}モード")
        print("=" * 60)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"実行時間: {duration_hours}時間")
        print(f"モード: {'ペーパートレード' if self.mode == 'paper' else 'ライブ（手動確認）'}")
        print("\n設定:")
        print(f"  - 最大ベット/レース: ¥{self.config['max_bet_per_race']:,}")
        print(f"  - 最大日次損失: ¥{self.config['max_daily_loss']:,}")
        print(f"  - 最小期待値: {self.config['min_expected_value']}")
        print(f"  - Kelly係数: {self.config['kelly_fraction']*100}%")
        print("\n" + "=" * 60 + "\n")
        
        # システム初期化
        system = IntegratedKeibaSystem(self.config)
        
        # デモ用にログ記録を追加
        self._setup_demo_logging(system)
        
        # 時間制限付きで実行
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        
        try:
            while datetime.now().timestamp() < end_time:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] サイクル開始")
                
                # レース情報取得
                races = await system._get_today_races()
                
                if races:
                    print(f"📊 {len(races)}件のレースを検出")
                    
                    # 各レースを処理（最大3レース）
                    for i, race in enumerate(races[:3]):
                        print(f"\n  レース{i+1}: {race['race_id']}")
                        
                        # デモ用の処理
                        await self._demo_process_race(system, race)
                        
                        # レート制限
                        await asyncio.sleep(5)
                else:
                    print("⚠️ レースが見つかりません")
                
                # 統計表示
                self._show_demo_stats()
                
                # 次のサイクルまで待機
                wait_time = min(300, end_time - datetime.now().timestamp())
                if wait_time > 0:
                    print(f"\n💤 {int(wait_time)}秒待機中...")
                    await asyncio.sleep(wait_time)
                else:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️ ユーザーによる中断")
        except Exception as e:
            print(f"\n❌ エラー: {e}")
        finally:
            # 最終レポート
            self._generate_demo_report()
    
    async def _demo_process_race(self, system, race):
        """デモ用レース処理"""
        try:
            # 実際の処理をシミュレート
            print(f"    - データ取得中...")
            await asyncio.sleep(1)  # API呼び出しをシミュレート
            
            # ダミー予測結果
            import random
            if random.random() < 0.3:  # 30%の確率でベッティング機会
                opportunity = {
                    'race_id': race['race_id'],
                    'horse_number': random.randint(1, 12),
                    'horse_name': f'テスト馬{random.randint(1, 100)}',
                    'expected_value': round(random.uniform(1.1, 1.5), 2),
                    'suggested_bet': random.randint(1, 5) * 1000,
                    'timestamp': datetime.now()
                }
                
                print(f"    💡 ベッティング機会検出!")
                print(f"       馬番: {opportunity['horse_number']}")
                print(f"       期待値: {opportunity['expected_value']}")
                print(f"       推奨額: ¥{opportunity['suggested_bet']:,}")
                
                self.results.append(opportunity)
                
                if self.mode == 'paper':
                    print(f"    📝 ペーパートレード記録")
                else:
                    print(f"    ⚠️ 手動確認が必要です")
            else:
                print(f"    ✓ 期待値基準を満たしません")
                
        except Exception as e:
            print(f"    ❌ エラー: {e}")
    
    def _setup_demo_logging(self, system):
        """デモ用ログ設定"""
        # ログディレクトリ作成
        log_dir = Path(f"demo_logs/{self.mode}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイルパス設定
        self.log_file = log_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def _show_demo_stats(self):
        """デモ統計表示"""
        if self.results:
            print(f"\n📊 現在の統計:")
            print(f"  - 検出機会: {len(self.results)}件")
            
            total_suggested = sum(r['suggested_bet'] for r in self.results)
            avg_ev = sum(r['expected_value'] for r in self.results) / len(self.results)
            
            print(f"  - 推奨総額: ¥{total_suggested:,}")
            print(f"  - 平均期待値: {avg_ev:.3f}")
    
    def _generate_demo_report(self):
        """デモレポート生成"""
        print("\n" + "=" * 60)
        print("📄 デモ実行レポート")
        print("=" * 60)
        
        if self.results:
            print(f"\n検出されたベッティング機会: {len(self.results)}件")
            
            # 期待値でソート
            sorted_results = sorted(self.results, 
                                  key=lambda x: x['expected_value'], 
                                  reverse=True)
            
            print("\nトップ5機会:")
            for i, r in enumerate(sorted_results[:5], 1):
                print(f"{i}. {r['timestamp'].strftime('%H:%M')} - "
                      f"レース{r['race_id']} 馬番{r['horse_number']} "
                      f"EV={r['expected_value']} ¥{r['suggested_bet']:,}")
            
            # レポート保存
            report_data = {
                'mode': self.mode,
                'start_time': self.results[0]['timestamp'].isoformat() if self.results else None,
                'end_time': datetime.now().isoformat(),
                'total_opportunities': len(self.results),
                'opportunities': [
                    {
                        'timestamp': r['timestamp'].isoformat(),
                        'race_id': r['race_id'],
                        'horse_number': r['horse_number'],
                        'expected_value': r['expected_value'],
                        'suggested_bet': r['suggested_bet']
                    }
                    for r in sorted_results
                ]
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 レポートを保存しました: {self.log_file}")
        else:
            print("\n⚠️ ベッティング機会は検出されませんでした")


def quick_test():
    """クイックテスト（5分間）"""
    print("🚀 クイックテストモード（5分間）")
    print("=" * 60)
    
    runner = DemoRunner(mode='paper')
    asyncio.run(runner.run_demo(duration_hours=0.083))  # 5分


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description='競馬予測システム デモランナー'
    )
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'quick'],
        default='quick',
        help='実行モード (default: quick)'
    )
    parser.add_argument(
        '--hours',
        type=float,
        default=1.0,
        help='実行時間（時間） (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test()
    else:
        runner = DemoRunner(mode=args.mode)
        asyncio.run(runner.run_demo(duration_hours=args.hours))


if __name__ == "__main__":
    # 使用例を表示
    if len(sys.argv) == 1:
        print("🏇 競馬予測システム デモランナー")
        print("\n使用方法:")
        print("  1. クイックテスト（5分）:")
        print("     python demo_runner.py")
        print("\n  2. ペーパートレード（1時間）:")
        print("     python demo_runner.py --mode paper --hours 1")
        print("\n  3. ライブモード（手動確認、2時間）:")
        print("     python demo_runner.py --mode live --hours 2")
        print("\n" + "-" * 40)
        print("デフォルトでクイックテストを実行します...\n")
    
    main()