#!/usr/bin/env python3
"""
詳細なバックテスト分析を実行し、期待値計算の仕組みを説明する簡易スクリプト
"""

from pathlib import Path
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


class DetailedBacktestAnalyzer:
    """詳細なバックテスト分析クラス"""
    
    def __init__(self, output_dir: str = 'detailed_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_expectation_values(self):
        """期待値計算の詳細分析"""
        
        # サンプルデータを使って期待値計算を説明
        print("期待値計算の仕組みを分析中...")
        
        # 期待値の計算式を説明するレポートを作成
        report = []
        report.append("=" * 80)
        report.append("期待値計算の仕組み")
        report.append("=" * 80)
        report.append("")
        
        report.append("【期待値とは】")
        report.append("期待値 = 的中確率 × オッズ")
        report.append("期待値が1.0を超える場合、理論上は長期的にプラスになる賭けです。")
        report.append("")
        
        report.append("【確率計算の流れ】")
        report.append("1. LightGBMモデルが各馬の着順を予測")
        report.append("2. 予測着順から各種確率を計算：")
        report.append("   - 単勝確率（1着になる確率）")
        report.append("   - 複勝確率（3着以内に入る確率）")
        report.append("   - 2着確率（馬連用）")
        report.append("")
        
        report.append("【三連単の期待値計算】")
        report.append("確率 = P(馬1が1着) × P(馬2が2着|馬1が1着) × P(馬3が3着|馬1,2が1,2着)")
        report.append("オッズ = 実際のオッズ（データがある場合）または人気順位から推定")
        report.append("")
        
        report.append("【馬連の期待値計算】")
        report.append("確率 = P(馬1が1着かつ馬2が2着) + P(馬2が1着かつ馬1が2着)")
        report.append("")
        
        report.append("【ワイドの期待値計算】")
        report.append("確率 = P(馬1と馬2が両方3着以内)")
        report.append("")
        
        # ファイルに保存
        with open(self.output_dir / 'expectation_value_explanation.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"期待値説明を保存: {self.output_dir / 'expectation_value_explanation.txt'}")
        
    def analyze_betting_patterns(self, backtest_dir: str = 'backtest_kelly25'):
        """賭けパターンの分析"""
        
        # 保存されたバックテスト結果から読み込み
        trades_file = Path(backtest_dir) / 'backtest_trades.csv'
        if not trades_file.exists():
            print(f"賭けデータが見つかりません: {trades_file}")
            return
            
        # CSVファイルから読み込み
        import csv
        trades_data = []
        with open(trades_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades_data.append({
                    'race_id': row['race_id'],
                    'amount': abs(float(row['amount'])),
                    'profit': float(row['profit']),
                    'capital': float(row['capital']),
                    'is_win': row['is_win'] == 'True'
                })
        
        if not trades_data:
            print("賭けデータが空です")
            return
            
        # 賭けデータの要約をテキストで出力
        total_bets = len(trades_data)
        total_wins = sum(1 for t in trades_data if t['is_win'])
        win_rate = total_wins / total_bets * 100 if total_bets > 0 else 0
        total_amount = sum(t['amount'] for t in trades_data)
        avg_amount = total_amount / total_bets if total_bets > 0 else 0
        
        # 資金に対する賭け割合を計算
        bet_ratios = [t['amount'] / t['capital'] * 100 for t in trades_data]
        avg_bet_ratio = sum(bet_ratios) / len(bet_ratios) if bet_ratios else 0
        max_bet_ratio = max(bet_ratios) if bet_ratios else 0
        
        # 賭け種別ごとの統計を作成
        # betデータから種別を抽出
        bet_type_stats = {}
        
        # bet列から賭け種別を抽出するために、backtest_results.jsonを読み込み
        results_file = Path(backtest_dir) / 'backtest_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
                if 'metrics' in results_data and 'by_type' in results_data['metrics']:
                    bet_type_stats = results_data['metrics']['by_type']
        
        # 詳細レポートの作成
        report = []
        report.append("=" * 80)
        report.append("賭けパターンの詳細分析")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"総賭け回数: {total_bets:,}回")
        report.append(f"総賭け金額: ¥{total_amount:,.0f}")
        report.append(f"平均賭け金額: ¥{avg_amount:,.0f}")
        report.append(f"勝利数: {total_wins}回")
        report.append(f"勝率: {win_rate:.1f}%")
        report.append("")
        
        report.append("【資金管理】")
        report.append(f"平均賭け割合: {avg_bet_ratio:.2f}%")
        report.append(f"最大賭け割合: {max_bet_ratio:.2f}%")
        report.append(f"Kelly基準: 25%")
        report.append("")
        
        if bet_type_stats:
            report.append("【賭け種別ごとの分析】")
            for bet_type, stats in bet_type_stats.items():
                report.append(f"\n{bet_type}:")
                report.append(f"  賭け回数: {stats['count']:,}回")
                report.append(f"  勝利数: {stats['wins']}回")
                report.append(f"  勝率: {stats['win_rate']*100:.1f}%")
                report.append(f"  利益: ¥{stats['profit']:,.0f}")
                report.append(f"  ROI: {stats['roi']*100:.1f}%")
            
        with open(self.output_dir / 'betting_patterns_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"賭けパターン分析を保存: {self.output_dir / 'betting_patterns_report.txt'}")
        
    def analyze_kelly_criterion(self):
        """ケリー基準の説明と分析"""
        
        report = []
        report.append("=" * 80)
        report.append("ケリー基準による資金管理")
        report.append("=" * 80)
        report.append("")
        
        report.append("【ケリー基準とは】")
        report.append("長期的な資産成長を最大化する最適な賭け金額を計算する数学的公式")
        report.append("")
        
        report.append("【計算式】")
        report.append("賭け金額 = 資金 × ケリー割合 × エッジ")
        report.append("")
        report.append("エッジ = (期待値 - 1) / (オッズ - 1)")
        report.append("")
        
        report.append("【例】")
        report.append("- 資金: 100万円")
        report.append("- ケリー割合: 25% (0.25)")
        report.append("- 期待値: 1.5")
        report.append("- オッズ: 10倍")
        report.append("")
        report.append("エッジ = (1.5 - 1) / (10 - 1) = 0.5 / 9 = 0.056")
        report.append("賭け金額 = 1,000,000 × 0.25 × 0.056 = 14,000円")
        report.append("")
        
        report.append("【なぜ25%？】")
        report.append("- フルケリー（100%）は理論上最適だが、実践では高リスク")
        report.append("- 25%（1/4ケリー）は実用的なバランス")
        report.append("- リスクを抑えながら着実な成長を目指す")
        report.append("")
        
        report.append("【上限設定】")
        report.append("- 1回の賭けは資金の5%まで")
        report.append("- 過度な集中投資を防ぐ安全装置")
        
        with open(self.output_dir / 'kelly_criterion_explanation.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"ケリー基準説明を保存: {self.output_dir / 'kelly_criterion_explanation.txt'}")


def main():
    """メイン実行関数"""
    analyzer = DetailedBacktestAnalyzer()
    
    # 期待値計算の説明
    analyzer.analyze_expectation_values()
    
    # ケリー基準の説明
    analyzer.analyze_kelly_criterion()
    
    # 賭けパターンの分析
    analyzer.analyze_betting_patterns()
    
    print(f"\n分析結果を {analyzer.output_dir} に保存しました。")
    print("\n以下のファイルが生成されました：")
    print("- expectation_value_explanation.txt: 期待値計算の仕組み")
    print("- kelly_criterion_explanation.txt: ケリー基準の説明")
    print("- betting_patterns_report.txt: 賭けパターンの詳細レポート")


if __name__ == "__main__":
    main()