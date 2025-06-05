#!/usr/bin/env python3
"""
セグメント別モデル評価システム
芝/ダート、距離別の専門モデルと評価
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SegmentModelEvaluator:
    """セグメント別モデルの詳細評価"""
    
    def __init__(self):
        self.segment_definitions = {
            'track_distance': {
                '芝_sprint': {'track': '芝', 'distance': (0, 1400)},
                '芝_mile': {'track': '芝', 'distance': (1400, 1800)},
                '芝_intermediate': {'track': '芝', 'distance': (1800, 2200)},
                '芝_long': {'track': '芝', 'distance': (2200, 3600)},
                'ダート_sprint': {'track': 'ダート', 'distance': (0, 1400)},
                'ダート_mile': {'track': 'ダート', 'distance': (1400, 1800)},
                'ダート_intermediate': {'track': 'ダート', 'distance': (1800, 2200)},
                'ダート_long': {'track': 'ダート', 'distance': (2200, 3600)}
            },
            'field_size': {
                'small_field': {'field_size': (5, 12)},
                'medium_field': {'field_size': (13, 16)},
                'large_field': {'field_size': (17, 18)}
            },
            'track_condition': {
                '良馬場': {'condition': ['良']},
                '稍重_重': {'condition': ['稍', '稍重', '重']},
                '不良': {'condition': ['不', '不良']}
            },
            'venue_type': {
                '中央場': {'venue': ['東京', '中山', '阪神', '京都', '中京']},
                'ローカル': {'venue': ['新潟', '札幌', '函館', '福島', '小倉']}
            }
        }
        
        self.evaluation_results = {}
    
    def evaluate_segments(self, data: pd.DataFrame, model_dict: Dict, 
                         feature_cols: List[str]) -> Dict:
        """全セグメントの評価"""
        results = {}
        
        for segment_type, segments in self.segment_definitions.items():
            print(f"\n{segment_type}セグメントの評価中...")
            results[segment_type] = {}
            
            for segment_name, criteria in segments.items():
                segment_data = self._filter_segment_data(data, criteria)
                
                if len(segment_data) < 100:
                    print(f"  {segment_name}: データ不足でスキップ")
                    continue
                
                # 各モデルで評価
                segment_results = self._evaluate_single_segment(
                    segment_data, model_dict, feature_cols, segment_name
                )
                
                results[segment_type][segment_name] = segment_results
                
                # サマリー表示
                print(f"  {segment_name}:")
                print(f"    データ数: {len(segment_data)}")
                print(f"    最良モデル順位相関: {segment_results['best_correlation']:.3f}")
                print(f"    ROI@20%: {segment_results['roi_at_20pct']:.3f}")
        
        return results
    
    def _filter_segment_data(self, data: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
        """セグメント条件に基づいてデータをフィルタ"""
        filtered = data.copy()
        
        if 'track' in criteria:
            filtered = filtered[filtered['芝・ダート'].str.contains(criteria['track'])]
        
        if 'distance' in criteria:
            min_dist, max_dist = criteria['distance']
            filtered = filtered[(filtered['距離'] >= min_dist) & (filtered['距離'] < max_dist)]
        
        if 'field_size' in criteria:
            min_size, max_size = criteria['field_size']
            filtered = filtered[(filtered['出走頭数'] >= min_size) & (filtered['出走頭数'] <= max_size)]
        
        if 'condition' in criteria:
            filtered = filtered[filtered['馬場'].isin(criteria['condition'])]
        
        if 'venue' in criteria:
            filtered = filtered[filtered['場名'].isin(criteria['venue'])]
        
        return filtered
    
    def _evaluate_single_segment(self, segment_data: pd.DataFrame, 
                               model_dict: Dict, feature_cols: List[str],
                               segment_name: str) -> Dict:
        """単一セグメントの詳細評価"""
        results = {
            'data_size': len(segment_data),
            'model_correlations': {},
            'calibration_scores': {},
            'best_correlation': 0,
            'best_model': None
        }
        
        # データ準備
        X = segment_data[feature_cols].fillna(0).values
        y = segment_data['着順_numeric'].values
        
        # 各モデルで評価
        for model_name, model in model_dict.items():
            if model is None:
                continue
            
            # 予測
            if 'lgb' in model_name:
                predictions = model.predict(X, num_iteration=model.best_iteration)
            else:
                predictions = model.predict(X)
            
            # 順位相関
            pred_ranks = pd.Series(predictions).rank()
            true_ranks = pd.Series(y).rank()
            correlation = pred_ranks.corr(true_ranks, method='spearman')
            
            results['model_correlations'][model_name] = correlation
            
            if correlation > results['best_correlation']:
                results['best_correlation'] = correlation
                results['best_model'] = model_name
            
            # Calibration評価（勝率予測の場合）
            if segment_data['着順_numeric'].min() == 1:  # 勝ちデータがある場合
                win_probs = self._convert_to_win_probability(predictions, segment_data)
                actual_wins = (y == 1).astype(int)
                
                if len(np.unique(actual_wins)) > 1:  # 勝ちと負け両方ある
                    calibration_score = self._evaluate_calibration(actual_wins, win_probs)
                    results['calibration_scores'][model_name] = calibration_score
        
        # ROIシミュレーション（簡易版）
        if results['best_model']:
            results['roi_at_20pct'] = self._calculate_segment_roi(
                segment_data, predictions, threshold_pct=0.2
            )
        
        return results
    
    def _convert_to_win_probability(self, predictions: np.ndarray, 
                                  race_data: pd.DataFrame) -> np.ndarray:
        """予測値を勝率に変換"""
        # レースごとに正規化
        win_probs = []
        
        for race_id in race_data['race_id'].unique():
            race_mask = race_data['race_id'] == race_id
            race_preds = predictions[race_mask]
            
            # ソフトマックス変換
            exp_neg_preds = np.exp(-race_preds * 2)
            race_probs = exp_neg_preds / exp_neg_preds.sum()
            
            win_probs.extend(race_probs)
        
        return np.array(win_probs)
    
    def _evaluate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calibration評価"""
        # Reliability diagram用のデータ
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='uniform'
        )
        
        # Brier Score
        brier = brier_score_loss(y_true, y_pred_proba)
        
        # Expected Calibration Error (ECE)
        ece = np.average(
            np.abs(fraction_of_positives - mean_predicted_value),
            weights=np.histogram(y_pred_proba, bins=10)[0]
        )
        
        # Log Loss
        ll = log_loss(y_true, y_pred_proba)
        
        return {
            'brier_score': brier,
            'ece': ece,
            'log_loss': ll,
            'reliability_data': {
                'fraction_positives': fraction_of_positives.tolist(),
                'mean_predicted': mean_predicted_value.tolist()
            }
        }
    
    def _calculate_segment_roi(self, segment_data: pd.DataFrame, 
                             predictions: np.ndarray, threshold_pct: float) -> float:
        """セグメント別ROI計算"""
        total_bet = 0
        total_return = 0
        
        unique_races = segment_data['race_id'].unique()
        
        for race_id in unique_races[:100]:  # 最初の100レース
            race_mask = segment_data['race_id'] == race_id
            race_data = segment_data[race_mask]
            race_preds = predictions[race_mask]
            
            # 上位X%を選択
            n_horses = len(race_data)
            n_select = max(1, int(n_horses * threshold_pct))
            
            # 予測上位馬
            top_indices = np.argpartition(race_preds, n_select)[:n_select]
            
            for idx in top_indices:
                horse = race_data.iloc[idx]
                
                # 単勝100円賭け
                total_bet += 100
                
                if horse['着順_numeric'] == 1:
                    # 的中
                    odds = horse.get('オッズ_numeric', 10)
                    total_return += 100 * odds
        
        return total_return / total_bet if total_bet > 0 else 0
    
    def create_segment_report(self, evaluation_results: Dict, output_dir: str = 'results_improved') -> None:
        """セグメント評価レポートの作成"""
        # 可視化
        self._create_segment_visualizations(evaluation_results, output_dir)
        
        # テキストレポート
        self._create_segment_text_report(evaluation_results, output_dir)
        
        # 最適セグメント戦略の提案
        self._suggest_optimal_segments(evaluation_results, output_dir)
    
    def _create_segment_visualizations(self, results: Dict, output_dir: str) -> None:
        """セグメント結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Segment Model Performance Analysis', fontsize=16)
        
        # 1. Track×Distance の相関ヒートマップ
        ax = axes[0, 0]
        if 'track_distance' in results:
            segment_names = []
            correlations = []
            
            for segment, data in results['track_distance'].items():
                segment_names.append(segment.replace('_', '\n'))
                correlations.append(data.get('best_correlation', 0))
            
            # ヒートマップ用にreshape（仮に4x2のグリッド）
            corr_matrix = np.array(correlations).reshape(-1, 1)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', 
                       yticklabels=segment_names,
                       xticklabels=['Correlation'],
                       cmap='RdYlGn', center=0.7,
                       ax=ax)
            ax.set_title('Track×Distance Segment Correlations')
        
        # 2. ROI by Segment
        ax = axes[0, 1]
        segment_rois = {}
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                roi = data.get('roi_at_20pct', 0)
                segment_rois[f"{seg_type}\n{seg_name}"] = roi
        
        if segment_rois:
            sorted_segments = sorted(segment_rois.items(), key=lambda x: x[1], reverse=True)[:10]
            names, rois = zip(*sorted_segments)
            
            colors = ['green' if r > 1.0 else 'red' for r in rois]
            ax.barh(names, rois, color=colors)
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('ROI')
            ax.set_title('Top 10 Segments by ROI')
        
        # 3. Calibration plots
        ax = axes[1, 0]
        # 最良セグメントのキャリブレーションプロット
        best_segment = None
        best_calibration = None
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                if 'calibration_scores' in data and data['calibration_scores']:
                    for model, cal_data in data['calibration_scores'].items():
                        if cal_data and 'reliability_data' in cal_data:
                            best_segment = seg_name
                            best_calibration = cal_data['reliability_data']
                            break
                    if best_calibration:
                        break
            if best_calibration:
                break
        
        if best_calibration:
            frac_pos = best_calibration['fraction_positives']
            mean_pred = best_calibration['mean_predicted']
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.plot(mean_pred, frac_pos, 'ro-', label=f'{best_segment}')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot (Best Segment)')
            ax.legend()
        
        # 4. セグメントサイズ vs パフォーマンス
        ax = axes[1, 1]
        sizes = []
        correlations = []
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                if 'data_size' in data and 'best_correlation' in data:
                    sizes.append(data['data_size'])
                    correlations.append(data['best_correlation'])
        
        if sizes and correlations:
            ax.scatter(sizes, correlations, alpha=0.6, s=100)
            ax.set_xlabel('Segment Size (number of races)')
            ax.set_ylabel('Best Model Correlation')
            ax.set_title('Segment Size vs Performance')
            ax.set_xscale('log')
            
            # トレンドライン
            if len(sizes) > 3:
                z = np.polyfit(np.log(sizes), correlations, 1)
                p = np.poly1d(z)
                x_trend = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
                ax.plot(x_trend, p(np.log(x_trend)), 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/segment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_segment_text_report(self, results: Dict, output_dir: str) -> None:
        """セグメント評価のテキストレポート"""
        report = []
        report.append("="*60)
        report.append("SEGMENT MODEL EVALUATION REPORT")
        report.append("="*60)
        
        # Best performing segments
        report.append("\n[Top Performing Segments]")
        all_segments = []
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                if 'best_correlation' in data:
                    all_segments.append({
                        'type': seg_type,
                        'name': seg_name,
                        'correlation': data['best_correlation'],
                        'roi': data.get('roi_at_20pct', 0),
                        'size': data.get('data_size', 0)
                    })
        
        # Sort by correlation
        all_segments.sort(key=lambda x: x['correlation'], reverse=True)
        
        for i, seg in enumerate(all_segments[:10], 1):
            report.append(f"{i}. {seg['name']} ({seg['type']})")
            report.append(f"   Correlation: {seg['correlation']:.3f}")
            report.append(f"   ROI@20%: {seg['roi']:.3f}")
            report.append(f"   Data size: {seg['size']}")
        
        # Calibration summary
        report.append("\n[Calibration Performance]")
        best_calibrated = []
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                if 'calibration_scores' in data:
                    for model, cal_scores in data['calibration_scores'].items():
                        if cal_scores and 'brier_score' in cal_scores:
                            best_calibrated.append({
                                'segment': seg_name,
                                'model': model,
                                'brier': cal_scores['brier_score'],
                                'ece': cal_scores['ece']
                            })
        
        if best_calibrated:
            best_calibrated.sort(key=lambda x: x['brier'])
            
            for i, cal in enumerate(best_calibrated[:5], 1):
                report.append(f"{i}. {cal['segment']} - {cal['model']}")
                report.append(f"   Brier Score: {cal['brier']:.4f}")
                report.append(f"   ECE: {cal['ece']:.4f}")
        
        # Segment-specific insights
        report.append("\n[Segment-Specific Insights]")
        
        # Track type comparison
        turf_segments = [s for s in all_segments if '芝' in s['name']]
        dirt_segments = [s for s in all_segments if 'ダート' in s['name']]
        
        if turf_segments and dirt_segments:
            avg_turf_corr = np.mean([s['correlation'] for s in turf_segments])
            avg_dirt_corr = np.mean([s['correlation'] for s in dirt_segments])
            
            report.append(f"\nTurf average correlation: {avg_turf_corr:.3f}")
            report.append(f"Dirt average correlation: {avg_dirt_corr:.3f}")
            
            if avg_turf_corr > avg_dirt_corr:
                report.append("→ Models perform better on turf races")
            else:
                report.append("→ Models perform better on dirt races")
        
        # Distance insights
        distance_categories = {
            'sprint': [s for s in all_segments if 'sprint' in s['name']],
            'mile': [s for s in all_segments if 'mile' in s['name']],
            'intermediate': [s for s in all_segments if 'intermediate' in s['name']],
            'long': [s for s in all_segments if 'long' in s['name']]
        }
        
        report.append("\nDistance category performance:")
        for cat, segs in distance_categories.items():
            if segs:
                avg_corr = np.mean([s['correlation'] for s in segs])
                report.append(f"  {cat}: {avg_corr:.3f}")
        
        # Save report
        with open(f'{output_dir}/segment_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    
    def _suggest_optimal_segments(self, results: Dict, output_dir: str) -> None:
        """最適セグメント戦略の提案"""
        suggestions = []
        suggestions.append("="*60)
        suggestions.append("OPTIMAL SEGMENT STRATEGY RECOMMENDATIONS")
        suggestions.append("="*60)
        
        # Find segments with ROI > 1.0
        profitable_segments = []
        
        for seg_type, segments in results.items():
            for seg_name, data in segments.items():
                roi = data.get('roi_at_20pct', 0)
                if roi > 1.0:
                    profitable_segments.append({
                        'type': seg_type,
                        'name': seg_name,
                        'roi': roi,
                        'correlation': data.get('best_correlation', 0),
                        'size': data.get('data_size', 0)
                    })
        
        profitable_segments.sort(key=lambda x: x['roi'], reverse=True)
        
        suggestions.append("\n[Profitable Segments (ROI > 1.0)]")
        if profitable_segments:
            for i, seg in enumerate(profitable_segments, 1):
                suggestions.append(f"{i}. {seg['name']} ({seg['type']})")
                suggestions.append(f"   ROI: {seg['roi']:.3f}")
                suggestions.append(f"   Correlation: {seg['correlation']:.3f}")
                suggestions.append(f"   Annual races: ~{seg['size'] * 0.2:.0f}")
        else:
            suggestions.append("No segments with ROI > 1.0 found")
        
        # Portfolio approach
        suggestions.append("\n[Recommended Portfolio Approach]")
        
        if len(profitable_segments) >= 3:
            suggestions.append("1. Diversified Strategy:")
            suggestions.append("   - Allocate capital across top 3-5 profitable segments")
            suggestions.append("   - Weight by (ROI - 1.0) * correlation")
            
            # Calculate weights
            weights = {}
            total_weight = 0
            
            for seg in profitable_segments[:5]:
                weight = (seg['roi'] - 1.0) * seg['correlation']
                weights[seg['name']] = weight
                total_weight += weight
            
            suggestions.append("\n   Suggested allocation:")
            for name, weight in weights.items():
                pct = (weight / total_weight) * 100
                suggestions.append(f"   - {name}: {pct:.1f}%")
        
        elif len(profitable_segments) > 0:
            suggestions.append("1. Focused Strategy:")
            suggestions.append(f"   - Focus on {profitable_segments[0]['name']}")
            suggestions.append("   - Monitor performance closely")
            suggestions.append("   - Have exit criteria if ROI drops below 1.0")
        
        # Risk management
        suggestions.append("\n[Risk Management Guidelines]")
        suggestions.append("1. Start with 0.5% Kelly (very conservative)")
        suggestions.append("2. Maximum 2% of capital per race")
        suggestions.append("3. Stop loss at -20% monthly")
        suggestions.append("4. Review and rebalance monthly")
        
        # Implementation notes
        suggestions.append("\n[Implementation Notes]")
        suggestions.append("- Use separate models for each profitable segment")
        suggestions.append("- Track actual vs predicted ROI weekly")
        suggestions.append("- Be prepared for 15-20% drawdowns")
        suggestions.append("- Consider paper trading for 1 month first")
        
        # Save suggestions
        with open(f'{output_dir}/segment_strategy_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(suggestions))


def demonstrate_segment_evaluation():
    """セグメント評価のデモンストレーション"""
    # 仮想データとモデルの生成
    np.random.seed(42)
    
    # サンプルデータ
    n_races = 10000
    data = pd.DataFrame({
        'race_id': [f'2024{i//50:04d}{i%50:02d}' for i in range(n_races)],
        '芝・ダート': np.random.choice(['芝', 'ダート'], n_races, p=[0.6, 0.4]),
        '距離': np.random.choice([1200, 1600, 2000, 2400], n_races),
        '馬場': np.random.choice(['良', '稍重', '重', '不良'], n_races, p=[0.6, 0.2, 0.15, 0.05]),
        '場名': np.random.choice(['東京', '中山', '阪神', '京都', '新潟'], n_races),
        '出走頭数': np.random.randint(8, 19, n_races),
        '馬番': np.random.randint(1, 19, n_races),
        '着順_numeric': np.random.randint(1, 19, n_races),
        'オッズ_numeric': np.random.exponential(10, n_races),
        '人気': np.random.randint(1, 19, n_races),
        '日付': pd.date_range('2024-01-01', periods=n_races, freq='H')
    })
    
    # 特徴量追加
    for i in range(10):
        data[f'feature_{i}'] = np.random.randn(n_races)
    
    feature_cols = [f'feature_{i}' for i in range(10)]
    
    # ダミーモデル
    model_dict = {
        'test_model': None  # 実際はLightGBMモデルなど
    }
    
    # 評価実行
    evaluator = SegmentModelEvaluator()
    results = evaluator.evaluate_segments(data, model_dict, feature_cols)
    evaluator.create_segment_report(results)
    
    print("\nセグメント評価が完了しました。")
    print("結果は results_improved/ フォルダに保存されました。")


if __name__ == "__main__":
    demonstrate_segment_evaluation()