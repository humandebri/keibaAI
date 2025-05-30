"""
データ品質チェックの自動化スクリプト
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

from utils.data_utils import validate_data, load_race_data
from utils.config_loader import get_config

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """データ品質チェッククラス"""
    
    def __init__(self):
        """初期化"""
        self.config = get_config()
        self.results = []
        
    def check_file(self, file_path: Path) -> Dict[str, Any]:
        """
        単一ファイルの品質チェック
        
        Args:
            file_path: チェック対象ファイルパス
            
        Returns:
            チェック結果の辞書
        """
        logger.info(f"Checking file: {file_path}")
        
        try:
            # データ読み込み
            df = load_race_data(str(file_path))
            
            # 必須カラムの取得
            required_columns = self.config.get('data_processing.required_columns', [])
            
            # データ検証
            validation_results = validate_data(df, required_columns)
            
            # 追加の品質チェック
            additional_checks = self._perform_additional_checks(df)
            
            result = {
                'file': str(file_path),
                'timestamp': datetime.now().isoformat(),
                'validation': validation_results,
                'additional_checks': additional_checks,
                'status': 'PASS' if validation_results['is_valid'] else 'FAIL'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return {
                'file': str(file_path),
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'ERROR'
            }
    
    def _perform_additional_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        追加の品質チェック
        
        Args:
            df: チェック対象DataFrame
            
        Returns:
            チェック結果の辞書
        """
        checks = {}
        
        # 日付の整合性チェック
        if '日付' in df.columns:
            try:
                dates = pd.to_datetime(df['日付'], format='%Y/%m/%d', errors='coerce')
                invalid_dates = dates.isna().sum()
                checks['invalid_dates'] = int(invalid_dates)
                
                if not dates.isna().all():
                    checks['date_range'] = {
                        'min': dates.min().strftime('%Y/%m/%d'),
                        'max': dates.max().strftime('%Y/%m/%d')
                    }
            except:
                checks['date_check_error'] = True
        
        # オッズの妥当性チェック
        if 'オッズ' in df.columns:
            odds = pd.to_numeric(df['オッズ'], errors='coerce')
            checks['odds_stats'] = {
                'min': float(odds.min()) if not odds.isna().all() else None,
                'max': float(odds.max()) if not odds.isna().all() else None,
                'invalid_count': int(odds.isna().sum())
            }
            
            # 異常なオッズ（1.0未満または1000以上）
            if not odds.isna().all():
                abnormal_odds = ((odds < 1.0) | (odds > 1000)).sum()
                checks['abnormal_odds_count'] = int(abnormal_odds)
        
        # レースごとの頭数チェック
        if 'race_id' in df.columns and '馬番' in df.columns:
            horse_counts = df.groupby('race_id')['馬番'].count()
            checks['horse_count_stats'] = {
                'min': int(horse_counts.min()),
                'max': int(horse_counts.max()),
                'races_with_less_than_5': int((horse_counts < 5).sum())
            }
        
        # 着順の整合性チェック
        if '着順' in df.columns and 'race_id' in df.columns:
            invalid_finishes = 0
            for race_id, race_df in df.groupby('race_id'):
                finishes = pd.to_numeric(race_df['着順'], errors='coerce')
                valid_finishes = finishes.dropna()
                
                if len(valid_finishes) > 0:
                    # 1から始まる連続した着順があるかチェック
                    expected = set(range(1, len(valid_finishes) + 1))
                    actual = set(valid_finishes.astype(int))
                    if not expected.issubset(actual):
                        invalid_finishes += 1
                        
            checks['races_with_invalid_finish_order'] = invalid_finishes
        
        return checks
    
    def check_all_files(self, data_dir: str = None) -> List[Dict[str, Any]]:
        """
        データディレクトリ内の全ファイルをチェック
        
        Args:
            data_dir: データディレクトリパス
            
        Returns:
            全チェック結果のリスト
        """
        if data_dir is None:
            data_dir = self.config.get_path('data_dir')
        else:
            data_dir = Path(data_dir)
            
        results = []
        
        # CSVファイルとExcelファイルを検索
        for pattern in ['*.csv', '*.xlsx']:
            for file_path in data_dir.glob(pattern):
                result = self.check_file(file_path)
                results.append(result)
                
        self.results = results
        return results
    
    def generate_report(self, output_path: str = None) -> str:
        """
        品質チェックレポートを生成
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            レポートファイルパス
        """
        if not self.results:
            logger.warning("No results to report")
            return ""
            
        if output_path is None:
            output_dir = self.config.get_path('output_dir')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f'data_quality_report_{timestamp}.json'
        else:
            output_path = Path(output_path)
            
        # サマリー作成
        summary = {
            'total_files': len(self.results),
            'passed': sum(1 for r in self.results if r['status'] == 'PASS'),
            'failed': sum(1 for r in self.results if r['status'] == 'FAIL'),
            'errors': sum(1 for r in self.results if r['status'] == 'ERROR'),
            'timestamp': datetime.now().isoformat()
        }
        
        report = {
            'summary': summary,
            'details': self.results
        }
        
        # レポート保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Report saved to: {output_path}")
        
        # コンソールにサマリーを表示
        print("\n=== Data Quality Check Summary ===")
        print(f"Total files checked: {summary['total_files']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        
        # 失敗したファイルの詳細表示
        if summary['failed'] > 0:
            print("\n=== Failed Files ===")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"\nFile: {result['file']}")
                    if 'validation' in result:
                        for issue in result['validation'].get('issues', []):
                            print(f"  - {issue}")
        
        return str(output_path)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='データ品質チェックツール')
    parser.add_argument('--data-dir', type=str, help='データディレクトリパス')
    parser.add_argument('--output', type=str, help='レポート出力パス')
    parser.add_argument('--file', type=str, help='特定のファイルのみチェック')
    
    args = parser.parse_args()
    
    checker = DataQualityChecker()
    
    if args.file:
        # 単一ファイルチェック
        result = checker.check_file(Path(args.file))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 全ファイルチェック
        checker.check_all_files(args.data_dir)
        checker.generate_report(args.output)


if __name__ == '__main__':
    main()