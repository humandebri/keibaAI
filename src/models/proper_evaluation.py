"""
適切な時系列検証を行うモデル評価スクリプト
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
import ast
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ProperModelEvaluator:
    """時系列を考慮した適切なモデル評価クラス"""
    
    def __init__(self, base_dir: str = '.'):
        self.base_dir = base_dir
        
    def load_yearly_data(self, years: List[int]) -> pd.DataFrame:
        """年単位でデータを読み込む"""
        dfs = []
        for year in years:
            file_path = f'{self.base_dir}/encoded/{year}encoded_data.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded {year} data: {len(df)} rows")
                dfs.append(df)
            else:
                print(f"Warning: {file_path} not found")
        
        if not dfs:
            raise ValueError("No data files found")
            
        return pd.concat(dfs, ignore_index=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の準備"""
        # 着順を二値分類用に変換（3着以内を1）
        df['target'] = df['着順'].map(lambda x: 1 if x < 4 else 0)
        
        # 不要なカラムを除外
        exclude_cols = ['着順', 'target', 'オッズ', '人気', '上がり', '走破時間', '通過順']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df, feature_cols
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_valid: pd.DataFrame, y_valid: pd.Series) -> Tuple[lgb.LGBMClassifier, float]:
        """モデルの学習"""
        # クラス比率の計算
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        print(f"\nTraining set:")
        print(f"Negative/Positive ratio: {ratio:.2f}")
        print(f"負例数（3着外）: {(y_train == 0).sum()}")
        print(f"正例数（3着内）: {(y_train == 1).sum()}")
        
        # パラメータ設定
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'scale_pos_weight': ratio,
            'random_state': 42,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'n_estimators': 300,
            'early_stopping_rounds': 50
        }
        
        # モデル学習
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 検証データでの予測
        y_pred = model.predict_proba(X_valid)[:, 1]
        auc_score = roc_auc_score(y_valid, y_pred)
        
        # 最適閾値の探索
        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_pred)
        fbeta_scores = (1 + 0.5**2) * (precisions * recalls) / (0.5**2 * precisions + recalls)
        best_idx = np.argmax(fbeta_scores[:-1])  # 最後の要素を除く
        optimal_threshold = thresholds[best_idx]
        
        print(f"\nValidation AUC: {auc_score:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        return model, optimal_threshold
    
    def evaluate_on_test(self, model: lgb.LGBMClassifier, 
                        X_test: pd.DataFrame, y_test: pd.Series,
                        threshold: float) -> Dict[str, float]:
        """テストデータでの評価"""
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        
        # 混同行列の計算
        TP = ((y_test == 1) & (y_pred >= threshold)).sum()
        FP = ((y_test == 0) & (y_pred >= threshold)).sum()
        TN = ((y_test == 0) & (y_pred < threshold)).sum()
        FN = ((y_test == 1) & (y_pred < threshold)).sum()
        
        total = len(y_test)
        
        print(f"\nTest set evaluation:")
        print(f"Total cases: {total}")
        print(f"True positives: {TP} ({TP/total*100:.2f}%)")
        print(f"False positives: {FP} ({FP/total*100:.2f}%)")
        print(f"True negatives: {TN} ({TN/total*100:.2f}%)")
        print(f"False negatives: {FN} ({FN/total*100:.2f}%)")
        print(f"Test AUC: {auc_score:.4f}")
        
        return {
            'auc': auc_score,
            'predictions': y_pred,
            'threshold': threshold
        }
    
    def calculate_returns(self, test_df: pd.DataFrame, predictions: np.ndarray,
                         threshold: float, payback_year: int) -> Dict[str, float]:
        """回収率の計算"""
        # 払戻データの読み込み
        payback_path = f'{self.base_dir}/payback/{payback_year}.csv'
        if not os.path.exists(payback_path):
            print(f"Warning: Payback data not found for {payback_year}")
            return {'win_return': 0, 'place_return': 0, 'bet_count': 0}
        
        payback_df = pd.read_csv(payback_path, encoding='SHIFT-JIS', dtype={'race_id': str})
        payback_df['race_id'] = payback_df['race_id'].str.replace(r'\.0$', '', regex=True)
        payback_df.set_index('race_id', inplace=True)
        
        # 払戻データの変換
        for col in ['単勝', '複勝']:
            if col in payback_df.columns:
                payback_df[col] = payback_df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).strip().startswith('[') else []
                )
        
        # 賭ける馬の決定
        betting_horses = []
        for i in range(len(predictions)):
            if predictions[i] >= threshold:
                race_id = str(int(float(test_df.iloc[i]['race_id'])))
                horse_num = str(int(float(test_df.iloc[i]['馬番'])))
                betting_horses.append((race_id, horse_num))
        
        print(f"\nBetting on {len(betting_horses)} horses")
        
        # 回収金額の計算
        win_return = 0
        place_return = 0
        
        for race_id, horse_num in betting_horses:
            if race_id in payback_df.index:
                race_data = payback_df.loc[race_id]
                
                # 単勝
                if '単勝' in race_data and isinstance(race_data['単勝'], list):
                    win_data = race_data['単勝']
                    for j in range(0, len(win_data), 2):
                        if j+1 < len(win_data) and win_data[j] == horse_num:
                            win_return += int(win_data[j + 1].replace(',', ''))
                
                # 複勝
                if '複勝' in race_data and isinstance(race_data['複勝'], list):
                    place_data = race_data['複勝']
                    for j in range(0, len(place_data), 2):
                        if j+1 < len(place_data) and place_data[j] == horse_num:
                            place_return += int(place_data[j + 1].replace(',', ''))
        
        # 回収率計算（100円賭けと仮定）
        bet_amount = len(betting_horses) * 100
        win_return_rate = (win_return / bet_amount * 100) if bet_amount > 0 else 0
        place_return_rate = (place_return / bet_amount * 100) if bet_amount > 0 else 0
        
        print(f"Win return rate: {win_return_rate:.2f}%")
        print(f"Place return rate: {place_return_rate:.2f}%")
        
        return {
            'win_return_rate': win_return_rate,
            'place_return_rate': place_return_rate,
            'bet_count': len(betting_horses),
            'total_win_return': win_return,
            'total_place_return': place_return
        }
    
    def run_time_series_validation(self, train_years: List[int], 
                                 valid_year: int, 
                                 test_year: int):
        """時系列検証の実行"""
        print(f"\n{'='*60}")
        print(f"Time Series Validation")
        print(f"Train years: {train_years}")
        print(f"Validation year: {valid_year}")
        print(f"Test year: {test_year}")
        print(f"{'='*60}")
        
        # データ読み込み
        train_data = self.load_yearly_data(train_years)
        valid_data = self.load_yearly_data([valid_year])
        test_data = self.load_yearly_data([test_year])
        
        # 特徴量準備
        train_data, feature_cols = self.prepare_features(train_data)
        valid_data, _ = self.prepare_features(valid_data)
        test_data, _ = self.prepare_features(test_data)
        
        # データ分割
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_valid = valid_data[feature_cols]
        y_valid = valid_data['target']
        X_test = test_data[feature_cols]
        y_test = test_data['target']
        
        # モデル学習
        model, threshold = self.train_model(X_train, y_train, X_valid, y_valid)
        
        # テストデータ評価
        test_results = self.evaluate_on_test(model, X_test, y_test, threshold)
        
        # 回収率計算
        test_data_with_pred = test_data.copy()
        return_results = self.calculate_returns(
            test_data_with_pred, 
            test_results['predictions'],
            threshold,
            test_year
        )
        
        return {
            'model': model,
            'threshold': threshold,
            'test_auc': test_results['auc'],
            'return_results': return_results
        }
    
    def run_walk_forward_analysis(self, start_year: int, end_year: int, 
                                train_window: int = 2):
        """ウォークフォワード分析"""
        results = []
        
        for test_year in range(start_year + train_window + 1, end_year + 1):
            train_years = list(range(test_year - train_window - 1, test_year - 1))
            valid_year = test_year - 1
            
            result = self.run_time_series_validation(train_years, valid_year, test_year)
            result['test_year'] = test_year
            results.append(result)
        
        # 結果のサマリー
        print(f"\n{'='*60}")
        print("Walk Forward Analysis Summary")
        print(f"{'='*60}")
        
        total_bet = 0
        total_win_return = 0
        total_place_return = 0
        
        for r in results:
            print(f"\nYear {r['test_year']}:")
            print(f"  AUC: {r['test_auc']:.4f}")
            print(f"  Bets: {r['return_results']['bet_count']}")
            print(f"  Win return: {r['return_results']['win_return_rate']:.2f}%")
            print(f"  Place return: {r['return_results']['place_return_rate']:.2f}%")
            
            total_bet += r['return_results']['bet_count']
            total_win_return += r['return_results']['total_win_return']
            total_place_return += r['return_results']['total_place_return']
        
        # 全体の回収率
        if total_bet > 0:
            overall_win_rate = (total_win_return / (total_bet * 100)) * 100
            overall_place_rate = (total_place_return / (total_bet * 100)) * 100
            
            print(f"\nOverall Performance:")
            print(f"Total bets: {total_bet}")
            print(f"Overall win return rate: {overall_win_rate:.2f}%")
            print(f"Overall place return rate: {overall_place_rate:.2f}%")
        
        return results


def main():
    """メイン実行関数"""
    evaluator = ProperModelEvaluator()
    
    # 例1: 単一の時系列検証
    # 2019-2020年で学習、2021年で検証、2022年でテスト
    result = evaluator.run_time_series_validation(
        train_years=[2019, 2020],
        valid_year=2021,
        test_year=2022
    )
    
    # 例2: ウォークフォワード分析
    # 2019年から2023年まで、2年間の訓練データで順次予測
    results = evaluator.run_walk_forward_analysis(
        start_year=2019,
        end_year=2023,
        train_window=2
    )


if __name__ == '__main__':
    main()