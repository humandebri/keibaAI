"""
ベッティング戦略の基底クラス
すべてのバックテスト戦略の共通インターフェース
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

from ..core.config import config
from ..core.utils import (
    DataLoader, FeatureProcessor, ModelManager, 
    setup_logger, calculate_return_metrics, format_currency
)


class BaseStrategy(ABC):
    """バックテスト戦略の基底クラス"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.logger = setup_logger(f"{__name__}.{name}")
        self.config = config
        
        # 共通コンポーネント
        self.data_loader = DataLoader(self.logger)
        self.feature_processor = FeatureProcessor(self.logger)
        self.model_manager = ModelManager(self.logger)
        
        # データ
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # 結果
        self.results = {
            'trades': [],
            'capital_history': [],
            'metrics': {}
        }
    
    def load_data(self, start_year: int = 2014, end_year: int = 2024, use_payout_data: bool = False):
        """データの読み込みと準備"""
        self.logger.info(f"Loading data from {start_year} to {end_year}")
        
        # データ読み込み
        years = list(range(start_year, end_year + 1))
        self.data = self.data_loader.load_race_data(years, use_payout_data=use_payout_data)
        
        # 基本的な特徴量準備
        self.data = self.feature_processor.prepare_basic_features(self.data)
        self.data = self.feature_processor.create_target_variables(self.data)
        
        # 追加の特徴量作成（戦略固有）
        self.data = self.create_additional_features(self.data)
        
        self.logger.info(f"Total data loaded: {len(self.data)} rows")
    
    def split_data(self, train_years: List[int], test_years: List[int]):
        """訓練・テストデータの分割"""
        self.train_data = self.data[self.data['year'].isin(train_years)]
        self.test_data = self.data[self.data['year'].isin(test_years)]
        
        self.logger.info(f"Train data: {len(self.train_data)} rows ({train_years})")
        self.logger.info(f"Test data: {len(self.test_data)} rows ({test_years})")
    
    @abstractmethod
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """戦略固有の追加特徴量作成"""
        pass
    
    @abstractmethod
    def train_model(self) -> any:
        """モデルの訓練"""
        pass
    
    @abstractmethod
    def select_bets(self, race_data: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """ベッティング対象の選択"""
        pass
    
    @abstractmethod
    def calculate_bet_amount(self, capital: float, bet_info: Dict) -> float:
        """ベット額の計算"""
        pass
    
    def run_backtest(self, data: pd.DataFrame, train_years: List[int], 
                     test_years: List[int], feature_cols: List[str],
                     initial_capital: float = 1_000_000) -> Dict:
        """バックテストの実行（統一システム用）"""
        self.logger.info(f"Running backtest with initial capital: {format_currency(initial_capital)}")
        
        # データを分割
        train_data = data[data['year'].isin(train_years)]
        test_data = data[data['year'].isin(test_years)]
        
        self.train_data = train_data
        self.test_data = test_data
        self.feature_cols = feature_cols
        
        # モデル訓練
        model = self.train_model()
        
        # バックテスト初期化
        capital = initial_capital
        total_bets = 0
        total_wins = 0
        
        # レースごとに処理
        unique_races = self.test_data['race_id'].unique()
        self.logger.info(f"Processing {len(unique_races)} races")
        
        for i, race_id in enumerate(unique_races):
            if i % 1000 == 0:
                self.logger.debug(f"Processing race {i}/{len(unique_races)}")
            
            race_data = self.test_data[self.test_data['race_id'] == race_id]
            
            # 予測
            features = self._get_features(race_data)
            if features is None:
                continue
            
            predictions = model.predict(features)
            
            # ベット選択
            bets = self.select_bets(race_data, predictions)
            
            # ベット実行
            for bet in bets:
                bet_amount = self.calculate_bet_amount(capital, bet)
                
                if bet_amount <= 0 or bet_amount > capital:
                    continue
                
                total_bets += 1
                
                # 結果判定
                profit = self._calculate_profit(bet, bet_amount)
                capital += profit
                
                if profit > 0:
                    total_wins += 1
                
                # 結果記録
                self.results['trades'].append({
                    'race_id': race_id,
                    'bet': bet,
                    'amount': bet_amount,
                    'profit': profit,
                    'capital': capital
                })
                
                # 破産チェック
                if capital <= 0:
                    self.logger.warning(f"Bankrupt at race {i}")
                    break
            
            self.results['capital_history'].append(capital)
            
            if capital <= 0:
                break
        
        # 最終結果の計算
        self._calculate_final_metrics(initial_capital, capital, total_bets, total_wins)
        
        return self.results
    
    def _get_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """特徴量の取得"""
        feature_cols = self.feature_processor.get_feature_columns(data)
        
        if not feature_cols:
            return None
        
        features = []
        for col in feature_cols:
            if col in data.columns:
                features.append(data[col].values)
        
        if not features:
            return None
        
        return np.column_stack(features)
    
    @abstractmethod
    def _calculate_profit(self, bet: Dict, bet_amount: float) -> float:
        """利益の計算（戦略固有）"""
        pass
    
    def _calculate_final_metrics(self, initial_capital: float, final_capital: float,
                               total_bets: int, total_wins: int):
        """最終指標の計算"""
        metrics = calculate_return_metrics(initial_capital, final_capital, n_years=3)
        
        self.results['metrics'] = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'profit_loss': metrics['profit_loss'],
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0,
            'avg_bet': np.mean([t['amount'] for t in self.results['trades']]) if self.results['trades'] else 0
        }
    
    def print_results(self):
        """結果の表示"""
        metrics = self.results['metrics']
        
        print(f"\n=== {self.name} Results ===")
        print(f"Initial Capital: {format_currency(metrics['initial_capital'])}")
        print(f"Final Capital: {format_currency(metrics['final_capital'])}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Total Bets: {metrics['total_bets']}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Average Bet: {format_currency(metrics['avg_bet'])}")