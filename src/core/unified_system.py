#!/usr/bin/env python3
"""
統一競馬AIシステム - 複数の実装を統合したクリーンなアーキテクチャ

このモジュールは以下の複数実装を統合します:
- keiba_ai_system.py
- keiba_ai_improved_system_fixed.py 
- integrated_betting_system.py
- jra_realtime_system.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Core imports
from .config import Config
from .utils import DataLoader, FeatureProcessor, ModelManager, setup_logger
from ..strategies.base import BaseStrategy
from ..ml.ensemble_model import EnsembleRacePredictor
from ..data_processing.data_encoding_v2 import DataEncoder


class SystemMode(Enum):
    """システム動作モード"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    RESEARCH = "research"


@dataclass
class SystemStatus:
    """システム状態"""
    mode: SystemMode
    is_running: bool = False
    start_time: Optional[datetime] = None
    total_trades: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    last_update: Optional[datetime] = None


class UnifiedKeibaAISystem:
    """
    統一競馬AIシステム
    
    複数の実装を統合し、一貫性のあるAPIとアーキテクチャを提供します。
    """
    
    def __init__(
        self, 
        config: Optional[Union[Dict, Config]] = None,
        mode: SystemMode = SystemMode.BACKTEST
    ):
        """
        Args:
            config: システム設定
            mode: 動作モード
        """
        # 設定の初期化
        if isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config.load_default()
            
        # 基本属性
        self.mode = mode
        self.status = SystemStatus(mode=mode)
        self.logger = setup_logger('UnifiedKeibaAI')
        
        # コンポーネント
        self.data_loader = DataLoader()
        self.feature_processor = FeatureProcessor()
        self.model_manager = ModelManager()
        self.strategy: Optional[BaseStrategy] = None
        self.predictor: Optional[EnsembleRacePredictor] = None
        
        # データとモデル
        self.data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.feature_cols: List[str] = []
        self.results: Dict = {}
        
        # リアルタイム用
        self._data_sources = {}
        self._pending_bets = []
        self._active_races = {}
        
        self.logger.info(f"UnifiedKeibaAISystem initialized in {mode.value} mode")
    
    def load_data(
        self, 
        years: List[int], 
        use_payout_data: bool = True
    ) -> pd.DataFrame:
        """
        データの読み込み
        
        Args:
            years: 対象年のリスト
            use_payout_data: 配当データを使用するか
            
        Returns:
            読み込まれたデータ
        """
        self.logger.info(f"Loading data for years: {years}")
        
        self.data = self.data_loader.load_race_data(
            years=years, 
            use_payout_data=use_payout_data
        )
        
        self.logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def prepare_features(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        特徴量の準備
        
        Args:
            data: 処理対象データ（Noneの場合は self.data を使用）
            
        Returns:
            特徴量が追加されたデータ
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Please load data first.")
        
        self.logger.info("Preparing features...")
        
        # 基本特徴量の準備
        data = self.feature_processor.prepare_basic_features(data)
        
        # ターゲット変数の作成
        data = self.feature_processor.create_target_variables(data)
        
        # 戦略固有の特徴量（戦略が設定されている場合）
        if self.strategy:
            data = self.strategy.create_additional_features(data)
        
        # 特徴量カラムの更新
        self.feature_cols = self.feature_processor.get_feature_columns(data)
        
        if data is self.data:
            self.data = data
            
        self.logger.info(f"Prepared {len(self.feature_cols)} features")
        return data
    
    def set_strategy(self, strategy: BaseStrategy) -> None:
        """
        ベッティング戦略の設定
        
        Args:
            strategy: 使用する戦略
        """
        self.strategy = strategy
        self.logger.info(f"Strategy set: {strategy.__class__.__name__}")
    
    def train_models(
        self, 
        data: Optional[pd.DataFrame] = None,
        save_models: bool = True
    ) -> Dict:
        """
        モデルの訓練
        
        Args:
            data: 訓練データ（Noneの場合は self.data を使用）
            save_models: モデルを保存するか
            
        Returns:
            訓練されたモデルの辞書
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available for training")
        
        self.logger.info("Training models...")
        
        # アンサンブル予測器の作成
        self.predictor = EnsembleRacePredictor(self.config.model)
        
        # モデルの訓練
        self.models = self.predictor.train(
            data=data,
            feature_cols=self.feature_cols,
            target_col='着順',
            race_id_col='race_id'
        )
        
        # モデルの保存
        if save_models:
            model_path = self.config.paths.model_dir / f"unified_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.model_manager.save_models(self.models, model_path)
            self.logger.info(f"Models saved to: {model_path}")
        
        self.logger.info("Model training completed")
        return self.models
    
    def load_models(self, model_path: Union[str, Path]) -> Dict:
        """
        モデルの読み込み
        
        Args:
            model_path: モデルファイルのパス
            
        Returns:
            読み込まれたモデルの辞書
        """
        self.logger.info(f"Loading models from: {model_path}")
        
        self.models = self.model_manager.load_models(model_path)
        
        # 予測器の再構築
        self.predictor = EnsembleRacePredictor(self.config.model)
        self.predictor.models = self.models
        
        self.logger.info("Models loaded successfully")
        return self.models
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        予測の実行
        
        Args:
            data: 予測対象データ
            
        Returns:
            予測結果が追加されたデータ
        """
        if self.predictor is None:
            raise ValueError("No predictor available. Please train or load models first.")
        
        self.logger.info(f"Making predictions for {len(data)} records")
        
        predictions = self.predictor.predict(data, self.feature_cols)
        data['predicted_rank'] = predictions
        
        return data
    
    def run_backtest(
        self,
        train_years: List[int],
        test_years: List[int],
        strategy: Optional[BaseStrategy] = None
    ) -> Dict:
        """
        バックテストの実行
        
        Args:
            train_years: 訓練期間
            test_years: テスト期間
            strategy: 使用する戦略（Noneの場合は設定済みの戦略を使用）
            
        Returns:
            バックテスト結果
        """
        if strategy:
            self.set_strategy(strategy)
            
        if self.strategy is None:
            raise ValueError("No strategy set for backtesting")
        
        self.logger.info(f"Running backtest: train={train_years}, test={test_years}")
        
        # データの読み込みと準備
        all_years = sorted(set(train_years + test_years))
        self.load_data(all_years)
        self.prepare_features()
        
        # バックテストの実行
        results = self.strategy.run_backtest(
            data=self.data,
            train_years=train_years,
            test_years=test_years,
            feature_cols=self.feature_cols
        )
        
        self.results = results
        self.logger.info("Backtest completed")
        
        return results
    
    async def run_realtime(self) -> None:
        """
        リアルタイム実行
        """
        if self.mode not in [SystemMode.PAPER_TRADING, SystemMode.LIVE_TRADING]:
            raise ValueError(f"Real-time execution not supported in {self.mode.value} mode")
        
        self.status.is_running = True
        self.status.start_time = datetime.now()
        
        self.logger.info("Starting real-time execution...")
        
        try:
            while self.status.is_running:
                # レース情報の取得
                upcoming_races = await self._get_upcoming_races()
                
                for race_info in upcoming_races:
                    await self._process_race(race_info)
                
                # 短い間隔で監視
                await asyncio.sleep(30)
                
        except Exception as e:
            self.logger.error(f"Error in real-time execution: {e}")
            self.status.is_running = False
        
        self.logger.info("Real-time execution stopped")
    
    def stop_realtime(self) -> None:
        """リアルタイム実行の停止"""
        self.status.is_running = False
        self.logger.info("Stop signal sent")
    
    async def _get_upcoming_races(self) -> List[Dict]:
        """近日開催レースの取得"""
        # 実装詳細は別途
        return []
    
    async def _process_race(self, race_info: Dict) -> None:
        """個別レースの処理"""
        # 実装詳細は別途
        pass
    
    def get_status(self) -> SystemStatus:
        """システム状態の取得"""
        return self.status
    
    def get_results(self) -> Dict:
        """結果の取得"""
        return self.results
    
    def export_results(self, output_path: Union[str, Path]) -> None:
        """
        結果のエクスポート
        
        Args:
            output_path: 出力先パス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Results exported to: {output_path}")


# 後方互換性のためのエイリアス
KeibaAISystem = UnifiedKeibaAISystem