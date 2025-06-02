#!/usr/bin/env python3
"""
統合競馬予測・自動投票システム
既存の予測モデルとリアルタイムデータ取得を統合
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import pickle

# 既存モジュールのインポート
from keiba_ai_improved_system_fixed import ImprovedKeibaAISystem
from jra_realtime_system import (
    JRARealTimeSystem, 
    NetkeibaDataCollector,
    JRAIPATInterface
)


class IntegratedKeibaSystem:
    """統合競馬予測・投票システム"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 各コンポーネントの初期化
        self.predictor = self._load_predictor()
        self.data_collector = JRARealTimeSystem()
        self.netkeiba = NetkeibaDataCollector()
        self.ipat = None  # ログイン時に初期化
        
        # 状態管理
        self.is_running = False
        self.pending_bets = []
        self.confirmed_bets = []
        self.daily_stats = self._init_daily_stats()
        
    def _get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'model_path': 'model_2020_2025/model_2020_2025.pkl',  # 2020-2025モデルを使用
            'max_bet_per_race': 10000,
            'max_daily_loss': 50000,
            'min_expected_value': 1.1,
            'kelly_fraction': 0.05,
            'data_refresh_interval': 300,  # 5分
            'enable_auto_betting': False,  # 安全のためデフォルトはオフ
            'notification': {
                'email': None,
                'slack_webhook': None
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger('IntegratedKeibaSystem')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # ファイルハンドラー
            Path('logs').mkdir(exist_ok=True)
            fh = logging.FileHandler(
                f'logs/integrated_system_{datetime.now().strftime("%Y%m%d")}.log'
            )
            fh.setLevel(logging.DEBUG)
            
            # コンソールハンドラー
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def _load_predictor(self) -> ImprovedKeibaAISystem:
        """訓練済みモデルの読み込み"""
        try:
            # 既存のImprovedKeibaAISystemを使用
            predictor = ImprovedKeibaAISystem()
            
            # 保存されたモデルがあれば読み込み
            model_path = Path(self.config['model_path'])
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    saved_models = pickle.load(f)
                    predictor.models = saved_models
                    self.logger.info("保存済みモデルを読み込みました")
            else:
                self.logger.warning("保存済みモデルが見つかりません。新規訓練が必要です。")
            
            return predictor
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            return ImprovedKeibaAISystem()
    
    def _init_daily_stats(self) -> Dict:
        """日次統計の初期化"""
        return {
            'date': datetime.now().date(),
            'total_bets': 0,
            'total_amount': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0,
            'races_analyzed': 0
        }
    
    async def start(self):
        """システム起動"""
        self.logger.info("統合システムを起動します...")
        self.is_running = True
        
        try:
            # IPATログイン（必要な場合）
            if self.config['enable_auto_betting']:
                await self._init_ipat()
            
            # メインループ
            await self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("システムを停止します...")
        except Exception as e:
            self.logger.error(f"システムエラー: {e}")
        finally:
            self.is_running = False
            await self._cleanup()
    
    async def _init_ipat(self):
        """IPAT初期化（クレデンシャルは環境変数から）"""
        import os
        
        member_id = os.getenv('JRA_MEMBER_ID')
        pin = os.getenv('JRA_PIN')
        pars = os.getenv('JRA_PARS')
        
        if all([member_id, pin, pars]):
            self.ipat = JRAIPATInterface(member_id, pin, pars)
            if self.ipat.login():
                self.logger.info("IPATログイン成功")
            else:
                self.logger.error("IPATログイン失敗")
                self.config['enable_auto_betting'] = False
        else:
            self.logger.warning("IPAT認証情報が設定されていません")
            self.config['enable_auto_betting'] = False
    
    async def _main_loop(self):
        """メインループ"""
        while self.is_running:
            try:
                # 今後のレース情報取得（デフォルトで明日から3日間）
                races = await self._get_upcoming_races(
                    days_ahead=self.config.get('days_ahead', 1),
                    max_days=self.config.get('max_days_to_analyze', 3)
                )
                
                if races:
                    self.logger.info(f"{len(races)}件のレースを分析します")
                    
                    # 日付ごとにグループ化して処理
                    from itertools import groupby
                    from operator import itemgetter
                    
                    grouped_races = groupby(races, key=itemgetter('date'))
                    for date, races_on_date in grouped_races:
                        races_list = list(races_on_date)
                        self.logger.info(f"{date}: {len(races_list)}レース")
                        
                        # 各レースを処理
                        for race in races_list:
                            await self._process_race(race)
                            
                            # レート制限
                            await asyncio.sleep(2)
                
                # 統計更新
                self._update_daily_stats()
                
                # 次のサイクルまで待機
                await asyncio.sleep(self.config['data_refresh_interval'])
                
            except Exception as e:
                self.logger.error(f"メインループエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
    
    async def _get_today_races(self) -> List[Dict]:
        """本日のレース情報を統合取得"""
        return await self._get_upcoming_races(days_ahead=0, max_days=1)
    
    async def _get_upcoming_races(self, days_ahead: int = 1, max_days: int = 7) -> List[Dict]:
        """今後のレース情報を統合取得
        
        Args:
            days_ahead: 何日先から取得するか（0=今日、1=明日）
            max_days: 最大何日先まで取得するか
        
        Returns:
            レース情報のリスト
        """
        all_races = []
        
        try:
            # JRA公式から取得
            jra_races = self.data_collector.get_upcoming_races(days_ahead=days_ahead, max_days=max_days)
            
            # netkeiba.comから取得
            netkeiba_races = self.netkeiba.get_upcoming_race_list(days_ahead=days_ahead, max_days=max_days)
            
            # データ統合（重複排除）
            race_ids = set()
            
            for race in jra_races:
                race_id = f"{race.get('date', '')}_{race['racecourse']}_{race['race_number']}"
                if race_id not in race_ids:
                    race_ids.add(race_id)
                    all_races.append({
                        'source': 'jra',
                        'race_id': race_id,
                        **race
                    })
            
            for race in netkeiba_races:
                if race['race_id'] not in race_ids:
                    race_ids.add(race['race_id'])
                    all_races.append({
                        'source': 'netkeiba',
                        **race
                    })
            
            # 日付と時刻でソート
            all_races.sort(key=lambda x: (x.get('date', ''), x.get('time', '00:00')))
            
        except Exception as e:
            self.logger.error(f"レース情報取得エラー: {e}")
        
        return all_races
    
    async def _process_race(self, race: Dict):
        """個別レースの処理"""
        try:
            race_id = race['race_id']
            self.logger.info(f"レース処理開始: {race_id}")
            
            # レース詳細情報取得
            race_details = await self._get_race_details(race)
            
            if not race_details:
                return
            
            # 予測用データ準備
            race_df = self._prepare_prediction_data(race_details)
            
            # 予測実行
            predictions = self._run_prediction(race_df)
            
            # ベッティング判断
            betting_opportunities = self._analyze_betting_opportunities(
                predictions, race_details
            )
            
            if betting_opportunities:
                await self._handle_betting_opportunities(
                    race, betting_opportunities
                )
            
            self.daily_stats['races_analyzed'] += 1
            
        except Exception as e:
            self.logger.error(f"レース処理エラー ({race['race_id']}): {e}")
    
    async def _get_race_details(self, race: Dict) -> Optional[Dict]:
        """レース詳細情報の取得"""
        try:
            if race['source'] == 'jra':
                return self.data_collector.get_race_details(race['race_id'])
            elif race['source'] == 'netkeiba':
                # netkeiba用の詳細取得
                race_card = self.netkeiba.get_race_card(race['race_id'])
                odds = self.netkeiba.get_real_time_odds(race['race_id'])
                
                return {
                    'race_id': race['race_id'],
                    'race_card': race_card,
                    'odds': odds,
                    'race_info': race
                }
        except Exception as e:
            self.logger.error(f"詳細情報取得エラー: {e}")
            return None
    
    def _prepare_prediction_data(self, race_details: Dict) -> pd.DataFrame:
        """予測用データの準備"""
        # race_detailsの形式に応じてDataFrameを作成
        if 'race_card' in race_details:
            # netkeibaデータの場合
            df = race_details['race_card'].copy()
        else:
            # JRAデータの場合
            horses = race_details.get('horses', [])
            df = pd.DataFrame(horses)
        
        # 必要なカラムの追加・変換
        if 'オッズ' not in df.columns and 'odds' in race_details:
            # オッズ情報の追加
            odds_data = race_details['odds'].get('win', {})
            df['オッズ'] = df['馬番'].astype(str).map(odds_data)
        
        # 既存モデルが期待するカラムを確保
        required_columns = [
            '馬番', '馬名', '性齢', '斤量', '騎手', 
            'オッズ', '人気', '調教師'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # race_idの追加
        df['race_id'] = race_details['race_id']
        
        return df
    
    def _run_prediction(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """予測実行"""
        try:
            # 特徴量エンジニアリング
            race_df = self.predictor.create_advanced_features(race_df)
            
            # 予測
            if self.predictor.models and 'pure_ability_lgb' in self.predictor.models:
                model = self.predictor.models['pure_ability_lgb']
                features = self.predictor.non_odds_feature_cols
                
                X = race_df[features].fillna(0).values
                predictions = model.predict(X, num_iteration=model.best_iteration)
                
                race_df['predicted_rank'] = predictions
                race_df['win_probability'] = self._calculate_win_probability(predictions)
            else:
                self.logger.warning("予測モデルが利用できません")
                race_df['predicted_rank'] = np.nan
                race_df['win_probability'] = np.nan
            
            return race_df
            
        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            return race_df
    
    def _calculate_win_probability(self, predictions: np.ndarray) -> np.ndarray:
        """予測順位から勝率を計算"""
        # ソフトマックス変換
        exp_neg_pred = np.exp(-predictions * 2)
        win_probs = exp_neg_pred / exp_neg_pred.sum()
        return win_probs
    
    def _analyze_betting_opportunities(self, 
                                     predictions: pd.DataFrame,
                                     race_details: Dict) -> List[Dict]:
        """ベッティング機会の分析"""
        opportunities = []
        
        # 予測上位馬を取得
        top_horses = predictions.nsmallest(5, 'predicted_rank')
        
        for _, horse in top_horses.iterrows():
            if pd.isna(horse['predicted_rank']):
                continue
            
            # 期待値計算
            win_prob = horse['win_probability']
            odds = horse.get('オッズ', 10)
            
            if pd.isna(odds) or odds <= 0:
                continue
            
            expected_value = win_prob * odds
            
            # 期待値が閾値を超える場合
            if expected_value >= self.config['min_expected_value']:
                opportunities.append({
                    'horse_number': int(horse['馬番']),
                    'horse_name': horse.get('馬名', 'Unknown'),
                    'win_probability': win_prob,
                    'odds': odds,
                    'expected_value': expected_value,
                    'predicted_rank': horse['predicted_rank']
                })
        
        # 期待値でソート
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return opportunities
    
    async def _handle_betting_opportunities(self, 
                                          race: Dict,
                                          opportunities: List[Dict]):
        """ベッティング機会の処理"""
        self.logger.info(
            f"ベッティング機会検出: {race['race_id']} "
            f"({len(opportunities)}件)"
        )
        
        for opp in opportunities[:3]:  # 上位3つまで
            # ベットサイズ計算
            bet_size = self._calculate_bet_size(opp)
            
            if bet_size < 100:
                continue
            
            # 日次制限チェック
            if self._check_daily_limits(bet_size):
                bet_info = {
                    'race': race,
                    'opportunity': opp,
                    'bet_size': bet_size,
                    'timestamp': datetime.now()
                }
                
                if self.config['enable_auto_betting']:
                    await self._place_bet(bet_info)
                else:
                    await self._log_betting_opportunity(bet_info)
    
    def _calculate_bet_size(self, opportunity: Dict) -> int:
        """Kelly基準によるベットサイズ計算"""
        win_prob = opportunity['win_probability']
        odds = opportunity['odds']
        
        # Kelly計算
        kelly = (win_prob * odds - 1) / (odds - 1)
        
        # 保守的なKelly
        safe_kelly = max(0, kelly * self.config['kelly_fraction'])
        
        # 現在の資金（仮）
        current_bankroll = 1000000 - self.daily_stats['profit_loss']
        
        # ベットサイズ
        bet_size = min(
            current_bankroll * safe_kelly,
            self.config['max_bet_per_race']
        )
        
        # 100円単位に丸める
        return int(bet_size / 100) * 100
    
    def _check_daily_limits(self, bet_size: int) -> bool:
        """日次制限チェック"""
        # 日次損失制限
        if self.daily_stats['profit_loss'] < -self.config['max_daily_loss']:
            self.logger.warning("日次損失制限に達しました")
            return False
        
        # 日次ベット額制限
        if self.daily_stats['total_amount'] + bet_size > 100000:
            self.logger.warning("日次ベット額制限に達しました")
            return False
        
        return True
    
    async def _place_bet(self, bet_info: Dict):
        """実際の投票処理"""
        if not self.ipat:
            self.logger.error("IPATが初期化されていません")
            return
        
        try:
            # 投票実行
            result = self.ipat.place_bet(
                race_id=bet_info['race']['race_id'],
                bet_type='WIN',  # 単勝
                selections=[bet_info['opportunity']['horse_number']],
                amount=bet_info['bet_size']
            )
            
            # 結果記録
            bet_record = {
                **bet_info,
                'result': result,
                'status': 'pending_confirmation'
            }
            
            self.pending_bets.append(bet_record)
            
            # 通知送信
            await self._send_confirmation_notification(bet_record)
            
            # 統計更新
            self.daily_stats['total_bets'] += 1
            self.daily_stats['total_amount'] += bet_info['bet_size']
            
        except Exception as e:
            self.logger.error(f"投票エラー: {e}")
    
    async def _log_betting_opportunity(self, bet_info: Dict):
        """ベッティング機会のログ記録（実投票なし）"""
        log_entry = {
            'timestamp': bet_info['timestamp'].isoformat(),
            'race_id': bet_info['race']['race_id'],
            'horse_number': bet_info['opportunity']['horse_number'],
            'horse_name': bet_info['opportunity']['horse_name'],
            'expected_value': bet_info['opportunity']['expected_value'],
            'suggested_bet': bet_info['bet_size']
        }
        
        # ログファイルに記録
        log_path = Path('logs/betting_opportunities.json')
        
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_path, 'w') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"ベッティング機会記録: {log_entry['race_id']} "
                f"馬番{log_entry['horse_number']} EV={log_entry['expected_value']:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"ログ記録エラー: {e}")
    
    async def _send_confirmation_notification(self, bet_record: Dict):
        """確認通知の送信"""
        message = f"""
        【投票確認依頼】
        時刻: {bet_record['timestamp'].strftime('%H:%M')}
        レース: {bet_record['race']['race_id']}
        馬番: {bet_record['opportunity']['horse_number']}
        馬名: {bet_record['opportunity']['horse_name']}
        金額: ¥{bet_record['bet_size']:,}
        期待値: {bet_record['opportunity']['expected_value']:.2f}
        
        ※必ずJRA IPATで手動確認してください
        """
        
        # 実際の通知実装（Email/LINE/Slack等）
        self.logger.warning(message)
    
    def _update_daily_stats(self):
        """日次統計の更新"""
        # 日付が変わった場合はリセット
        if self.daily_stats['date'] != datetime.now().date():
            self._save_daily_stats()
            self.daily_stats = self._init_daily_stats()
    
    def _save_daily_stats(self):
        """日次統計の保存"""
        stats_path = Path('logs/daily_stats.json')
        
        try:
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    all_stats = json.load(f)
            else:
                all_stats = []
            
            all_stats.append({
                **self.daily_stats,
                'date': self.daily_stats['date'].isoformat()
            })
            
            with open(stats_path, 'w') as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"統計保存エラー: {e}")
    
    async def _cleanup(self):
        """クリーンアップ処理"""
        self.logger.info("クリーンアップ処理を実行中...")
        
        # 最終統計の保存
        self._save_daily_stats()
        
        # 未確認ベットの警告
        if self.pending_bets:
            self.logger.warning(
                f"未確認の投票が{len(self.pending_bets)}件あります！"
            )
        
        # セッションのクローズ
        if hasattr(self.data_collector, 'session') and self.data_collector.session:
            self.data_collector.session.close()


async def main():
    """メイン実行"""
    print("=" * 60)
    print("統合競馬予測・投票システム")
    print("=" * 60)
    
    # 設定
    config = {
        'enable_auto_betting': False,  # 安全のため手動モード
        'min_expected_value': 1.2,
        'kelly_fraction': 0.025,  # 2.5% Kelly
        'max_bet_per_race': 5000,
        'max_daily_loss': 30000,
        'days_ahead': 1,  # 明日から検索開始（0=今日、1=明日）
        'max_days_to_analyze': 3  # 最大3日先まで分析
    }
    
    # システム起動
    system = IntegratedKeibaSystem(config)
    
    print("\n[モード: 分析のみ（投票なし）]")
    print(f"分析対象: 明日から{config['max_days_to_analyze']}日間のレース")
    print("ベッティング機会はログに記録されます")
    print("\nCtrl+C で終了します...\n")
    
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())