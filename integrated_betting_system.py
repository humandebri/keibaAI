#!/usr/bin/env python3
"""
統合競馬予測・自動投票システム
改良版モデル（train_model_2020_2025.py）とリアルタイムデータ取得を統合
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
import joblib
import lightgbm as lgb

# リアルタイムシステムのインポート
try:
    from jra_realtime_system import (
        JRARealTimeSystem, 
        NetkeibaDataCollector,
        JRAIPATInterface
    )
except ImportError:
    print("警告: jra_realtime_system.pyが見つかりません")
    # ダミークラスを定義
    class JRARealTimeSystem:
        def get_upcoming_races(self, **kwargs): return []
        def get_race_details(self, race_id): return None
    
    class NetkeibaDataCollector:
        def get_upcoming_race_list(self, **kwargs): return []
        def get_race_card(self, race_id): return None
        def get_real_time_odds(self, race_id): return None
    
    class JRAIPATInterface:
        def __init__(self, *args): pass
        def login(self): return False
        def place_bet(self, **kwargs): return None


class IntegratedKeibaSystem:
    """統合競馬予測・投票システム"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # 各コンポーネントの初期化
        self.model = None
        self.feature_cols = None
        self._load_model()
        
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
    
    def _load_model(self):
        """訓練済みモデルの読み込み（改良版）"""
        try:
            # モデルファイルのパス
            model_path = Path(self.config['model_path'])
            feature_path = Path(self.config['model_path']).parent / 'feature_cols_2020_2025.pkl'
            
            if model_path.exists() and feature_path.exists():
                # LightGBMモデルの読み込み
                self.model = joblib.load(model_path)
                self.feature_cols = joblib.load(feature_path)
                self.logger.info(f"保存済みモデルを読み込みました: {model_path}")
                self.logger.info(f"特徴量数: {len(self.feature_cols)}")
            else:
                self.logger.error("保存済みモデルが見つかりません。train_model_2020_2025.pyを実行してください。")
                raise FileNotFoundError("Model files not found")
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            raise
    
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
        # シミュレーションモードチェック
        if self.config.get('simulation_mode', False):
            await self._run_simulation()
            return
        
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
        """予測用データの準備（改良版）"""
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
        
        # レース情報の追加
        race_info = race_details.get('race_info', {})
        df['race_id'] = race_details['race_id']
        df['距離'] = race_info.get('distance', 2000)
        df['芝・ダート'] = 0 if race_info.get('surface', '芝') == '芝' else 1
        df['頭数'] = len(df)
        
        # 改良版モデルに必要な基本特徴量
        # 数値特徴量のデフォルト値
        numeric_defaults = {
            '体重': 480, '体重変化': 0, '斤量': 55, '上がり': 35.0,
            '出走頭数': len(df), '距離': 2000, 'クラス': 6, '性': 0,
            '芝・ダート': 0, '回り': 1, '馬場': 0, '天気': 1, '場id': 5,
            '枠番': 1, '馬番': 1
        }
        
        # 過去成績関連のデフォルト値
        for i in range(1, 6):
            numeric_defaults[f'着順{i}'] = 8
            numeric_defaults[f'距離{i}'] = 2000
            numeric_defaults[f'通過順{i}'] = 8
            numeric_defaults[f'走破時間{i}'] = 120
            numeric_defaults[f'オッズ{i}'] = 10
            numeric_defaults[f'騎手{i}'] = 0
            numeric_defaults[f'出走頭数{i}'] = 16
            numeric_defaults[f'上がり{i}'] = 35
            numeric_defaults[f'芝・ダート{i}'] = 0
            numeric_defaults[f'天気{i}'] = 1
            numeric_defaults[f'馬場{i}'] = 0
        
        # 騎手・調教師統計のデフォルト値
        jockey_defaults = {
            '騎手の勝率': 0.08, '騎手の複勝率': 0.25,
            '騎手の騎乗数': np.log1p(100), '騎手の平均着順': 8.0,
            '騎手のROI': 1.0, '騎手の勝率_30日': 0.08,
            '騎手の複勝率_30日': 0.25, '騎手の勝率_60日': 0.08,
            '騎手の連続不勝': 0, '騎手の最後勝利日数': np.exp(-30/30),
            '騎手の勝率_芝': 0.08, '騎手の勝率_ダート': 0.08,
            '騎手の勝率_短距離': 0.08, '騎手の勝率_中距離': 0.08,
            '騎手の勝率_長距離': 0.08, '騎手調教師相性': 0.08,
            '調教師の勝率': 0.08, '調教師の複勝率': 0.25
        }
        
        # その他の特徴量
        other_defaults = {
            '前走からの日数': 30, '放牧区分': 1,
            '平均中間日数': 30, '中間日数標準偏差': 0,
            '中間日数1': 30, '中間日数2': 30, '中間日数3': 30,
            '過去平均着順': 8, '過去最高着順': 3,
            '勝利経験': 0, '複勝経験': 3, '過去レース数': 10,
            '騎手の乗り替わり': 0
        }
        
        # 全てのデフォルト値を統合
        all_defaults = {**numeric_defaults, **jockey_defaults, **other_defaults}
        
        # モデルが期待する特徴量を確保
        for col in self.feature_cols:
            if col not in df.columns:
                if col in all_defaults:
                    df[col] = all_defaults[col]
                else:
                    df[col] = 0  # その他は0で埋める
        
        return df
    
    def _run_prediction(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """予測実行（改良版）"""
        try:
            # モデルチェック
            if self.model is None or self.feature_cols is None:
                self.logger.error("モデルが読み込まれていません")
                race_df['predicted_score'] = np.nan
                race_df['win_probability'] = np.nan
                return race_df
            
            # 特徴量の準備
            X = race_df[self.feature_cols].fillna(0)
            
            # 予測実行（LightGBMモデル）
            # predict_probaを使用して確率を取得
            try:
                predictions = self.model.predict_proba(X, num_iteration=self.model.best_iteration_)
                # 正例（3着以内）の確率を取得
                if len(predictions.shape) > 1:
                    predictions = predictions[:, 1]
            except:
                # predict_probaが使えない場合はpredictを使用
                predictions = self.model.predict(X, num_iteration=self.model.best_iteration_)
            
            # 予測結果の追加
            race_df['predicted_score'] = predictions
            
            # 複勝（3着以内）確率を勝率に変換
            # 18頭レースで3着以内に入る基準確率は3/18=0.167
            # これを基準に勝率を推定
            
            # まず予測値を調整（0-1の範囲に正規化）
            min_pred = predictions.min()
            max_pred = predictions.max()
            if max_pred > min_pred:
                normalized_preds = (predictions - min_pred) / (max_pred - min_pred)
            else:
                normalized_preds = np.ones(len(predictions)) / len(predictions)
            
            # 勝率への変換（複勝率から勝率を推定）
            # 上位馬ほど高い変換率を適用
            sorted_indices = np.argsort(-normalized_preds)
            win_probs = np.zeros(len(predictions))
            
            for i, idx in enumerate(sorted_indices):
                if i == 0:  # 最も高い予測値
                    win_probs[idx] = normalized_preds[idx] * 0.35
                elif i == 1:
                    win_probs[idx] = normalized_preds[idx] * 0.25
                elif i == 2:
                    win_probs[idx] = normalized_preds[idx] * 0.15
                elif i < 6:
                    win_probs[idx] = normalized_preds[idx] * 0.08
                else:
                    win_probs[idx] = normalized_preds[idx] * 0.02
            
            # 正規化して合計を1.0にする
            total_prob = win_probs.sum()
            if total_prob > 0:
                win_probs = win_probs / total_prob
            
            race_df['win_probability'] = win_probs
            
            self.logger.info(f"予測完了: {len(race_df)}頭")
            
            return race_df
            
        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            import traceback
            traceback.print_exc()
            race_df['predicted_score'] = np.nan
            race_df['win_probability'] = np.nan
            return race_df
    
    
    def _analyze_betting_opportunities(self, 
                                     predictions: pd.DataFrame,
                                     race_details: Dict) -> List[Dict]:
        """ベッティング機会の分析"""
        opportunities = []
        
        # 予測上位馬を取得（スコアが高い順）
        top_horses = predictions.nlargest(5, 'predicted_score')
        
        for _, horse in top_horses.iterrows():
            if pd.isna(horse['predicted_score']):
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
                    'predicted_score': horse['predicted_score']
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
    
    async def _run_simulation(self):
        """シミュレーションモード実行"""
        self.logger.info("シミュレーションモードで実行中...")
        
        # CSVファイルのリスト
        csv_files = self.config.get('simulation_files', [
            'live_race_data_202505021212.csv',
            'live_race_data_202505021211.csv'
        ])
        
        for csv_file in csv_files:
            if not Path(csv_file).exists():
                self.logger.warning(f"ファイルが見つかりません: {csv_file}")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"シミュレーション: {csv_file}")
            self.logger.info(f"{'='*60}")
            
            try:
                # CSVファイル読み込み
                race_df = pd.read_csv(csv_file)
                self.logger.info(f"レースデータ読み込み完了: {len(race_df)}頭")
                
                # レース情報を抽出
                race_info = {
                    'race_id': csv_file.replace('.csv', ''),
                    'date': race_df.get('date', ['2025年5月2日'])[0] if 'date' in race_df.columns else '2025年5月2日',
                    'racecourse': race_df.get('racecourse', ['東京'])[0] if 'racecourse' in race_df.columns else '東京',
                    'race_number': race_df.get('race_number', [11])[0] if 'race_number' in race_df.columns else 11,
                    'distance': race_df.get('distance', [2000])[0] if 'distance' in race_df.columns else 2000,
                    'surface': '芝'
                }
                
                # 予測用データ準備
                prediction_df = self._prepare_prediction_data_from_csv(race_df)
                
                # 予測実行
                predictions = self._run_prediction(prediction_df)
                
                # 結果表示
                self._display_simulation_results(predictions, race_info)
                
                # ベッティング機会の分析
                race_details = {'race_info': race_info}
                betting_opportunities = self._analyze_betting_opportunities(
                    predictions, race_details
                )
                
                if betting_opportunities:
                    await self._handle_betting_opportunities(
                        race_info, betting_opportunities
                    )
                
            except Exception as e:
                self.logger.error(f"シミュレーションエラー ({csv_file}): {e}")
                import traceback
                traceback.print_exc()
        
        self.logger.info("\nシミュレーション完了")
    
    def _prepare_prediction_data_from_csv(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """CSVファイルから予測用データを準備"""
        df = race_df.copy()
        
        # 必要なカラムのマッピング
        column_mapping = {
            '馬名': '馬名',
            '枠': '枠番',
            '馬番': '馬番',
            '騎手': '騎手',
            '斤量': '斤量',
            '馬体重': '体重',
            '馬体重変化': '体重変化',
            '単勝オッズ': 'オッズ'
        }
        
        # カラム名の統一
        df = df.rename(columns=column_mapping)
        
        # オッズがない場合はデフォルト値
        if 'オッズ' not in df.columns:
            df['オッズ'] = 10.0
        
        # race_idがあれば追加
        if 'race_id' in race_df.columns:
            df['race_id'] = race_df['race_id'].iloc[0]
        
        # 距離情報
        if 'distance' in race_df.columns:
            df['距離'] = race_df['distance'].iloc[0]
        else:
            df['距離'] = 2000
        
        # 標準的な予測データ準備（race_idを含む辞書を渡す）
        race_details = {
            'race_card': df,
            'race_id': df.get('race_id', 'simulation_race'),
            'race_info': {
                'distance': df.get('距離', 2000),
                'surface': '芝'
            }
        }
        return self._prepare_prediction_data(race_details)
    
    def _display_simulation_results(self, predictions: pd.DataFrame, race_info: Dict):
        """シミュレーション結果の表示"""
        self.logger.info(f"\n🏇 レース情報:")
        self.logger.info(f"   日付: {race_info['date']}")
        self.logger.info(f"   競馬場: {race_info['racecourse']}")
        self.logger.info(f"   レース番号: {race_info['race_number']}R")
        self.logger.info(f"   距離: {race_info['distance']}m")
        
        self.logger.info(f"\n🎯 予測結果:")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"{'順位':>4} {'馬番':>4} {'馬名':>20} {'オッズ':>8} {'勝率':>8} {'期待値':>8}")
        self.logger.info(f"{'='*80}")
        
        sorted_predictions = predictions.sort_values('win_probability', ascending=False)
        for i, (_, row) in enumerate(sorted_predictions.head(10).iterrows(), 1):
            self.logger.info(
                f"{i:4d}. {int(row['馬番']):3d}番 {row.get('馬名', 'Unknown'):>20s} "
                f"{row.get('オッズ', 0):7.1f}倍 {row['win_probability']*100:6.1f}% "
                f"{row.get('オッズ', 0) * row['win_probability']:7.2f}"
            )
        
        self.logger.info(f"\n📊 統計:")
        self.logger.info(f"   予測完了: {len(predictions)}頭")
        self.logger.info(f"   勝率合計: {predictions['win_probability'].sum()*100:.1f}%")
        self.logger.info(f"   最高勝率: {predictions['win_probability'].max()*100:.1f}%")
        self.logger.info(f"   最低勝率: {predictions['win_probability'].min()*100:.1f}%")
    
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
        'model_path': 'model_2020_2025/model_2020_2025.pkl',  # 2020-2025モデルを使用
        'enable_auto_betting': False,  # 安全のため手動モード
        'min_expected_value': 1.2,
        'kelly_fraction': 0.025,  # 2.5% Kelly
        'max_bet_per_race': 5000,
        'max_daily_loss': 30000,
        'simulation_mode': True,  # シミュレーションモードを有効化
        'simulation_files': [  # テストするCSVファイル
            'live_race_data_202505021212.csv',
            'live_race_data_202505021211.csv'
        ],
        'data_refresh_interval': 300  # デフォルト値
    }
    
    # システム起動
    system = IntegratedKeibaSystem(config)
    
    print("\n[モード: シミュレーション（保存済みCSVファイル使用）]")
    print(f"分析対象: {config['simulation_files']}")
    print("ベッティング機会はログに記録されます")
    print("\n")
    
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())