# 🏇 JRA競馬予測システム - リアルタイムデータ取得＆自動投票 実装計画

## 📊 現状と目標

### 現在の達成事項
- ✅ 予測モデル構築完了（ROI: 27.75）
- ✅ バックテストシステム完成
- ✅ 高度な特徴量エンジニアリング実装

### 今回の目標
- 🎯 リアルタイムレース情報の取得
- 🎯 自動投票機能の実装（手動確認付き）
- 🎯 無料データソースの活用

## 🚀 段階的実装計画

### Phase 1: 基礎インフラ構築（1週目）

#### 1.1 データ収集基盤
```python
# 実装するクラス
- JRARealTimeSystem: メインシステム
- NetkeibaDataCollector: netkeiba.com連携
- DataCache: キャッシュ管理
- RateLimiter: レート制限管理
```

**タスク:**
- [ ] robots.txt遵守システムの実装
- [ ] レート制限機能（1-3秒の遅延）
- [ ] エラーハンドリングとリトライ機構
- [ ] キャッシュシステム（Redis推奨）

#### 1.2 無料データソースの調査と実装

**利用可能な無料データソース:**

1. **JRA公式サイト (jra.go.jp)**
   - 出馬表
   - レース結果
   - 基本的なオッズ情報
   - 制限: 詳細データなし、レート制限あり

2. **netkeiba.com（準公式）**
   - より詳細なデータ
   - 過去データアクセス
   - エンコーディング: EUC-JP

3. **keibalab.jp**
   - 予想データ
   - コラム・分析

### Phase 2: リアルタイムデータ収集（2週目）

#### 2.1 実装する機能

```python
class RealTimeDataPipeline:
    """リアルタイムデータパイプライン"""
    
    def __init__(self):
        self.collectors = {
            'jra': JRARealTimeSystem(),
            'netkeiba': NetkeibaDataCollector(),
        }
        self.predictor = load_trained_model()
    
    async def run_pipeline(self):
        # 1. 本日のレース一覧取得
        races = await self.get_today_races()
        
        # 2. 各レースの詳細情報収集
        for race in races:
            race_data = await self.collect_race_data(race)
            
            # 3. 予測実行
            prediction = self.predictor.predict(race_data)
            
            # 4. 投票判断
            if self.should_bet(prediction):
                await self.prepare_bet(race, prediction)
```

#### 2.2 データ統合とフォーマット

```python
# 統一データフォーマット
RaceData = {
    'race_id': str,
    'datetime': datetime,
    'racecourse': str,
    'race_number': int,
    'distance': int,
    'surface': str,  # 芝/ダート
    'horses': [
        {
            'number': int,
            'name': str,
            'jockey': str,
            'weight': float,
            'odds': float,
            'popularity': int
        }
    ],
    'track_condition': str,
    'weather': str
}
```

### Phase 3: 自動投票システム（3-4週目）

#### 3.1 IPAT連携の実装

```python
class SafeBettingSystem:
    """安全な自動投票システム"""
    
    def __init__(self, ipat_credentials):
        self.ipat = JRAIPATInterface(**ipat_credentials)
        self.manual_confirmation_queue = []
        self.bet_limits = {
            'max_bet_per_race': 10000,
            'max_daily_loss': 50000,
            'kelly_fraction': 0.1
        }
    
    def calculate_bet_size(self, prediction, bankroll):
        """Kelly基準によるベットサイズ計算"""
        edge = prediction['expected_value'] - 1.0
        odds = prediction['odds']
        
        kelly = edge / (odds - 1)
        safe_kelly = kelly * self.bet_limits['kelly_fraction']
        
        bet_size = min(
            bankroll * safe_kelly,
            self.bet_limits['max_bet_per_race']
        )
        
        return max(100, int(bet_size / 100) * 100)
```

#### 3.2 手動確認フロー

```python
class ManualConfirmationSystem:
    """手動確認システム"""
    
    def __init__(self):
        self.pending_bets = []
        self.confirmed_bets = []
    
    def add_to_confirmation_queue(self, bet_info):
        """確認待ちキューに追加"""
        self.pending_bets.append({
            'timestamp': datetime.now(),
            'bet_info': bet_info,
            'status': 'pending',
            'confirmation_deadline': datetime.now() + timedelta(minutes=5)
        })
        
        # 通知送信
        self.send_confirmation_alert(bet_info)
    
    def send_confirmation_alert(self, bet_info):
        """確認アラート送信（メール/LINE/Slack等）"""
        message = f"""
        【投票確認依頼】
        レース: {bet_info['race_name']}
        式別: {bet_info['bet_type']}
        選択: {bet_info['selections']}
        金額: ¥{bet_info['amount']:,}
        期待値: {bet_info['expected_value']:.2f}
        
        5分以内に確認してください。
        """
        # 実装: メール/LINE/Slack通知
```

### Phase 4: 統合とモニタリング（5週目）

#### 4.1 完全統合システム

```python
class JRAAutomatedBettingSystem:
    """完全自動化競馬投票システム"""
    
    def __init__(self, config):
        self.data_pipeline = RealTimeDataPipeline()
        self.predictor = ImprovedKeibaAISystem()
        self.betting_system = SafeBettingSystem(config['ipat'])
        self.monitor = SystemMonitor()
        
    async def run(self):
        """メインループ"""
        while True:
            try:
                # 1. データ収集
                races = await self.data_pipeline.get_upcoming_races()
                
                # 2. 各レースを処理
                for race in races:
                    await self.process_race(race)
                
                # 3. モニタリング
                self.monitor.check_system_health()
                
                # 4. 次のサイクルまで待機
                await asyncio.sleep(300)  # 5分間隔
                
            except Exception as e:
                self.logger.error(f"システムエラー: {e}")
                await self.handle_error(e)
```

## 🛠️ 技術的実装詳細

### 1. スクレイピング実装のベストプラクティス

```python
class RobustScraper:
    """堅牢なスクレイピング実装"""
    
    def __init__(self):
        self.session = self._create_session()
        self.rate_limiter = RateLimiter(
            calls=60,
            period=60,  # 1分間に60回まで
            peak_hours_limit=30  # ピーク時は半分
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=4, max=60),
        retry=retry_if_exception_type(requests.RequestException)
    )
    async def fetch_with_retry(self, url):
        """リトライ機能付きフェッチ"""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(url) as response:
                if response.status == 429:
                    # レート制限に達した場合
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    raise RateLimitError()
                
                response.raise_for_status()
                return await response.text()
                
        except Exception as e:
            self.logger.error(f"Fetch error: {url} - {e}")
            raise
```

### 2. データベース設計

```sql
-- レース情報
CREATE TABLE races (
    race_id VARCHAR(20) PRIMARY KEY,
    race_date DATE NOT NULL,
    racecourse VARCHAR(20),
    race_number INT,
    race_name VARCHAR(100),
    distance INT,
    surface VARCHAR(10),
    grade VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- リアルタイムオッズ
CREATE TABLE odds_history (
    id SERIAL PRIMARY KEY,
    race_id VARCHAR(20) REFERENCES races(race_id),
    horse_number INT,
    odds_win DECIMAL(5,1),
    odds_place DECIMAL(5,1),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_race_time (race_id, recorded_at)
);

-- 投票履歴
CREATE TABLE betting_history (
    bet_id SERIAL PRIMARY KEY,
    race_id VARCHAR(20) REFERENCES races(race_id),
    bet_type VARCHAR(20),
    selections JSON,
    amount INT,
    expected_value DECIMAL(5,3),
    actual_result VARCHAR(10),
    profit_loss INT,
    confirmed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. 非同期処理アーキテクチャ

```python
import asyncio
from asyncio import Queue
import aioredis

class AsyncDataProcessor:
    """非同期データ処理システム"""
    
    def __init__(self):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        
    async def start_workers(self, num_workers=5):
        """ワーカープロセス起動"""
        for i in range(num_workers):
            worker = asyncio.create_task(self.worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def worker(self, name):
        """ワーカープロセス"""
        while True:
            try:
                task = await self.task_queue.get()
                result = await self.process_task(task)
                await self.result_queue.put(result)
            except Exception as e:
                self.logger.error(f"{name} error: {e}")
```

## 📋 実装チェックリスト

### Week 1: 基礎構築
- [ ] プロジェクト構造の整理
- [ ] 基本的なスクレイピング機能
- [ ] レート制限とrobots.txt対応
- [ ] エラーハンドリング
- [ ] ログシステム

### Week 2: データ収集
- [ ] JRA公式サイト連携
- [ ] netkeiba.com連携  
- [ ] データ統合パイプライン
- [ ] キャッシュシステム
- [ ] リアルタイムモニタリング

### Week 3: 予測統合
- [ ] 既存モデルとの連携
- [ ] リアルタイムデータでの予測
- [ ] ベッティング戦略の実装
- [ ] リスク管理機能

### Week 4: 投票システム
- [ ] IPAT基本連携
- [ ] 手動確認フロー
- [ ] 通知システム
- [ ] 投票履歴管理

### Week 5: 統合とテスト
- [ ] 全体統合テスト
- [ ] パフォーマンス最適化
- [ ] モニタリングダッシュボード
- [ ] ドキュメント作成

## ⚠️ 重要な注意事項

### 法的コンプライアンス
1. **利用規約の遵守** - 各サイトの利用規約を必ず確認
2. **レート制限** - 過度なアクセスは避ける
3. **手動確認義務** - 自動投票は必ず手動で確認
4. **責任の所在** - システムエラーも自己責任

### リスク管理
1. **資金管理** - Kelly基準の保守的運用
2. **損失制限** - 日次・月次の損失上限設定
3. **システム監視** - 24/7モニタリング
4. **バックアップ** - データと設定の定期バックアップ

### 推奨事項
1. **段階的導入** - 小額から開始
2. **ペーパートレード** - 実資金投入前のテスト
3. **継続的改善** - データ収集と分析の継続
4. **コミュニティ** - 他の開発者との情報交換

## 🎯 次のステップ

1. **環境構築**
   ```bash
   pip install -r requirements.txt
   python setup_database.py
   ```

2. **基本機能テスト**
   ```bash
   python jra_realtime_system.py
   ```

3. **統合テスト**
   ```bash
   python test_integration.py
   ```

## 💰 有料API検討

現時点では無料データソースで基本機能は実装可能ですが、以下の場合は有料APIを検討：

### JRA-VAN Data Lab（月額2,090円）を検討すべき場合
- より詳細な過去データが必要
- 調教データ・血統データが必要  
- 商用利用を検討
- より高い信頼性が必要

### メリット
- 公式データの信頼性
- 包括的なデータセット
- 安定したAPI
- サポート体制

現段階では無料データソースで十分ですが、本格運用時は検討価値があります。