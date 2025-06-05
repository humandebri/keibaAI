#!/bin/bash
# 視覚的バックテストの実行スクリプト

echo "=== 競馬AI 視覚的バックテスト ==="
echo ""
echo "利用可能なデータ:"
echo "- 2022年: 払戻データ付き"
echo "- 2024年: 払戻データ付き"
echo "- 2025年: 払戻データ付き"
echo ""

# デフォルト設定で実行
echo "1. デフォルト設定で実行（2022年訓練、2024-2025年テスト）"
python enhanced_visual_backtest.py

# カスタム設定の例
echo ""
echo "2. カスタム設定の例:"

# 2022年で訓練、2024年でテスト
echo "   - 2022年訓練、2024年テスト:"
echo "     python enhanced_visual_backtest.py --train-years 2022 --test-years 2024"

# 2022年と2024年で訓練、2025年でテスト
echo "   - 2022,2024年訓練、2025年テスト:"
echo "     python enhanced_visual_backtest.py --train-years 2022 2024 --test-years 2025"

# 高い期待値閾値
echo "   - 期待値1.2以上のみ:"
echo "     python enhanced_visual_backtest.py --min-ev 1.2"

# main.pyを使用
echo ""
echo "3. main.pyを使用する場合:"
echo "   python main.py backtest --use-payout-data --train-years 2022 --test-years 2024 2025"