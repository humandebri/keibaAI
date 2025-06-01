#!/bin/bash
# 最適化戦略を実行するスクリプト

echo "競馬AI最適化戦略を開始します..."
echo "================================"

# Python仮想環境をアクティブ化
if [ -d ".venv" ]; then
    echo "仮想環境 .venv をアクティブ化中..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "仮想環境 venv をアクティブ化中..."
    source venv/bin/activate
else
    echo "エラー: 仮想環境が見つかりません"
    echo "以下のコマンドで仮想環境を作成してください:"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install pandas numpy lightgbm optuna scikit-learn matplotlib"
    exit 1
fi

# 必要なパッケージの確認
echo "必要なパッケージを確認中..."
python -c "import pandas, numpy, lightgbm, optuna" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "必要なパッケージが不足しています。インストール中..."
    pip install pandas numpy lightgbm optuna scikit-learn matplotlib
fi

# 最適化戦略を実行
echo ""
echo "最適化戦略を実行中..."
echo "================================"
python optimized_strategy.py

echo ""
echo "実行完了！"
echo "結果は以下のディレクトリに保存されています:"
echo "- backtest_optimized_iter1/"
echo "- backtest_optimized_iter2/"
echo "- backtest_optimized_iter3/"
echo "- backtest_optimized_iter4/"
echo "- backtest_optimized_iter5/"
echo "- best_optimization_result.json"