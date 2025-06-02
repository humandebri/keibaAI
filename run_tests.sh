#!/bin/bash
# テスト実行スクリプト

set -e

echo "🧪 競馬AI統一システム テスト実行"
echo "=================================="

# 仮想環境の有効化
if [ -d ".venv" ]; then
    echo "📦 仮想環境を有効化中..."
    source .venv/bin/activate
else
    echo "⚠️  警告: 仮想環境が見つかりません (.venv)"
fi

# Pythonパスの設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# テストの実行
echo "🔬 テスト実行中..."

# 全テスト実行
echo "📋 全テスト実行:"
python -m pytest tests/ -v

echo ""
echo "📊 カバレッジ付きテスト実行:"
if command -v coverage &> /dev/null; then
    coverage run -m pytest tests/
    coverage report -m
    coverage html
    echo "📈 カバレッジレポートが htmlcov/ に生成されました"
else
    echo "⚠️  coverage が見つかりません。pip install coverage でインストールしてください"
fi

echo ""
echo "🎯 特定テストの実行例:"
echo "  設定テスト:           python -m pytest tests/test_config.py -v"
echo "  特徴量テスト:         python -m pytest tests/test_features.py -v"
echo "  ユーティリティテスト:  python -m pytest tests/test_utils.py -v"
echo "  システムテスト:       python -m pytest tests/test_unified_system.py -v"
echo ""
echo "🏷️  マーカー別実行例:"
echo "  ユニットテストのみ:   python -m pytest -m unit"
echo "  遅いテストを除外:     python -m pytest -m 'not slow'"
echo ""

echo "✅ テスト実行完了"