#!/bin/bash
# 仮想環境のセットアップスクリプト

echo "=== Keiba AI 環境セットアップ ==="

# Python3の確認
echo "Pythonバージョンの確認..."
python3 --version

# 仮想環境の作成
echo "仮想環境を作成中..."
python3 -m venv .venv

# 仮想環境をアクティベート
echo "仮想環境をアクティベート中..."
source .venv/bin/activate

# pipのアップグレード
echo "pipをアップグレード中..."
python -m pip install --upgrade pip

# 必要なパッケージのインストール
echo "必要なパッケージをインストール中..."
pip install -r requirements.txt

echo ""
echo "✅ セットアップが完了しました！"
echo ""
echo "使い方:"
echo "1. 仮想環境をアクティベート: source .venv/bin/activate"
echo "2. バックテストを実行: python quickstart.py"
echo "3. 仮想環境を終了: deactivate"