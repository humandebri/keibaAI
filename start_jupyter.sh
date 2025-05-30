#!/bin/bash

# 仮想環境をアクティベート
source venv/bin/activate

# Jupyterをインストール（まだの場合）
pip install jupyter ipykernel

# 仮想環境をJupyterカーネルとして登録
python -m ipykernel install --user --name keiba_env --display-name "Keiba AI (venv)"

# Jupyter Notebookを起動
jupyter notebook

echo "Jupyter Notebookが起動しました。"
echo "カーネルメニューから 'Keiba AI (venv)' を選択してください。"