#!/bin/bash

# 仮想環境をアクティベート
source venv/bin/activate

# 必要なパッケージをインストール
pip install -r requirements.txt

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"