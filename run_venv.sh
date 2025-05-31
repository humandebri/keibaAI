#!/bin/bash
# 仮想環境で実行するためのスクリプト

# 仮想環境をアクティベート
source .venv/bin/activate

# 引数をそのまま渡して実行
python "$@"