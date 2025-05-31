# 🏇 Keiba AI クイックスタート

## 今すぐ始める（3ステップ）

### 1️⃣ 準備
```bash
cd Keiba_AI
source .venv/bin/activate  # 仮想環境を有効化
```

### 2️⃣ 実行
```bash
python main.py backtest
```

### 3️⃣ 完了！
結果が表示されます：
- 初期資金: ¥1,000,000
- 最終資金: ¥XXX,XXX
- リターン: XX%

---

## よくある質問

### Q: データがないと言われた
```bash
# データをダウンロード（2023年の例）
python src/data_processing/enhanced_scraping.py --year 2023
```

### Q: もっと詳しい設定をしたい
```bash
# 期待値を高く設定
python main.py backtest --min-ev 1.5

# 処理を速くしたい
python main.py backtest --no-trifecta
```

### Q: エラーが出た
```bash
# 依存関係を再インストール
pip install -r requirements.txt
```

---

## 次のステップ

詳しい使い方は [README.md](README.md) を参照してください。