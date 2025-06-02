#!/usr/bin/env python3
"""
2020-2025年のデータをエンコードするスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_encoding_v2 import RaceDataEncoderV2

def main():
    """2020-2025年のデータをエンコード"""
    print("=" * 60)
    print("2020-2025年データのエンコーディング")
    print("=" * 60)
    
    # エンコーダー初期化
    encoder = RaceDataEncoderV2(config_dir="config", encoded_dir="encoded")
    
    # 2020-2025年のデータをエンコード
    output_path = encoder.encode_data(
        year_start=2020, 
        year_end=2025,
        data_dir="data"  # dataディレクトリからXLSXファイルを読む
    )
    
    print(f"\n✅ エンコード完了: {output_path}")

if __name__ == "__main__":
    main()