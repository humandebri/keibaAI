#!/usr/bin/env python3
"""
LightGBMのlibomp問題を修正するスクリプト
"""
import os
import subprocess
import sys

def fix_lightgbm():
    """LightGBMの問題を修正"""
    
    # Homebrewでlibompをインストール
    print("Checking for libomp...")
    result = subprocess.run(['brew', 'list', 'libomp'], capture_output=True)
    if result.returncode != 0:
        print("Installing libomp...")
        subprocess.run(['brew', 'install', 'libomp'])
    
    # 環境変数を設定
    os.environ['LDFLAGS'] = "-L/opt/homebrew/opt/libomp/lib"
    os.environ['CPPFLAGS'] = "-I/opt/homebrew/opt/libomp/include"
    
    # LightGBMをアンインストール
    print("Uninstalling lightgbm...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'lightgbm'])
    
    # 環境変数を設定してLightGBMを再インストール
    print("Reinstalling lightgbm with proper configuration...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'lightgbm', '--no-cache-dir',
        '--config-settings=cmake.define.USE_OPENMP=OFF'
    ])
    
    print("\nDone! Try importing lightgbm again.")
    
    # テスト
    try:
        import lightgbm
        print("✓ lightgbm imported successfully!")
    except Exception as e:
        print(f"✗ Still having issues: {e}")
        print("\nTry running this in your notebook:")
        print("import os")
        print("os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'")

if __name__ == "__main__":
    fix_lightgbm()