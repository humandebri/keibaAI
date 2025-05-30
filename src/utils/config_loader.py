"""
設定ファイル読み込みユーティリティ
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os


class ConfigLoader:
    """設定ファイルを読み込むクラス"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path)
        self._config = None
        
    def load(self) -> Dict[str, Any]:
        """
        設定ファイルを読み込む
        
        Returns:
            設定の辞書
        """
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
                
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得
        
        Args:
            key: ドット区切りのキー (例: "model.lgb_params.objective")
            default: デフォルト値
            
        Returns:
            設定値
        """
        config = self.load()
        
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_path(self, key: str) -> Path:
        """
        パス設定を取得してPathオブジェクトとして返す
        
        Args:
            key: パス設定のキー
            
        Returns:
            Pathオブジェクト
        """
        path_str = self.get(f"paths.{key}")
        if path_str:
            return Path(path_str)
        raise ValueError(f"Path not found in config: {key}")
        
    def ensure_directories(self):
        """設定ファイルに記載されているディレクトリを作成"""
        paths = self.get('paths', {})
        for key, path in paths.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)


# グローバルインスタンス
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """設定ローダーのインスタンスを取得"""
    return config