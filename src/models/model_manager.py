"""
モデルバージョニングとモデル管理
"""
import json
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import shutil
import logging

from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ModelManager:
    """モデルのバージョン管理とライフサイクル管理"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        初期化
        
        Args:
            model_dir: モデル保存ディレクトリ
        """
        self.config = get_config()
        self.model_dir = model_dir or self.config.get_path('model_dir')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / 'model_metadata.json'
        self._load_metadata()
        
    def _load_metadata(self):
        """メタデータを読み込む"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'models': {},
                'current_version': None
            }
            
    def _save_metadata(self):
        """メタデータを保存"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
    def save_model(self, model: Any, 
                  model_type: str,
                  metrics: Dict[str, float],
                  features: List[str],
                  params: Dict[str, Any],
                  additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        モデルを保存
        
        Args:
            model: 保存するモデルオブジェクト
            model_type: モデルタイプ（'lightgbm', 'xgboost'等）
            metrics: 評価メトリクス
            features: 使用した特徴量リスト
            params: モデルパラメータ
            additional_info: 追加情報
            
        Returns:
            バージョンID
        """
        # バージョンID生成
        timestamp = datetime.now()
        version_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # モデルファイルパス
        model_filename = f"{self.config.get('versioning.model_prefix')}_{version_id}.pkl"
        model_path = self.model_dir / model_filename
        
        # モデル保存
        if model_type == 'lightgbm':
            # LightGBM専用の保存方法
            import lightgbm as lgb
            if hasattr(model, 'save_model'):
                model.save_model(str(model_path.with_suffix('.txt')))
            else:
                joblib.dump(model, model_path)
        else:
            # 汎用的な保存方法
            joblib.dump(model, model_path)
            
        # メタデータ作成
        model_info = {
            'version_id': version_id,
            'model_type': model_type,
            'timestamp': timestamp.isoformat(),
            'file_path': str(model_path),
            'metrics': metrics,
            'features': features,
            'feature_count': len(features),
            'params': params,
            'additional_info': additional_info or {}
        }
        
        # メタデータ更新
        self.metadata['models'][version_id] = model_info
        self.metadata['current_version'] = version_id
        self._save_metadata()
        
        # 古いモデルの削除
        self._cleanup_old_models()
        
        logger.info(f"Model saved: version={version_id}, metrics={metrics}")
        
        return version_id
        
    def load_model(self, version_id: Optional[str] = None) -> tuple:
        """
        モデルを読み込む
        
        Args:
            version_id: バージョンID（Noneの場合は最新版）
            
        Returns:
            (model, model_info) のタプル
        """
        if version_id is None:
            version_id = self.metadata.get('current_version')
            if version_id is None:
                raise ValueError("No models found")
                
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version not found: {version_id}")
            
        model_info = self.metadata['models'][version_id]
        model_path = Path(model_info['file_path'])
        
        if not model_path.exists():
            # .txtファイルを探す（LightGBM）
            txt_path = model_path.with_suffix('.txt')
            if txt_path.exists():
                model_path = txt_path
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # モデル読み込み
        if model_info['model_type'] == 'lightgbm' and model_path.suffix == '.txt':
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
        else:
            model = joblib.load(model_path)
            
        logger.info(f"Model loaded: version={version_id}")
        
        return model, model_info
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        保存されているモデルのリストを取得
        
        Returns:
            モデル情報のリスト
        """
        models = []
        for version_id, info in self.metadata['models'].items():
            models.append({
                'version_id': version_id,
                'timestamp': info['timestamp'],
                'model_type': info['model_type'],
                'metrics': info['metrics'],
                'is_current': version_id == self.metadata.get('current_version')
            })
            
        # 新しい順にソート
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return models
        
    def get_model_info(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        モデルの詳細情報を取得
        
        Args:
            version_id: バージョンID（Noneの場合は最新版）
            
        Returns:
            モデル情報
        """
        if version_id is None:
            version_id = self.metadata.get('current_version')
            
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version not found: {version_id}")
            
        return self.metadata['models'][version_id]
        
    def compare_models(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        2つのモデルを比較
        
        Args:
            version_id1: 比較対象1のバージョンID
            version_id2: 比較対象2のバージョンID
            
        Returns:
            比較結果
        """
        info1 = self.get_model_info(version_id1)
        info2 = self.get_model_info(version_id2)
        
        comparison = {
            'version_comparison': {
                'version1': version_id1,
                'version2': version_id2,
                'timestamp1': info1['timestamp'],
                'timestamp2': info2['timestamp']
            },
            'metrics_comparison': {}
        }
        
        # メトリクスの比較
        all_metrics = set(info1['metrics'].keys()) | set(info2['metrics'].keys())
        for metric in all_metrics:
            val1 = info1['metrics'].get(metric)
            val2 = info2['metrics'].get(metric)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                improvement = (diff / val1 * 100) if val1 != 0 else 0
                comparison['metrics_comparison'][metric] = {
                    'value1': val1,
                    'value2': val2,
                    'difference': diff,
                    'improvement_percent': improvement
                }
                
        # 特徴量の比較
        features1 = set(info1['features'])
        features2 = set(info2['features'])
        comparison['features_comparison'] = {
            'count1': len(features1),
            'count2': len(features2),
            'added': list(features2 - features1),
            'removed': list(features1 - features2),
            'common': len(features1 & features2)
        }
        
        return comparison
        
    def set_current_version(self, version_id: str):
        """
        現在のバージョンを設定
        
        Args:
            version_id: バージョンID
        """
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version not found: {version_id}")
            
        self.metadata['current_version'] = version_id
        self._save_metadata()
        
        logger.info(f"Current model version set to: {version_id}")
        
    def delete_model(self, version_id: str):
        """
        モデルを削除
        
        Args:
            version_id: 削除するバージョンID
        """
        if version_id not in self.metadata['models']:
            raise ValueError(f"Model version not found: {version_id}")
            
        if version_id == self.metadata.get('current_version'):
            raise ValueError("Cannot delete current version")
            
        # ファイル削除
        model_info = self.metadata['models'][version_id]
        model_path = Path(model_info['file_path'])
        
        if model_path.exists():
            model_path.unlink()
            
        # .txtファイルも削除（LightGBM）
        txt_path = model_path.with_suffix('.txt')
        if txt_path.exists():
            txt_path.unlink()
            
        # メタデータから削除
        del self.metadata['models'][version_id]
        self._save_metadata()
        
        logger.info(f"Model deleted: {version_id}")
        
    def _cleanup_old_models(self):
        """古いモデルを削除"""
        keep_n = self.config.get('versioning.keep_n_models', 5)
        
        if len(self.metadata['models']) <= keep_n:
            return
            
        # バージョンIDをタイムスタンプ順にソート
        versions = sorted(
            self.metadata['models'].keys(),
            key=lambda v: self.metadata['models'][v]['timestamp'],
            reverse=True
        )
        
        # 現在のバージョンは保持
        current_version = self.metadata.get('current_version')
        
        # 削除対象を決定
        to_delete = []
        kept_count = 0
        
        for version in versions:
            if version == current_version or kept_count < keep_n:
                kept_count += 1
            else:
                to_delete.append(version)
                
        # 削除実行
        for version in to_delete:
            try:
                self.delete_model(version)
            except Exception as e:
                logger.error(f"Failed to delete model {version}: {e}")
                
    def export_model(self, version_id: Optional[str] = None, 
                    export_path: Path = None) -> Path:
        """
        モデルをエクスポート
        
        Args:
            version_id: バージョンID
            export_path: エクスポート先パス
            
        Returns:
            エクスポートしたファイルパス
        """
        model, info = self.load_model(version_id)
        
        if export_path is None:
            export_dir = Path('exported_models')
            export_dir.mkdir(exist_ok=True)
            export_path = export_dir / f"model_{info['version_id']}.zip"
            
        # モデルと情報をまとめて保存
        import zipfile
        with zipfile.ZipFile(export_path, 'w') as zf:
            # モデルファイル
            model_path = Path(info['file_path'])
            zf.write(model_path, f"model{model_path.suffix}")
            
            # メタデータ
            metadata_json = json.dumps(info, indent=2, ensure_ascii=False)
            zf.writestr('metadata.json', metadata_json)
            
        logger.info(f"Model exported to: {export_path}")
        
        return export_path