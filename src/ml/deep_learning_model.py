#!/usr/bin/env python3
"""
ディープラーニングモデル
パターン認識と複雑な相互作用の学習
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class HorseRacingDataset(Dataset):
    """競馬データ用のPyTorchデータセット"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class AttentionLayer(nn.Module):
    """アテンション機構"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return torch.sum(weights * x, dim=1)


class HorseRacingDNN(nn.Module):
    """競馬予測用ディープニューラルネットワーク"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.use_attention = use_attention
        
        # 入力層の正規化
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # アテンション層（オプション）
        if use_attention:
            self.attention = AttentionLayer(input_dim, hidden_dims[0] // 2)
            input_dim = input_dim  # アテンション出力は同じ次元
        
        # 隠れ層
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 出力層（着順予測）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        
        if self.use_attention:
            x = self.attention(x.unsqueeze(1)).squeeze(1)
        
        return self.network(x)


class TransformerRacePredictor(nn.Module):
    """Transformerベースの競馬予測モデル"""
    
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        # 入力を埋め込み空間に投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置エンコーディング（馬番順）
        self.positional_encoding = nn.Parameter(torch.randn(1, 18, d_model))  # 最大18頭
        
        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力層
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, n_horses, input_dim)
        batch_size, n_horses, _ = x.shape
        
        # 入力投影
        x = self.input_projection(x)
        
        # 位置エンコーディングを追加
        x = x + self.positional_encoding[:, :n_horses, :]
        
        # Transformer処理（seq_len, batch, features）の形式に変換
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        
        # 各馬の予測
        output = self.output_projection(x)
        
        return output.squeeze(-1)


class DeepLearningRacePredictor:
    """ディープラーニングを使用した競馬予測器"""
    
    def __init__(self, model_type: str = 'dnn', device: str = None):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def create_model(self, input_dim: int):
        """モデルを作成"""
        if self.model_type == 'dnn':
            self.model = HorseRacingDNN(
                input_dim=input_dim,
                hidden_dims=[512, 256, 128, 64],
                dropout_rate=0.3,
                use_attention=True
            )
        elif self.model_type == 'transformer':
            self.model = TransformerRacePredictor(
                input_dim=input_dim,
                d_model=256,
                nhead=8,
                num_layers=4,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
    def preprocess_data(self, X: np.ndarray, fit: bool = False):
        """データの前処理"""
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        return X_scaled
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, batch_size: int = 64, 
            learning_rate: float = 0.001, early_stopping_patience: int = 10):
        """モデルを訓練"""
        # データの前処理
        X_scaled = self.preprocess_data(X, fit=True)
        
        # モデル作成
        if self.model is None:
            self.create_model(X.shape[1])
        
        # データセットとデータローダー
        dataset = HorseRacingDataset(X_scaled, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 最適化とスケジューラ
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 損失関数（順位学習用）
        criterion = nn.MSELoss()
        
        # 訓練ループ
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training {self.model_type.upper()} model on {self.device}...")
        
        for epoch in range(epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0
            
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 検証フェーズ
            self.model.eval()
            val_loss = 0
            val_correlations = []
            
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    
                    # 順位相関を計算
                    pred_ranks = outputs.squeeze().cpu().numpy().argsort().argsort()
                    true_ranks = targets.cpu().numpy().argsort().argsort()
                    corr = np.corrcoef(pred_ranks, true_ranks)[0, 1]
                    val_correlations.append(corr)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_corr = np.mean(val_correlations)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Correlation: {avg_val_corr:.4f}")
            
            # 学習率調整
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # モデルを保存
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 最良のモデルをロード
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # データの前処理
        X_scaled = self.preprocess_data(X)
        
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(features)
            predictions = outputs.squeeze().cpu().numpy()
        
        return predictions
    
    def predict_race(self, race_features: np.ndarray) -> Dict:
        """レース単位での予測（各馬の順位と確率）"""
        predictions = self.predict(race_features)
        
        # 予測順位（小さいほど上位）
        predicted_ranks = predictions.argsort().argsort() + 1
        
        # 勝率の計算（ソフトマックス）
        exp_neg_pred = np.exp(-predictions * 2)  # 温度パラメータ
        win_probabilities = exp_neg_pred / exp_neg_pred.sum()
        
        # 連対率（2着以内）
        place_probabilities = []
        for i in range(len(predictions)):
            # 自分より良い予測値を持つ馬の数
            better_count = (predictions < predictions[i]).sum()
            # 2着以内に入る確率を近似
            place_prob = 1 / (1 + np.exp(better_count - 1))
            place_probabilities.append(place_prob)
        
        return {
            'predicted_ranks': predicted_ranks,
            'win_probabilities': win_probabilities,
            'place_probabilities': np.array(place_probabilities),
            'raw_predictions': predictions
        }