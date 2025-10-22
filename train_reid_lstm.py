import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from collections import deque

class ReIDTrackingDataset(Dataset):
    """Re-ID用のトラッキングデータセット"""
    def __init__(self, data_dir, sequence_length=10, feature_dim=256):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.sequences = []
        self.motion_targets = []
        self.feature_targets = []
        self.confidence_targets = []
        
        # データディレクトリから時系列データを読み込む
        for track_file in os.listdir(data_dir):
            if track_file.endswith('.npy'):
                track_data = np.load(os.path.join(data_dir, track_file))
                
                # 十分なデータがある場合のみ使用
                if len(track_data) >= sequence_length + 1:
                    for i in range(len(track_data) - sequence_length):
                        sequence = track_data[i:i+sequence_length]
                        next_position = track_data[i+sequence_length]
                        
                        # 移動予測ターゲット
                        motion_target = next_position[:4]  # [x, y, w, h]
                        
                        # Re-ID特徴量ターゲット（移動パターンから生成）
                        feature_target = self.generate_feature_target(sequence, next_position)
                        
                        # 信頼度ターゲット（移動の一貫性から計算）
                        confidence_target = self.calculate_confidence_target(sequence, next_position)
                        
                        self.sequences.append(sequence)
                        self.motion_targets.append(motion_target)
                        self.feature_targets.append(feature_target)
                        self.confidence_targets.append(confidence_target)
    
    def generate_feature_target(self, sequence, next_position):
        """移動パターンからRe-ID特徴量を生成"""
        # 移動ベクトルを計算
        movements = []
        for i in range(1, len(sequence)):
            dx = sequence[i][0] - sequence[i-1][0]
            dy = sequence[i][1] - sequence[i-1][1]
            movements.append([dx, dy])
        
        # 移動パターンの統計特徴
        movements = np.array(movements)
        mean_movement = np.mean(movements, axis=0)
        std_movement = np.std(movements, axis=0)
        
        # 方向の一貫性
        directions = np.arctan2(movements[:, 1], movements[:, 0])
        direction_consistency = 1.0 - np.std(directions) / np.pi
        
        # 速度の一貫性
        speeds = np.linalg.norm(movements, axis=1)
        speed_consistency = 1.0 - (np.std(speeds) / (np.mean(speeds) + 1e-8))
        
        # 256次元の特徴量ベクトルを生成
        feature_vector = np.zeros(self.feature_dim)
        
        # 基本統計特徴（最初の20次元）
        feature_vector[:2] = mean_movement
        feature_vector[2:4] = std_movement
        feature_vector[4] = direction_consistency
        feature_vector[5] = speed_consistency
        
        # 移動パターンの周期性特徴（6-50次元）
        if len(movements) >= 4:
            # FFTで周期性を検出
            fft_x = np.fft.fft(movements[:, 0])
            fft_y = np.fft.fft(movements[:, 1])
            
            # 主要周波数成分を特徴量に追加
            feature_vector[6:26] = np.abs(fft_x[:20])
            feature_vector[26:46] = np.abs(fft_y[:20])
        
        # 残りの次元はランダムノイズ（実際の実装ではより意味のある特徴を使用）
        feature_vector[46:] = np.random.normal(0, 0.1, self.feature_dim - 46)
        
        return feature_vector
    
    def calculate_confidence_target(self, sequence, next_position):
        """移動の一貫性から信頼度を計算"""
        if len(sequence) < 3:
            return 0.5
        
        # 移動の一貫性を計算
        movements = []
        for i in range(1, len(sequence)):
            dx = sequence[i][0] - sequence[i-1][0]
            dy = sequence[i][1] - sequence[i-1][1]
            movements.append(np.sqrt(dx**2 + dy**2))
        
        # 移動の標準偏差が小さいほど信頼度が高い
        movement_std = np.std(movements)
        movement_mean = np.mean(movements)
        
        if movement_mean == 0:
            return 1.0
        
        consistency = max(0, 1 - (movement_std / movement_mean))
        return min(1.0, consistency)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.motion_targets[idx]),
            torch.FloatTensor(self.feature_targets[idx]),
            torch.FloatTensor([self.confidence_targets[idx]])
        )

class LSTMMotionPredictor(nn.Module):
    """LSTMベースの移動予測と特徴抽出モデル"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, feature_dim=256):
        super(LSTMMotionPredictor, self).__init__()
        
        # LSTM層（移動パターンの学習）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 移動予測ヘッド
        self.motion_head = nn.Linear(hidden_size, 4)  # [x, y, w, h]の予測
        
        # 特徴抽出ヘッド（Re-ID用）
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 信頼度予測ヘッド
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, 4] - [x, y, w, h]の時系列データ
        Returns:
            dict: {
                'motion_pred': [batch_size, 4] - 次の位置予測,
                'features': [batch_size, feature_dim] - Re-ID特徴量,
                'confidence': [batch_size, 1] - 予測信頼度
            }
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # 最後のタイムステップの出力
        
        # 各ヘッドで予測
        motion_pred = self.motion_head(last_output)
        features = self.feature_head(last_output)
        confidence = self.confidence_head(last_output)
        
        return {
            'motion_pred': motion_pred,
            'features': features,
            'confidence': confidence,
            'hidden_state': hidden,
            'cell_state': cell
        }

def train_reid_lstm_model(train_data_dir, val_data_dir, model_save_path="best_reid_lstm_model.pth"):
    """Re-ID用LSTMモデルの学習"""
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセットの準備
    print("学習データを読み込み中...")
    train_dataset = ReIDTrackingDataset(train_data_dir)
    print(f"学習サンプル数: {len(train_dataset)}")
    
    print("検証データを読み込み中...")
    val_dataset = ReIDTrackingDataset(val_data_dir)
    print(f"検証サンプル数: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("エラー: 学習または検証データが見つかりません！")
        return
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # モデルの初期化
    model = LSTMMotionPredictor().to(device)
    
    # 損失関数とオプティマイザ
    motion_criterion = nn.MSELoss()
    feature_criterion = nn.MSELoss()
    confidence_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # 最良の検証損失を記録
    best_val_loss = float('inf')
    
    print("\n学習開始...")
    epochs = 100
    
    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_motion_loss = 0.0
        train_feature_loss = 0.0
        train_confidence_loss = 0.0
        
        for batch_idx, (sequences, motion_targets, feature_targets, confidence_targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            motion_targets = motion_targets.to(device)
            feature_targets = feature_targets.to(device)
            confidence_targets = confidence_targets.to(device)
            
            optimizer.zero_grad()
            
            # モデルの出力
            outputs = model(sequences)
            
            # 各損失を計算
            motion_loss = motion_criterion(outputs['motion_pred'], motion_targets)
            feature_loss = feature_criterion(outputs['features'], feature_targets)
            confidence_loss = confidence_criterion(outputs['confidence'], confidence_targets)
            
            # 総合損失（重み付き）
            total_loss = motion_loss + 0.5 * feature_loss + 0.3 * confidence_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_motion_loss += motion_loss.item()
            train_feature_loss += feature_loss.item()
            train_confidence_loss += confidence_loss.item()
            
            # バッチごとの進捗表示
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Motion: {motion_loss.item():.4f}, "
                      f"Feature: {feature_loss.item():.4f}, "
                      f"Confidence: {confidence_loss.item():.4f}")
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_motion_loss = 0.0
        val_feature_loss = 0.0
        val_confidence_loss = 0.0
        
        with torch.no_grad():
            for sequences, motion_targets, feature_targets, confidence_targets in val_loader:
                sequences = sequences.to(device)
                motion_targets = motion_targets.to(device)
                feature_targets = feature_targets.to(device)
                confidence_targets = confidence_targets.to(device)
                
                outputs = model(sequences)
                
                motion_loss = motion_criterion(outputs['motion_pred'], motion_targets)
                feature_loss = feature_criterion(outputs['features'], feature_targets)
                confidence_loss = confidence_criterion(outputs['confidence'], confidence_targets)
                
                total_loss = motion_loss + 0.5 * feature_loss + 0.3 * confidence_loss
                
                val_loss += total_loss.item()
                val_motion_loss += motion_loss.item()
                val_feature_loss += feature_loss.item()
                val_confidence_loss += confidence_loss.item()
        
        # エポックごとの結果を表示
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"学習 - Total: {avg_train_loss:.4f}, Motion: {train_motion_loss/len(train_loader):.4f}, "
              f"Feature: {train_feature_loss/len(train_loader):.4f}, Confidence: {train_confidence_loss/len(train_loader):.4f}")
        print(f"検証 - Total: {avg_val_loss:.4f}, Motion: {val_motion_loss/len(val_loader):.4f}, "
              f"Feature: {val_feature_loss/len(val_loader):.4f}, Confidence: {val_confidence_loss/len(val_loader):.4f}")
        
        # 学習率スケジューリング
        scheduler.step(avg_val_loss)
        
        # モデルの保存（検証損失が改善した場合）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"モデルを保存しました！新しい最良検証損失: {best_val_loss:.4f}")
        
        print("-" * 80)
    
    print(f"\n学習完了！")
    print(f"最良検証損失: {best_val_loss:.4f}")
    print(f"モデル保存先: {model_save_path}")

if __name__ == "__main__":
    # データディレクトリのパス
    train_data_dir = "processed_datasets/train"  # 学習データディレクトリ
    val_data_dir = "processed_datasets/val"      # 検証データディレクトリ
    
    # Re-ID用LSTMモデルの学習
    train_reid_lstm_model(train_data_dir, val_data_dir)
