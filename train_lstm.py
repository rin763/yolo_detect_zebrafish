import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
from torch.utils.data import Dataset, DataLoader
import os
from object_tracking import LSTMTracker, ObjectTracker

class TrackingDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # データディレクトリから時系列データを読み込む
        for track_file in os.listdir(data_dir):
            if track_file.endswith('.npy'):
                track_data = np.load(os.path.join(data_dir, track_file))
                
                # シーケンスとターゲットを作成
                for i in range(len(track_data) - sequence_length):
                    sequence = track_data[i:i+sequence_length]
                    target = track_data[i+sequence_length]
                    self.sequences.append(sequence)
                    self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def collect_training_data(video_paths, output_dir, model_path):
    """教師データを収集する関数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # トラッカーの初期化
    tracker = ObjectTracker(model_path)
    
    # 各動画からデータを収集
    for i, video_path in enumerate(video_paths):
        print(f"\nCollecting data from video {i+1}/{len(video_paths)}: {video_path}")
        tracker.collect_training_data(video_path, output_dir)
    
    print("\nData collection completed!")

def train_lstm_model(train_data_dir, val_data_dir, model_save_path, batch_size=32, epochs=100, learning_rate=0.001):
    """LSTMモデルをトレーニングする関数"""
    # モデルの初期化
    model = LSTMTracker()
    
    # データセットの準備
    print("Loading training data...")
    train_dataset = TrackingDataset(train_data_dir)
    print(f"Training samples: {len(train_dataset)}")
    
    print("Loading validation data...")
    val_dataset = TrackingDataset(val_data_dir)
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: No training or validation data found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 損失関数とオプティマイザの設定
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 最良の検証損失を記録
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    # トレーニングループ
    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            
            # バッチごとの進捗表示
            if train_batches % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {train_batches}, Loss: {loss.item():.4f}")
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        # エポックごとの結果を表示
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # モデルの保存（検証損失が改善した場合）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved! New best validation loss: {best_val_loss:.4f}")
        
        print("-" * 50)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # パスの設定
    video_paths = [
        "./video/training/30min/output_left.mp4",
        "./video/training/30min/output_right.mp4"
    ]
    model_path = "./train_results/weights/best.pt"
    train_data_dir = "train_data"
    val_data_dir = "val_data"
    model_save_path = "best_lstm_model.pth"
    
    # 教師データの収集
    collect_training_data(video_paths, train_data_dir, model_path)
    
    # データを訓練用と検証用に分割
    os.makedirs(val_data_dir, exist_ok=True)
    track_files = os.listdir(train_data_dir)
    val_count = int(len(track_files) * 0.2)
    
    for i, file in enumerate(track_files):
        if i < val_count:
            src = os.path.join(train_data_dir, file)
            dst = os.path.join(val_data_dir, file)
            os.rename(src, dst)
    
    # LSTMモデルのトレーニング
    train_lstm_model(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        model_save_path=model_save_path,
        batch_size=32,
        epochs=100,
        learning_rate=0.001
    ) 