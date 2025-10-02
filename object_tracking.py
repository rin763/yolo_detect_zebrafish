import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
from torch.utils.data import Dataset, DataLoader
import os

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

class LSTMTracker(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(LSTMTracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # x, y, w, h の予測

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def train_model(self, train_data_dir, val_data_dir, batch_size=32, epochs=100, learning_rate=0.001):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 最良の検証損失を記録
        best_val_loss = float('inf')
        
        print("\nStarting training...")
        # トレーニングループ
        for epoch in range(epochs):
            # 訓練フェーズ
            self.train()
            train_loss = 0.0
            train_batches = 0
            
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
                
                # バッチごとの進捗表示
                if train_batches % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {train_batches}, Loss: {loss.item():.4f}")
            
            # 検証フェーズ
            self.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    outputs = self(sequences)
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
                torch.save(self.state_dict(), 'best_lstm_model.pth')
                print(f"Model saved! New best validation loss: {best_val_loss:.4f}")
            
            print("-" * 50)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

class ObjectTracker:
    def __init__(self, model_path, lstm_model_path=None, sequence_length=10):
        # YOLOモデルの読み込み
        self.yolo = YOLO(model_path)
        
        # LSTMモデルの初期化
        self.lstm = LSTMTracker()
        if lstm_model_path and os.path.exists(lstm_model_path):
            self.lstm.load_state_dict(torch.load(lstm_model_path))
            print(f"Loaded LSTM model from {lstm_model_path}")
        self.sequence_length = sequence_length
        
        # 物体の追跡履歴を保存
        self.track_history = {}
        
        # ID管理システム
        self.available_ids = set(range(1, 6))  # 1から5までのID
        self.active_tracks = {}  # 現在追跡中の物体とそのID
        self.missed_frames = {}  # 物体を見失ったフレーム数を記録
        
    def get_new_id(self):
        """利用可能なIDを取得"""
        if self.available_ids:
            return self.available_ids.pop()
        return None
    
    def release_id(self, obj_id):
        """IDを解放"""
        if 1 <= obj_id <= 5:
            self.available_ids.add(obj_id)
            if obj_id in self.active_tracks:
                del self.active_tracks[obj_id]
            if obj_id in self.missed_frames:
                del self.missed_frames[obj_id]
    
    def preprocess_detection(self, detection):
        # YOLOの検出結果をLSTMの入力形式に変換
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections):
        # 現在のフレームで検出された物体のIDを記録
        current_detections = set()
        
        for det in detections:
            # 物体の位置情報を取得
            bbox = det.xywh[0].cpu().numpy()  # x, y, w, h
            
            # 既存の追跡とマッチング
            matched = False
            for track_id, track_info in list(self.active_tracks.items()):
                if track_id in current_detections:
                    continue
                    
                # 既存の追跡と新しい検出の距離を計算
                old_bbox = self.track_history[track_id][-1]
                distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                
                # 距離が閾値以下の場合、同じ物体とみなす
                if distance < 200:  # 閾値を200に変更
                    current_detections.add(track_id)
                    matched = True
                    self.missed_frames[track_id] = 0
                    self.track_history[track_id].append(self.preprocess_detection(bbox))
                    break
            
            # 新しい物体の場合
            if not matched:
                new_id = self.get_new_id()
                if new_id is not None:
                    current_detections.add(new_id)
                    self.active_tracks[new_id] = bbox
                    self.track_history[new_id] = deque(maxlen=self.sequence_length)
                    self.track_history[new_id].append(self.preprocess_detection(bbox))
                    self.missed_frames[new_id] = 0
        
        # 見失った物体の処理
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
                if self.missed_frames[track_id] > 30:  # 30フレーム以上見失った場合
                    self.release_id(track_id)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # 動画の設定を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 出力用のVideoWriterを設定
        output_path = "video/tracking_video_right.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # YOLOで物体検出
            results = self.yolo.track(frame, persist=True)
            
            if results[0].boxes is not None:
                # 追跡の更新
                self.update_tracking(cap.get(cv2.CAP_PROP_POS_FRAMES), 
                                   results[0].boxes)
                
                # 結果の可視化
                used_ids = set()  # このフレームで使用済みのIDを記録
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # 物体の中心座標を計算
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 最も近い追跡IDを探す（使用済みのIDは除外）
                    min_distance = float('inf')
                    closest_id = None
                    
                    for track_id, track_info in self.active_tracks.items():
                        if track_id in used_ids:  # 使用済みのIDはスキップ
                            continue
                        distance = np.linalg.norm(np.array([center_x, center_y]) - track_info[:2])
                        if distance < min_distance:
                            min_distance = distance
                            closest_id = track_id
                    
                    if closest_id is not None:
                        used_ids.add(closest_id)  # 使用したIDを記録
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {closest_id}", 
                                  (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # フレームを出力ファイルに書き込み
            out.write(frame)
            
            # 結果の表示
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Tracking result saved to {output_path}")

    def collect_training_data(self, video_path, output_dir):
        """
        動画から教師データを自動生成
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        track_data = {}  # 物体IDごとの追跡データを保存
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # YOLOで物体検出
            results = self.yolo.track(frame, persist=True)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # 物体の位置情報を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 物体のIDを取得
                    obj_id = int(box.id.item()) if box.id is not None else None
                    
                    if obj_id is not None:
                        # 追跡データを保存
                        if obj_id not in track_data:
                            track_data[obj_id] = []
                        
                        track_data[obj_id].append([frame_count, center_x, center_y, width, height])
            
            frame_count += 1
            
            # 進捗表示
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # 追跡データを保存
        for obj_id, data in track_data.items():
            if len(data) >= self.sequence_length:  # 十分なデータがある場合のみ保存
                data_array = np.array(data)
                output_path = os.path.join(output_dir, f"track_{obj_id}.npy")
                np.save(output_path, data_array)
                print(f"Saved tracking data for object {obj_id} with {len(data)} frames")
        
        cap.release()
        print(f"Data collection completed. Saved {len(track_data)} object tracks.")

if __name__ == "__main__":
    # モデルのパスを指定
    model_path = "./train_results/weights/best.pt"
    lstm_model_path = "best_lstm_model.pth"  # 学習済みLSTMモデルのパス
    
    # トラッカーの初期化
    tracker = ObjectTracker(model_path, lstm_model_path)
    
    # 動画のパスを指定
    video_path = "./video/processed_train_video_right.mp4"
    
    # 動画の処理開始
    tracker.process_video(video_path) 