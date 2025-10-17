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
    def __init__(self, model_path, lstm_model_path=None, sequence_length=10, max_fish=20):
        # YOLOモデルの読み込み（model_pathがNoneの場合はスキップ）
        if model_path is not None:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = None
        
        # LSTMモデルの初期化（表示用のみ）
        self.lstm = LSTMTracker()
        self.lstm_available = False
        if lstm_model_path and os.path.exists(lstm_model_path):
            try:
                self.lstm.load_state_dict(torch.load(lstm_model_path))
                self.lstm.eval()  # 推論モードに設定
                self.lstm_available = True
                print(f"Loaded LSTM model from {lstm_model_path} (display only)")
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
                self.lstm_available = False
        else:
            print(f"LSTM model not found at {lstm_model_path}")
        self.sequence_length = sequence_length
        self.max_fish = max_fish
        
        # 物体の追跡履歴を保存
        self.track_history = {}
        
        # 改善されたID管理システム
        self.next_id = 1  # 次の新しいID
        self.active_tracks = {}  # 現在追跡中の物体とそのID
        self.missed_frames = {}  # 物体を見失ったフレーム数を記録
        
        # 見失った魚の情報を記憶するシステム
        self.lost_fish = {}  # {id: {'last_position': [x, y, w, h], 'lost_frames': count, 'last_seen_frame': frame_id}}
        self.reuse_distance_threshold = 150  # ID再利用の距離閾値
        self.max_lost_frames = 60  # IDを保持する最大フレーム数
        
        # 破棄されたIDを記録するシステム
        self.discarded_ids = set()  # 破棄されたIDのセット
        self.max_discarded_ids = 20  # 記録する最大破棄ID数
        
        # LSTM予測ベース検出の設定
        self.use_lstm_detection = True  # LSTM予測を検出として使用するか
        self.lstm_detection_threshold = 0.7  # LSTM予測の信頼度閾値
        self.max_lstm_detection_frames = 15  # LSTM予測検出の最大フレーム数
        
    def predict_next_position(self, track_id):
        """指定されたトラックIDの次の位置をLSTMで予測（表示用のみ）"""
        if not self.lstm_available or track_id not in self.track_history:
            return None
            
        if len(self.track_history[track_id]) < self.sequence_length:
            return None
        
        # 履歴からシーケンスを取得
        history = list(self.track_history[track_id])
        sequence = np.array(history[-self.sequence_length:])
        
        # LSTMに入力する形式に変換
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, sequence_length, 4)
        
        with torch.no_grad():
            prediction = self.lstm(sequence_tensor).cpu().numpy()[0]  # [x, y, w, h]
        
        return prediction
    
    def compute_prediction_confidence(self, track_id):
        """LSTM予測の信頼度を計算"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return 0.0
        
        # 最近3フレームの移動パターンを分析
        recent_positions = list(self.track_history[track_id])[-3:]
        
        # 移動の一貫性を計算
        movements = []
        for i in range(1, len(recent_positions)):
            movement = np.linalg.norm(np.array(recent_positions[i][:2]) - np.array(recent_positions[i-1][:2]))
            movements.append(movement)
        
        if len(movements) < 2:
            return 0.0
        
        # 移動の一貫性（標準偏差が小さいほど信頼度が高い）
        movement_std = np.std(movements)
        avg_movement = np.mean(movements)
        
        # 信頼度計算（移動が一貫しているほど高い）
        if avg_movement == 0:
            return 1.0
        
        consistency = max(0, 1 - (movement_std / avg_movement))
        return min(1.0, consistency)
    
    def create_virtual_detection(self, track_id, prediction):
        """LSTM予測から仮想検出オブジェクトを作成"""
        # 仮想検出オブジェクトのクラス
        class VirtualDetection:
            def __init__(self, bbox):
                self.xywh = [torch.tensor(bbox)]  # YOLO検出と同じ形式
                self.conf = 0.8  # 仮想検出の信頼度
                self.cls = torch.tensor([0])  # クラス（魚）
        
        return VirtualDetection(prediction)
    
    def get_lstm_detections(self):
        """LSTM予測から仮想検出を生成"""
        virtual_detections = []
        
        if not self.lstm_available or not self.use_lstm_detection:
            return virtual_detections
        
        for track_id in self.active_tracks.keys():
            # 見失い中のトラックのみ対象
            if self.missed_frames.get(track_id, 0) > 0:
                # LSTM予測が可能かチェック
                if track_id in self.track_history and len(self.track_history[track_id]) >= self.sequence_length:
                    prediction = self.predict_next_position(track_id)
                    if prediction is not None:
                        # 予測の信頼度を計算
                        confidence = self.compute_prediction_confidence(track_id)
                        
                        # 信頼度が閾値以上で、見失いフレーム数が上限以下
                        if confidence >= self.lstm_detection_threshold and self.missed_frames[track_id] <= self.max_lstm_detection_frames:
                            virtual_detection = self.create_virtual_detection(track_id, prediction)
                            virtual_detection.track_id = track_id  # トラックIDを追加
                            virtual_detections.append(virtual_detection)
                            print(f"Created virtual detection for track {track_id} (confidence: {confidence:.3f}, missed: {self.missed_frames[track_id]})")
        
        return virtual_detections
        
    def get_new_id(self):
        """新しいIDを取得（破棄されたIDを優先的に再利用）"""
        # 破棄されたIDがあれば再利用
        if self.discarded_ids:
            reused_id = min(self.discarded_ids)  # 最小のIDを再利用
            self.discarded_ids.remove(reused_id)
            print(f"Reusing discarded ID: {reused_id}")
            return reused_id
        
        # 破棄されたIDがない場合は新しいIDを作成
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def release_id(self, obj_id, current_frame):
        """IDを解放し、見失った魚として記録（30フレーム以上見失った場合）"""
        if obj_id in self.active_tracks:
            # 見失った魚の情報を保存
            self.lost_fish[obj_id] = {
                'last_position': self.active_tracks[obj_id].copy(),
                'lost_frames': 0,
                'last_seen_frame': current_frame
            }
            del self.active_tracks[obj_id]
        if obj_id in self.missed_frames:
            del self.missed_frames[obj_id]
        if obj_id in self.track_history:
            del self.track_history[obj_id]
    
    def find_reusable_id(self, new_position, current_frame):
        """見失った魚の中で再利用可能なIDを探す（30フレーム以上）"""
        reusable_id = None
        min_distance = float('inf')
        
        for fish_id, fish_info in list(self.lost_fish.items()):
            # フレーム数が上限を超えている場合は破棄されたIDとして記録
            if fish_info['lost_frames'] > self.max_lost_frames:
                del self.lost_fish[fish_id]
                self.add_discarded_id(fish_id)
                continue
                
            # 距離を計算
            last_pos = fish_info['last_position']
            distance = np.linalg.norm(np.array(new_position[:2]) - np.array(last_pos[:2]))
            
            # 距離が閾値以下で、最も近い場合
            if distance <= self.reuse_distance_threshold and distance < min_distance:
                min_distance = distance
                reusable_id = fish_id
        
        return reusable_id
    
    def add_discarded_id(self, fish_id):
        """破棄されたIDを記録（20個まで）"""
        if len(self.discarded_ids) < self.max_discarded_ids:
            self.discarded_ids.add(fish_id)
            print(f"Added discarded ID {fish_id} to reuse pool (total: {len(self.discarded_ids)})")
        else:
            # 最大数に達している場合は、最も古いIDを削除して新しいIDを追加
            oldest_id = max(self.discarded_ids)
            self.discarded_ids.remove(oldest_id)
            self.discarded_ids.add(fish_id)
            print(f"Replaced discarded ID {oldest_id} with {fish_id}")
    
    def preprocess_detection(self, detection):
        # YOLOの検出結果をLSTMの入力形式に変換
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections):
        # 見失った魚のフレーム数を更新（30フレーム以上見失った場合）
        for fish_id in self.lost_fish:
            self.lost_fish[fish_id]['lost_frames'] += 1
        
        # 現在のフレームで検出された物体のIDを記録
        current_detections = set()
        
        for det in detections:
            # 物体の位置情報を取得
            bbox = det.xywh[0].cpu().numpy()  # x, y, w, h
            
            # 既存の追跡とマッチング（200ピクセル以内）
            matched = False
            min_distance = float('inf')
            best_match_id = None
            
            for track_id, track_info in list(self.active_tracks.items()):
                if track_id in current_detections:
                    continue
                    
                # 既存の追跡と新しい検出の距離を計算（200ピクセル以内）
                if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                    old_bbox = self.track_history[track_id][-1]
                    distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                    
                    # 距離が閾値以下の場合、候補として記録（200ピクセル以内）
                    if distance < 200:
                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = track_id
            
            # 最適なマッチを見つけた場合（200ピクセル以内）
            if best_match_id is not None:
                current_detections.add(best_match_id)
                matched = True
                self.missed_frames[best_match_id] = 0
                self.track_history[best_match_id].append(self.preprocess_detection(bbox))
                self.active_tracks[best_match_id] = bbox
            
            # 既存の追跡にマッチしなかった場合、見失った魚のIDを再利用（200ピクセル以内）
            if not matched:
                reusable_id = self.find_reusable_id(bbox, frame_id)
                if reusable_id is not None:
                    # 見失った魚のIDを再利用（200ピクセル以内）
                    current_detections.add(reusable_id)
                    self.active_tracks[reusable_id] = bbox
                    self.track_history[reusable_id] = deque(maxlen=self.sequence_length)
                    self.track_history[reusable_id].append(self.preprocess_detection(bbox))
                    self.missed_frames[reusable_id] = 0
                    # 見失った魚リストから削除
                    del self.lost_fish[reusable_id]
                    print(f"Reused ID {reusable_id} for fish near position {bbox[:2]}")
                else:
                    # 新しいIDを作成（破棄されたIDを優先的に再利用）
                    new_id = self.get_new_id()
                    current_detections.add(new_id)
                    self.active_tracks[new_id] = bbox
                    self.track_history[new_id] = deque(maxlen=self.sequence_length)
                    self.track_history[new_id].append(self.preprocess_detection(bbox))
                    self.missed_frames[new_id] = 0
                    print(f"Created new ID {new_id} for fish at position {bbox[:2]}")
        
        # 見失った物体の処理（30フレーム以上見失った場合） 
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
                if self.missed_frames[track_id] > 30:  # 30フレーム以上見失った場合
                    missed_count = self.missed_frames[track_id]
                    self.release_id(track_id, frame_id)
                    print(f"Lost fish ID {track_id} after {missed_count} frames")

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
            
            # YOLO検出とLSTM仮想検出を結合
            all_detections = []
            if results[0].boxes is not None:
                all_detections.extend(results[0].boxes)
            
            # LSTM予測から仮想検出を生成
            virtual_detections = self.get_lstm_detections()
            all_detections.extend(virtual_detections)
            
            if all_detections:
                # 追跡の更新（YOLO検出 + LSTM仮想検出）
                self.update_tracking(cap.get(cv2.CAP_PROP_POS_FRAMES), all_detections)
                
                # 結果の可視化
                used_ids = set()  # このフレームで使用済みのIDを記録
                for box in all_detections:
                    # YOLO検出とLSTM仮想検出を区別
                    is_virtual = hasattr(box, 'track_id')
                    
                    if is_virtual:
                        # LSTM仮想検出の可視化
                        bbox = box.xywh[0].cpu().numpy()
                        x, y, w, h = bbox
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色で仮想検出
                        cv2.putText(frame, f"LSTM-{box.track_id}", 
                                  (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    else:
                        # YOLO検出の可視化
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
            
            # 全アクティブトラックのLSTM予測位置を赤色で表示（表示のみ）
            # if self.lstm_available:
            #     for track_id in self.active_tracks.keys():
            #         if track_id in self.track_history and len(self.track_history[track_id]) >= self.sequence_length:
            #             prediction = self.predict_next_position(track_id)
            #             if prediction is not None:
            #                 pred_x, pred_y = prediction[:2]
            #                 # 赤色で予測位置を表示
            #                 cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (0, 0, 255), 2)  # 赤色の円
            #                 cv2.putText(frame, f"LSTM-{track_id}", 
            #                           (int(pred_x)+15, int(pred_y)), 
            #                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 赤色のテキスト
            
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
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3D_left.mp4"
    
    # 動画の処理開始
    tracker.process_video(video_path) 