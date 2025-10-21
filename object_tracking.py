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
        
        # IDごとの位置履歴を保存（x, y座標、動きの方向）
        self.position_history = {}  # {track_id: [(frame_id, x, y, source, direction_rad), ...]}
        
        # 改善されたID管理システム
        self.next_id = 1  # 次の新しいID
        self.active_tracks = {}  # 現在追跡中の物体とそのID
        self.missed_frames = {}  # 物体を見失ったフレーム数を記録
        
        # 見失った魚の情報を記憶するシステム
        self.lost_fish = {}  # {id: {'last_position': [x, y, w, h], 'lost_frames': count, 'last_seen_frame': frame_id}}
        self.reuse_distance_threshold = 150  # ID再利用の距離閾値
        self.max_lost_frames = 60  # IDを保持する最大フレーム数
        
        # 破棄されたIDを記録するシステム（位置情報も含む）
        self.discarded_ids = {}  # {id: {'position': [x, y], 'discarded_frame': frame_id}}
        self.max_discarded_ids = 20  # 記録する最大破棄ID数
        self.discarded_id_reuse_distance = 100  # 破棄ID再利用の距離閾値
        
        # LSTM予測ベース検出の設定
        self.use_lstm_detection = True  # LSTM予測を検出として使用するか
        self.lstm_detection_threshold = 0.75  # LSTM予測の信頼度閾値
        self.max_lstm_detection_frames = 15  # LSTM予測検出の最大フレーム数
        
        # 動きの方向ベースマッチングの設定
        self.use_direction_matching = True  # 動きの方向を考慮したマッチングを使用するか
        self.direction_weight = 0.2  # 方向の重み（0.0-1.0）
        self.distance_weight = 0.8  # 距離の重み（0.0-1.0）
        self.direction_threshold = 1.0  # 方向差の閾値（ラジアン）
        
        # MOT評価指標のためのデータ収集
        self.mot_evaluation = {
            'total_frames': 0,
            'total_gt_objects': 0,
            'total_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'id_switches': 0,
            'fragmentations': 0,
            'tracked_objects': {},  # {gt_id: {'frames': [], 'total_frames': 0, 'tracked_frames': 0}}
            'detection_matches': [],  # [(frame_id, detection_id, gt_id, iou)]
            'track_continuity': {}  # {track_id: {'last_gt_id': None, 'fragments': 0}}
        }
        
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
    
    def update_position_history(self, track_id, frame_id, x, y, source="detection"):
        """IDごとの位置履歴を更新（動きの方向も計算）"""
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        # 動きの方向を計算
        direction_rad = 0.0  # デフォルト値
        if len(self.position_history[track_id]) > 0:
            # 前の位置を取得
            prev_frame, prev_x, prev_y, prev_source, prev_direction = self.position_history[track_id][-1]
            
            # xとyの差を計算
            dx = x - prev_x
            dy = y - prev_y
            
            # 動きがある場合のみ方向を計算
            if dx != 0 or dy != 0:
                direction_rad = np.arctan2(dy, dx)
                print(f"Track {track_id}: dx={dx:.2f}, dy={dy:.2f}, direction={direction_rad:.3f} rad")
        
        # 位置履歴に追加（方向も含む）
        self.position_history[track_id].append((frame_id, x, y, source, direction_rad))
        
        # 履歴が長すぎる場合は古いものを削除（メモリ節約）
        if len(self.position_history[track_id]) > 1000:  # 最大1000フレーム分保持
            self.position_history[track_id] = self.position_history[track_id][-500:]  # 最新500フレーム分のみ保持
    
    def get_lstm_prediction_for_missed_track(self, track_id, frame_id):
        """見失ったトラックのLSTM予測を取得（信頼度0.7以上のみ）"""
        if not self.lstm_available or track_id not in self.track_history:
            return None
            
        if len(self.track_history[track_id]) < self.sequence_length:
            return None
        
        # LSTM予測を取得
        prediction = self.predict_next_position(track_id)
        if prediction is None:
            return None
        
        # 信頼度を計算
        confidence = self.compute_prediction_confidence(track_id)
        
        # 信頼度が0.7以上の場合のみ採用
        if confidence >= 0.7:
            pred_x, pred_y = prediction[:2]
            # 位置履歴に追加（動きの方向も計算される）
            self.update_position_history(track_id, frame_id, pred_x, pred_y, "lstm_prediction")
            print(f"Added LSTM prediction to position history for track {track_id} (confidence: {confidence:.3f})")
            return (pred_x, pred_y)
        
        return None
        
    def get_new_id(self):
        """新しいIDを取得（破棄されたIDを優先的に再利用）"""
        # 破棄されたIDがあれば再利用
        if self.discarded_ids:
            reused_id = min(self.discarded_ids.keys())  # 最小のIDを再利用
            del self.discarded_ids[reused_id]
            print(f"Reusing discarded ID: {reused_id}")
            return reused_id
        
        # 破棄されたIDがない場合は新しいIDを作成
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def add_discarded_id(self, fish_id, position, frame_id):
        """破棄されたIDを記録（位置情報も含む）"""
        self.discarded_ids[fish_id] = {
            'position': position[:2].copy(),  # x, y座標のみ
            'discarded_frame': frame_id
        }
        
        # 最大数を超えた場合は古いものを削除
        if len(self.discarded_ids) > self.max_discarded_ids:
            # 最も古いIDを削除
            oldest_id = min(self.discarded_ids.keys(), 
                          key=lambda x: self.discarded_ids[x]['discarded_frame'])
            del self.discarded_ids[oldest_id]
            print(f"Removed oldest discarded ID {oldest_id} from discarded_ids")
        
        print(f"Recorded discarded ID {fish_id} at position {position[:2]}")
    
    def find_nearest_discarded_id(self, new_position):
        """新しい位置に最も近い破棄IDを検索"""
        if not self.discarded_ids:
            return None
        
        min_distance = float('inf')
        nearest_id = None
        
        for discarded_id, info in self.discarded_ids.items():
            discarded_position = info['position']
            distance = np.linalg.norm(np.array(new_position[:2]) - np.array(discarded_position))
            
            if distance < self.discarded_id_reuse_distance and distance < min_distance:
                min_distance = distance
                nearest_id = discarded_id
        
        if nearest_id is not None:
            print(f"Found nearest discarded ID {nearest_id} at distance {min_distance:.2f}")
            return nearest_id, min_distance
        else:
            return None
    
    def get_last_direction(self, track_id):
        """指定されたトラックの最後の動きの方向を取得"""
        if track_id not in self.position_history or len(self.position_history[track_id]) < 2:
            return None
        
        # 最後の2つの位置から方向を計算
        last_positions = self.position_history[track_id][-2:]
        prev_frame, prev_x, prev_y, prev_source, prev_direction = last_positions[0]
        curr_frame, curr_x, curr_y, curr_source, curr_direction = last_positions[1]
        
        # 実際の移動から方向を再計算（より正確）
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        if dx == 0 and dy == 0:
            return 0.0
        
        return np.arctan2(dy, dx)
    
    def compute_direction_difference(self, dir1, dir2):
        """2つの方向の差を計算（-πからπの範囲で正規化）"""
        if dir1 is None or dir2 is None:
            return float('inf')
        
        diff = dir1 - dir2
        
        # -πからπの範囲に正規化
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        return abs(diff)
    
    def compute_enhanced_matching_cost(self, detection, track_id):
        """距離と方向を考慮したマッチングコストを計算"""
        if track_id not in self.track_history or len(self.track_history[track_id]) == 0:
            return float('inf')
        
        # 距離コスト
        old_bbox = self.track_history[track_id][-1]
        distance = np.linalg.norm(detection[:2] - old_bbox[:2])
        
        # 距離が閾値を超える場合は除外
        if distance > 200:
            return float('inf')
        
        # 方向コスト
        direction_cost = 0.0
        if self.use_direction_matching:
            # 検出の予想方向を計算（最後の位置から現在の位置へ）
            last_positions = self.position_history.get(track_id, [])
            if len(last_positions) >= 1:
                last_frame, last_x, last_y, last_source, last_direction = last_positions[-1]
                dx = detection[0] - last_x
                dy = detection[1] - last_y
                
                if dx != 0 or dy != 0:
                    detection_direction = np.arctan2(dy, dx)
                    track_direction = self.get_last_direction(track_id)
                    
                    if track_direction is not None:
                        direction_diff = self.compute_direction_difference(detection_direction, track_direction)
                        
                        # 方向差が閾値を超える場合はペナルティ
                        if direction_diff > self.direction_threshold:
                            direction_cost = direction_diff * 10  # 大きなペナルティ
                        else:
                            direction_cost = direction_diff
        
        # 総合コスト（距離と方向の重み付き和）
        total_cost = self.distance_weight * distance + self.direction_weight * direction_cost
        
        return total_cost, distance, direction_cost
    
    def compute_iou(self, box1, box2):
        """2つのバウンディングボックスのIoUを計算"""
        # box1, box2: [x, y, w, h] または [x1, y1, x2, y2]
        if len(box1) == 4 and len(box2) == 4:
            # [x, y, w, h] 形式の場合
            x1_1, y1_1, w1, h1 = box1
            x2_1, y2_1, w2, h2 = box2
            x1_2, y1_2 = x1_1 + w1, y1_1 + h1
            x2_2, y2_2 = x2_1 + w2, y2_1 + h2
        else:
            # [x1, y1, x2, y2] 形式の場合
            x1_1, y1_1, x1_2, y1_2 = box1
            x2_1, y2_1, x2_2, y2_2 = box2
        
        # 交差部分の計算
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 面積の計算
        area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
        area2 = (x2_2 - x2_1) * (y2_2 - y2_1)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_mot_metrics(self, ground_truth_data=None):
        """MOT評価指標を計算"""
        if not ground_truth_data:
            print("Ground truth data is required for MOT evaluation")
            return None
        
        # MOTA (Multi-Object Tracking Accuracy)
        mota = 1 - (self.mot_evaluation['false_positives'] + 
                   self.mot_evaluation['false_negatives'] + 
                   self.mot_evaluation['id_switches']) / max(self.mot_evaluation['total_gt_objects'], 1)
        
        # MOTP (Multi-Object Tracking Precision)
        if self.mot_evaluation['detection_matches']:
            total_iou = sum(match[3] for match in self.mot_evaluation['detection_matches'])
            motp = total_iou / len(self.mot_evaluation['detection_matches'])
        else:
            motp = 0.0
        
        # IDS (ID Switches)
        ids = self.mot_evaluation['id_switches']
        
        # Frag (Fragmentations)
        frag = self.mot_evaluation['fragmentations']
        
        # MT (Mostly Tracked) と ML (Mostly Lost)
        mt_count = 0
        ml_count = 0
        total_tracks = len(self.mot_evaluation['tracked_objects'])
        
        for gt_id, track_info in self.mot_evaluation['tracked_objects'].items():
            tracked_ratio = track_info['tracked_frames'] / max(track_info['total_frames'], 1)
            if tracked_ratio >= 0.8:
                mt_count += 1
            elif tracked_ratio < 0.2:
                ml_count += 1
        
        mt = mt_count / max(total_tracks, 1)
        ml = ml_count / max(total_tracks, 1)
        
        return {
            'MOTA': mota,
            'MOTP': motp,
            'IDS': ids,
            'Frag': frag,
            'MT': mt,
            'ML': ml,
            'Total_Frames': self.mot_evaluation['total_frames'],
            'Total_GT_Objects': self.mot_evaluation['total_gt_objects'],
            'Total_Detections': self.mot_evaluation['total_detections'],
            'False_Positives': self.mot_evaluation['false_positives'],
            'False_Negatives': self.mot_evaluation['false_negatives']
        }
    
    def save_mot_evaluation_to_file(self, output_file="mot_evaluation_results.txt"):
        """MOT評価結果をファイルに保存"""
        from datetime import datetime
        
        metrics = self.evaluate_mot_metrics()
        if not metrics:
            print("No evaluation data available")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Multi-Object Tracking Evaluation Results ===\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 主要指標
            f.write("=== Main Metrics ===\n")
            f.write(f"MOTA (Multi-Object Tracking Accuracy): {metrics['MOTA']:.4f}\n")
            f.write(f"MOTP (Multi-Object Tracking Precision): {metrics['MOTP']:.4f}\n")
            f.write(f"IDS (ID Switches): {metrics['IDS']}\n")
            f.write(f"Frag (Fragmentations): {metrics['Frag']}\n")
            f.write(f"MT (Mostly Tracked): {metrics['MT']:.4f}\n")
            f.write(f"ML (Mostly Lost): {metrics['ML']:.4f}\n\n")
            
            # 詳細統計
            f.write("=== Detailed Statistics ===\n")
            f.write(f"Total Frames: {metrics['Total_Frames']}\n")
            f.write(f"Total GT Objects: {metrics['Total_GT_Objects']}\n")
            f.write(f"Total Detections: {metrics['Total_Detections']}\n")
            f.write(f"False Positives: {metrics['False_Positives']}\n")
            f.write(f"False Negatives: {metrics['False_Negatives']}\n\n")
            
            # トラック情報
            f.write("=== Track Information ===\n")
            f.write(f"Total Active Tracks: {len(self.active_tracks)}\n")
            f.write(f"Total Track History: {len(self.track_history)}\n")
            f.write(f"Total Position History: {len(self.position_history)}\n")
            f.write(f"Discarded IDs: {len(self.discarded_ids)}\n")
            f.write(f"Lost Fish: {len(self.lost_fish)}\n\n")
            
            # 設定パラメータ
            f.write("=== Tracking Parameters ===\n")
            f.write(f"Use LSTM Detection: {self.use_lstm_detection}\n")
            f.write(f"LSTM Detection Threshold: {self.lstm_detection_threshold}\n")
            f.write(f"Max LSTM Detection Frames: {self.max_lstm_detection_frames}\n")
            f.write(f"Use Direction Matching: {self.use_direction_matching}\n")
            f.write(f"Direction Weight: {self.direction_weight}\n")
            f.write(f"Distance Weight: {self.distance_weight}\n")
            f.write(f"Direction Threshold: {self.direction_threshold}\n")
            f.write(f"Discarded ID Reuse Distance: {self.discarded_id_reuse_distance}\n")
            f.write(f"Max Discarded IDs: {self.max_discarded_ids}\n")
            f.write(f"Max Lost Frames: {self.max_lost_frames}\n")
            f.write(f"Reuse Distance Threshold: {self.reuse_distance_threshold}\n")
        
        print(f"MOT evaluation results saved to {output_file}")
    
    def save_detailed_tracking_data(self, output_file="detailed_tracking_data.json"):
        """詳細なトラッキングデータをJSONファイルに保存"""
        import json
        import numpy as np
        from datetime import datetime
        
        def convert_numpy(obj):
            """numpy配列をPythonのリストに変換"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        tracking_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'mot_evaluation': convert_numpy(self.mot_evaluation),
            'track_history': {str(k): convert_numpy(list(v)) for k, v in self.track_history.items()},
            'position_history': {str(k): convert_numpy(v) for k, v in self.position_history.items()},
            'active_tracks': {str(k): convert_numpy(v) for k, v in self.active_tracks.items()},
            'discarded_ids': {str(k): convert_numpy(v) for k, v in self.discarded_ids.items()},
            'lost_fish': {str(k): convert_numpy(v) for k, v in self.lost_fish.items()},
            'track_continuity': {str(k): convert_numpy(v) for k, v in self.mot_evaluation['track_continuity'].items()},
            'detection_matches': convert_numpy(self.mot_evaluation['detection_matches'])
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed tracking data saved to {output_file}")
    
    def visualize_mot_evaluation(self, save_plots=True):
        """MOT評価結果を可視化"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
        except ImportError:
            print("matplotlib is required for visualization. Please install it with: pip install matplotlib")
            return
        
        metrics = self.evaluate_mot_metrics()
        if not metrics:
            print("No evaluation data available")
            return
        
        # 図の設定
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Object Tracking Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. 主要指標のバーチャート
        main_metrics = ['MOTA', 'MOTP', 'MT', 'ML']
        main_values = [metrics['MOTA'], metrics['MOTP'], metrics['MT'], metrics['ML']]
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#DC143C']
        
        bars1 = ax1.bar(main_metrics, main_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Main MOT Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, value in zip(bars1, main_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. エラー統計のパイチャート
        error_labels = ['False Positives', 'False Negatives', 'ID Switches', 'Fragmentations']
        error_values = [metrics['False_Positives'], metrics['False_Negatives'], 
                       metrics['IDS'], metrics['Frag']]
        
        # ゼロでない値のみ表示
        non_zero_labels = []
        non_zero_values = []
        for label, value in zip(error_labels, error_values):
            if value > 0:
                non_zero_labels.append(f'{label}\n({value})')
                non_zero_values.append(value)
        
        if non_zero_values:
            colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            wedges, texts, autotexts = ax2.pie(non_zero_values, labels=non_zero_labels, 
                                              colors=colors_pie[:len(non_zero_values)], 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Error Distribution', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Errors Detected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, fontweight='bold')
            ax2.set_title('Error Distribution', fontweight='bold')
        
        # 3. トラッキング統計の棒グラフ
        track_stats = ['Total\nFrames', 'Total\nDetections', 'Active\nTracks', 'Discarded\nIDs']
        track_values = [metrics['Total_Frames'], metrics['Total_Detections'], 
                       len(self.active_tracks), len(self.discarded_ids)]
        
        bars3 = ax3.bar(track_stats, track_values, color='#87CEEB', alpha=0.7, 
                       edgecolor='black', linewidth=1)
        ax3.set_title('Tracking Statistics', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, value in zip(bars3, track_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(track_values)*0.01, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. 位置履歴の時系列プロット（最初の3つのトラック）
        ax4.set_title('Position History (First 3 Tracks)', fontweight='bold')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Position')
        
        colors_track = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        track_count = 0
        
        for track_id, positions in self.position_history.items():
            if track_count >= 3:  # 最初の3つのトラックのみ
                break
            
            if len(positions) > 1:
                frames = [pos[0] for pos in positions]
                x_coords = [pos[1] for pos in positions]
                y_coords = [pos[2] for pos in positions]
                
                ax4.plot(frames, x_coords, 'o-', color=colors_track[track_count], 
                        label=f'Track {track_id} (X)', alpha=0.7, markersize=3)
                ax4.plot(frames, y_coords, 's-', color=colors_track[track_count], 
                        label=f'Track {track_id} (Y)', alpha=0.5, markersize=2, linestyle='--')
            
            track_count += 1
        
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # レイアウト調整
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'mot_evaluation_visualization_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {filename}")
        
        plt.show()
    
    def create_tracking_timeline(self, save_plot=True):
        """トラッキングのタイムラインを可視化"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
        except ImportError:
            print("matplotlib is required for visualization. Please install it with: pip install matplotlib")
            return
        
        if not self.position_history:
            print("No position history data available")
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 各トラックのタイムラインを作成
        track_colors = plt.cm.tab20(np.linspace(0, 1, len(self.position_history)))
        
        for i, (track_id, positions) in enumerate(self.position_history.items()):
            if len(positions) < 2:
                continue
            
            frames = [pos[0] for pos in positions]
            x_coords = [pos[1] for pos in positions]
            y_coords = [pos[2] for pos in positions]
            sources = [pos[3] for pos in positions]
            
            # トラックの軌跡をプロット
            ax.plot(x_coords, y_coords, 'o-', color=track_colors[i], 
                   label=f'Track {track_id}', alpha=0.7, markersize=4, linewidth=2)
            
            # 開始点と終了点を強調
            ax.plot(x_coords[0], y_coords[0], 's', color=track_colors[i], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            ax.plot(x_coords[-1], y_coords[-1], '^', color=track_colors[i], 
                   markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            # LSTM予測点を特別にマーク
            lstm_points_x = [x for j, (x, y, source) in enumerate(zip(x_coords, y_coords, sources)) 
                           if source == 'lstm_prediction']
            lstm_points_y = [y for j, (x, y, source) in enumerate(zip(x_coords, y_coords, sources)) 
                           if source == 'lstm_prediction']
            
            if lstm_points_x:
                ax.scatter(lstm_points_x, lstm_points_y, color=track_colors[i], 
                          marker='*', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        ax.set_title('Fish Tracking Timeline', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 凡例の説明を追加
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='black', linestyle='None', 
                      markersize=8, label='Start Point'),
            plt.Line2D([0], [0], marker='^', color='black', linestyle='None', 
                      markersize=8, label='End Point'),
            plt.Line2D([0], [0], marker='*', color='red', linestyle='None', 
                      markersize=10, label='LSTM Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tracking_timeline_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Tracking timeline saved as {filename}")
        
        plt.show()
    
    def print_mot_evaluation(self):
        """MOT評価結果を表示"""
        metrics = self.evaluate_mot_metrics()
        if metrics:
            print("\n=== MOT Evaluation Results ===")
            print(f"MOTA (Multi-Object Tracking Accuracy): {metrics['MOTA']:.4f}")
            print(f"MOTP (Multi-Object Tracking Precision): {metrics['MOTP']:.4f}")
            print(f"IDS (ID Switches): {metrics['IDS']}")
            print(f"Frag (Fragmentations): {metrics['Frag']}")
            print(f"MT (Mostly Tracked): {metrics['MT']:.4f}")
            print(f"ML (Mostly Lost): {metrics['ML']:.4f}")
            print(f"\nDetailed Statistics:")
            print(f"Total Frames: {metrics['Total_Frames']}")
            print(f"Total GT Objects: {metrics['Total_GT_Objects']}")
            print(f"Total Detections: {metrics['Total_Detections']}")
            print(f"False Positives: {metrics['False_Positives']}")
            print(f"False Negatives: {metrics['False_Negatives']}")
            print("===============================\n")
        
    def release_id(self, obj_id, current_frame):
        """IDを解放し、見失った魚として記録（30フレーム以上見失った場合）"""
        if obj_id in self.active_tracks:
            # 見失った魚の情報を保存
            self.lost_fish[obj_id] = {
                'last_position': self.active_tracks[obj_id].copy(),
                'lost_frames': 0,
                'last_seen_frame': current_frame
            }
            
            # 破棄されたIDとして位置情報も記録
            self.add_discarded_id(obj_id, self.active_tracks[obj_id], current_frame)
            
            del self.active_tracks[obj_id]
        if obj_id in self.missed_frames:
            del self.missed_frames[obj_id]
        if obj_id in self.track_history:
            del self.track_history[obj_id]
        # 位置履歴は保持（分析用）
        # if obj_id in self.position_history:
        #     del self.position_history[obj_id]
    
    def find_reusable_id(self, new_position, current_frame):
        """見失った魚の中で再利用可能なIDを探す（30フレーム以上）"""
        reusable_id = None
        min_distance = float('inf')
        
        for fish_id, fish_info in list(self.lost_fish.items()):
            # フレーム数が上限を超えている場合は破棄されたIDとして記録
            if fish_info['lost_frames'] > self.max_lost_frames:
                del self.lost_fish[fish_id]
                self.add_discarded_id(fish_id, fish_info['last_position'], fish_info['last_seen_frame'])
                continue
                
            # 距離を計算
            last_pos = fish_info['last_position']
            distance = np.linalg.norm(np.array(new_position[:2]) - np.array(last_pos[:2]))
            
            # 距離が閾値以下で、最も近い場合
            if distance <= self.reuse_distance_threshold and distance < min_distance:
                min_distance = distance
                reusable_id = fish_id
        
        return reusable_id
    
    
    def preprocess_detection(self, detection):
        # YOLOの検出結果をLSTMの入力形式に変換
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections):
        # 見失った魚のフレーム数を更新（30フレーム以上見失った場合）
        for fish_id in self.lost_fish:
            self.lost_fish[fish_id]['lost_frames'] += 1
        
        # MOT評価のためのフレーム数更新
        self.mot_evaluation['total_frames'] += 1
        
        # 現在のフレームで検出された物体のIDを記録
        current_detections = set()
        
        for det in detections:
            # 物体の位置情報を取得
            bbox = det.xywh[0].cpu().numpy()  # x, y, w, h
            
            # MOT評価のための検出数更新
            self.mot_evaluation['total_detections'] += 1
            
            # 既存の追跡とマッチング（距離と方向を考慮）
            matched = False
            min_cost = float('inf')
            best_match_id = None
            
            for track_id, track_info in list(self.active_tracks.items()):
                if track_id in current_detections:
                    continue
                
                # 距離と方向を考慮したマッチングコストを計算
                if self.use_direction_matching:
                    cost_result = self.compute_enhanced_matching_cost(bbox, track_id)
                    if isinstance(cost_result, tuple):
                        total_cost, distance, direction_cost = cost_result
                    else:
                        total_cost = cost_result
                        distance = float('inf')
                        direction_cost = 0.0
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_match_id = track_id
                        print(f"Enhanced matching: Track {track_id}, cost={total_cost:.2f}, distance={distance:.2f}, direction_cost={direction_cost:.3f}")
                else:
                    # 従来の距離ベースマッチング
                    if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                        old_bbox = self.track_history[track_id][-1]
                        distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                        
                        if distance < 200 and distance < min_cost:
                            min_cost = distance
                            best_match_id = track_id
            
            # 最適なマッチを見つけた場合（距離と方向を考慮）
            if best_match_id is not None:
                current_detections.add(best_match_id)
                matched = True
                self.missed_frames[best_match_id] = 0
                self.track_history[best_match_id].append(self.preprocess_detection(bbox))
                self.active_tracks[best_match_id] = bbox
                # 位置履歴に追加（動きの方向も計算される）
                self.update_position_history(best_match_id, frame_id, bbox[0], bbox[1], "detection")
                
                # MOT評価のためのIDスイッチ検出
                if best_match_id in self.mot_evaluation['track_continuity']:
                    last_gt_id = self.mot_evaluation['track_continuity'][best_match_id]['last_gt_id']
                    # ここでGround Truthとの比較が必要（実装例）
                    # if last_gt_id is not None and current_gt_id != last_gt_id:
                    #     self.mot_evaluation['id_switches'] += 1
                
                print(f"Matched detection to track {best_match_id} with enhanced matching")
            
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
                    # 位置履歴に追加（動きの方向も計算される）
                    self.update_position_history(reusable_id, frame_id, bbox[0], bbox[1], "detection")
                    print(f"Reused ID {reusable_id} for fish near position {bbox[:2]}")
                else:
                    # 破棄されたIDの再利用を試行
                    result = self.find_nearest_discarded_id(bbox)
                    if result is not None:
                        nearest_discarded_id, distance = result
                        # 破棄されたIDを再利用
                        del self.discarded_ids[nearest_discarded_id]
                        print(f"Reusing discarded ID {nearest_discarded_id} for new detection at distance {distance:.2f}")
                        
                        # 破棄されたIDで再作成
                        current_detections.add(nearest_discarded_id)
                        self.active_tracks[nearest_discarded_id] = bbox
                        self.track_history[nearest_discarded_id] = deque(maxlen=self.sequence_length)
                        self.track_history[nearest_discarded_id].append(self.preprocess_detection(bbox))
                        self.missed_frames[nearest_discarded_id] = 0
                        # 位置履歴に追加（動きの方向も計算される）
                        self.update_position_history(nearest_discarded_id, frame_id, bbox[0], bbox[1], "detection")
                        
                        # MOT評価のためのフラグメンテーション検出
                        if nearest_discarded_id in self.mot_evaluation['track_continuity']:
                            self.mot_evaluation['track_continuity'][nearest_discarded_id]['fragments'] += 1
                            self.mot_evaluation['fragmentations'] += 1
                        
                        print(f"Reused discarded ID {nearest_discarded_id} for fish at position {bbox[:2]}")
                    else:
                        # 新しいIDを作成（破棄されたIDを優先的に再利用）
                        new_id = self.get_new_id()
                        current_detections.add(new_id)
                        self.active_tracks[new_id] = bbox
                        self.track_history[new_id] = deque(maxlen=self.sequence_length)
                        self.track_history[new_id].append(self.preprocess_detection(bbox))
                        self.missed_frames[new_id] = 0
                        # 位置履歴に追加（動きの方向も計算される）
                        self.update_position_history(new_id, frame_id, bbox[0], bbox[1], "detection")
                        
                        # MOT評価のための新しいトラック初期化
                        self.mot_evaluation['track_continuity'][new_id] = {
                            'last_gt_id': None,
                            'fragments': 0
                        }
                        
                        print(f"Created new ID {new_id} for fish at position {bbox[:2]}")
        
        # 見失った物体の処理（30フレーム以上見失った場合） 
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
                
                # 見失い中にLSTM予測で位置履歴を更新（信頼度0.7以上のみ）
                lstm_prediction = self.get_lstm_prediction_for_missed_track(track_id, frame_id)
                
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
        
        # 動画処理終了後にMOT評価結果を表示・保存・可視化
        self.print_mot_evaluation()
        self.save_mot_evaluation_to_file("mot_evaluation_results.txt")
        self.save_detailed_tracking_data("detailed_tracking_data.json")
        self.visualize_mot_evaluation(save_plots=True)
        self.create_tracking_timeline(save_plot=True)
                
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

    def evaluate_tracking_performance(self):
        """トラッキング性能の総合評価"""
        scores = {
            'id_consistency': self.evaluate_id_consistency(),
            'trajectory_continuity': self.evaluate_trajectory_continuity(),
            'detection_stability': self.evaluate_detection_stability(),
            'position_consistency': self.evaluate_position_consistency(),
            'lstm_accuracy': self.evaluate_lstm_prediction_accuracy()
        }
        
        # 重み付き平均
        weights = {
            'id_consistency': 0.3,
            'trajectory_continuity': 0.25,
            'detection_stability': 0.2,
            'position_consistency': 0.15,
            'lstm_accuracy': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'overall_score': overall_score,
            'detailed_scores': scores,
            'weights': weights
        }

    def evaluate_id_consistency(self):
        """IDの一貫性を評価"""
        id_changes = 0
        total_tracks = len(self.position_history)
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 1:
                # 位置の急激な変化を検出
                for i in range(1, len(positions)):
                    prev_pos = positions[i-1]
                    curr_pos = positions[i]
                    distance = np.sqrt((curr_pos[1] - prev_pos[1])**2 + (curr_pos[2] - prev_pos[2])**2)
                    
                    # 異常に大きな移動をIDスイッチとしてカウント
                    if distance > 100:  # 閾値
                        id_changes += 1
        
        consistency_score = 1 - (id_changes / max(total_tracks, 1))
        return consistency_score

    def evaluate_trajectory_continuity(self):
        """軌跡の連続性を評価"""
        total_fragments = 0
        total_tracks = len(self.position_history)
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 2:
                fragments = 0
                for i in range(1, len(positions)):
                    frame_diff = positions[i][0] - positions[i-1][0]
                    if frame_diff > 1:  # フレームが飛んでいる
                        fragments += 1
                total_fragments += fragments
        
        continuity_score = 1 - (total_fragments / max(total_tracks, 1))
        return continuity_score

    def evaluate_detection_stability(self):
        """検出の安定性を評価"""
        detection_counts = []
        
        for track_id, positions in self.position_history.items():
            detection_counts.append(len(positions))
        
        if detection_counts:
            mean_detections = np.mean(detection_counts)
            std_detections = np.std(detection_counts)
            stability_score = mean_detections / (mean_detections + std_detections)
            return stability_score
        return 0

    def evaluate_position_consistency(self):
        """位置の一貫性を評価"""
        consistency_scores = []
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 2:
                # 移動の一貫性を計算
                movements = []
                for i in range(1, len(positions)):
                    dx = positions[i][1] - positions[i-1][1]
                    dy = positions[i][2] - positions[i-1][2]
                    movement = np.sqrt(dx**2 + dy**2)
                    movements.append(movement)
                
                if movements:
                    movement_std = np.std(movements)
                    movement_mean = np.mean(movements)
                    consistency = movement_mean / (movement_mean + movement_std)
                    consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0

    def evaluate_lstm_prediction_accuracy(self):
        """LSTM予測の精度を評価"""
        prediction_errors = []
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 10:  # 十分な履歴がある場合
                for i in range(10, len(positions)):
                    if positions[i][3] == "lstm_prediction":  # LSTM予測
                        # 次のフレームの実際の位置と比較
                        if i < len(positions) - 1:
                            predicted_pos = [positions[i][1], positions[i][2]]
                            actual_pos = [positions[i+1][1], positions[i+1][2]]
                            error = np.sqrt((actual_pos[0] - predicted_pos[0])**2 + 
                                          (actual_pos[1] - predicted_pos[1])**2)
                            prediction_errors.append(error)
        
        if prediction_errors:
            mean_error = np.mean(prediction_errors)
            return 1 / (1 + mean_error)  # 0-1のスコアに変換
        return 0

    def calculate_tracking_success_rate(self):
        """追跡成功率を計算"""
        successful_tracks = 0
        total_tracks = len(self.position_history)
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 10:  # 10フレーム以上追跡できた場合
                successful_tracks += 1
        
        return successful_tracks / max(total_tracks, 1)

    def calculate_average_tracking_duration(self):
        """平均追跡時間を計算"""
        durations = []
        
        for track_id, positions in self.position_history.items():
            if len(positions) > 1:
                duration = positions[-1][0] - positions[0][0]
                durations.append(duration)
        
        return np.mean(durations) if durations else 0

    def visualize_tracking_quality(self, save_plots=True):
        """トラッキング品質を可視化"""
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        # 結果フォルダを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"tracking_evaluation_results_{timestamp}"
        os.makedirs(results_folder, exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 軌跡の長さ分布
        track_lengths = [len(positions) for positions in self.position_history.values()]
        ax1.hist(track_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Track Length Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Detections')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. 移動距離の分布
        all_movements = []
        for positions in self.position_history.values():
            for i in range(1, len(positions)):
                dx = positions[i][1] - positions[i-1][1]
                dy = positions[i][2] - positions[i-1][2]
                movement = np.sqrt(dx**2 + dy**2)
                all_movements.append(movement)
        
        ax2.hist(all_movements, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Movement Distance Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Distance (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. フレーム間隔の分布
        frame_gaps = []
        for positions in self.position_history.values():
            for i in range(1, len(positions)):
                gap = positions[i][0] - positions[i-1][0]
                frame_gaps.append(gap)
        
        ax3.hist(frame_gaps, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_title('Frame Gap Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Frame Gap')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. 評価スコア
        performance = self.evaluate_tracking_performance()
        scores = performance['detailed_scores']
        labels = list(scores.keys())
        values = list(scores.values())
        
        bars = ax4.bar(labels, values, alpha=0.7, color=['gold', 'lightblue', 'lightgreen', 'lightcoral', 'plum'])
        ax4.set_title('Tracking Quality Scores', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # スコア値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(results_folder, 'tracking_quality_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"トラッキング品質分析図を保存しました: {plot_path}")
        
        plt.show()
        
        return results_folder

    def create_detailed_evaluation_report(self, results_folder):
        """詳細な評価レポートを作成"""
        import json
        import os
        from datetime import datetime
        
        # 評価データを収集
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'video_info': {
                'total_frames': self.mot_evaluation.get('total_frames', 0),
                'total_detections': self.mot_evaluation.get('total_detections', 0),
                'active_tracks': len(self.active_tracks),
                'discarded_ids': len(self.discarded_ids)
            },
            'tracking_performance': self.evaluate_tracking_performance(),
            'additional_metrics': {
                'tracking_success_rate': self.calculate_tracking_success_rate(),
                'average_tracking_duration': self.calculate_average_tracking_duration(),
                'total_tracks': len(self.position_history),
                'lstm_predictions': sum(1 for positions in self.position_history.values() 
                                      for pos in positions if pos[3] == "lstm_prediction")
            },
            'track_statistics': {
                'track_lengths': [len(positions) for positions in self.position_history.values()],
                'track_durations': [positions[-1][0] - positions[0][0] 
                                   for positions in self.position_history.values() 
                                   if len(positions) > 1],
                'movement_distances': self._calculate_all_movement_distances(),
                'frame_gaps': self._calculate_all_frame_gaps()
            }
        }
        
        # JSONファイルとして保存
        report_path = os.path.join(results_folder, 'detailed_evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        # テキストレポートも作成
        text_report_path = os.path.join(results_folder, 'evaluation_summary.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=== トラッキング評価レポート ===\n\n")
            f.write(f"評価日時: {evaluation_data['timestamp']}\n")
            f.write(f"総フレーム数: {evaluation_data['video_info']['total_frames']}\n")
            f.write(f"総検出数: {evaluation_data['video_info']['total_detections']}\n")
            f.write(f"アクティブトラック数: {evaluation_data['video_info']['active_tracks']}\n")
            f.write(f"破棄されたID数: {evaluation_data['video_info']['discarded_ids']}\n\n")
            
            f.write("=== トラッキング性能スコア ===\n")
            performance = evaluation_data['tracking_performance']
            f.write(f"総合スコア: {performance['overall_score']:.3f}\n")
            f.write(f"ID一貫性: {performance['detailed_scores']['id_consistency']:.3f}\n")
            f.write(f"軌跡連続性: {performance['detailed_scores']['trajectory_continuity']:.3f}\n")
            f.write(f"検出安定性: {performance['detailed_scores']['detection_stability']:.3f}\n")
            f.write(f"位置一貫性: {performance['detailed_scores']['position_consistency']:.3f}\n")
            f.write(f"LSTM精度: {performance['detailed_scores']['lstm_accuracy']:.3f}\n\n")
            
            f.write("=== 追加指標 ===\n")
            additional = evaluation_data['additional_metrics']
            f.write(f"追跡成功率: {additional['tracking_success_rate']:.3f}\n")
            f.write(f"平均追跡時間: {additional['average_tracking_duration']:.1f} フレーム\n")
            f.write(f"総トラック数: {additional['total_tracks']}\n")
            f.write(f"LSTM予測数: {additional['lstm_predictions']}\n")
        
        print(f"詳細評価レポートを保存しました:")
        print(f"  - JSON: {report_path}")
        print(f"  - テキスト: {text_report_path}")
        
        return evaluation_data

    def _calculate_all_movement_distances(self):
        """全ての移動距離を計算"""
        all_movements = []
        for positions in self.position_history.values():
            for i in range(1, len(positions)):
                dx = positions[i][1] - positions[i-1][1]
                dy = positions[i][2] - positions[i-1][2]
                movement = np.sqrt(dx**2 + dy**2)
                all_movements.append(movement)
        return all_movements

    def _calculate_all_frame_gaps(self):
        """全てのフレーム間隔を計算"""
        frame_gaps = []
        for positions in self.position_history.values():
            for i in range(1, len(positions)):
                gap = positions[i][0] - positions[i-1][0]
                frame_gaps.append(gap)
        return frame_gaps

    def create_advanced_visualization(self, results_folder):
        """高度な可視化を作成"""
        import matplotlib.pyplot as plt
        import os
        
        # 1. トラックの時系列分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # トラック数の時系列
        frame_track_counts = {}
        for track_id, positions in self.position_history.items():
            for pos in positions:
                frame = pos[0]
                if frame not in frame_track_counts:
                    frame_track_counts[frame] = 0
                frame_track_counts[frame] += 1
        
        frames = sorted(frame_track_counts.keys())
        counts = [frame_track_counts[f] for f in frames]
        
        ax1.plot(frames, counts, linewidth=2, color='blue', alpha=0.7)
        ax1.set_title('Active Tracks Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Number of Active Tracks')
        ax1.grid(True, alpha=0.3)
        
        # 移動速度の分布
        speeds = []
        for positions in self.position_history.values():
            for i in range(1, len(positions)):
                dx = positions[i][1] - positions[i-1][1]
                dy = positions[i][2] - positions[i-1][2]
                frame_diff = positions[i][0] - positions[i-1][0]
                if frame_diff > 0:
                    speed = np.sqrt(dx**2 + dy**2) / frame_diff
                    speeds.append(speed)
        
        ax2.hist(speeds, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('Movement Speed Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Speed (pixels/frame)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # トラックの生存時間分布
        track_lifetimes = []
        for positions in self.position_history.values():
            if len(positions) > 1:
                lifetime = positions[-1][0] - positions[0][0]
                track_lifetimes.append(lifetime)
        
        ax3.hist(track_lifetimes, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_title('Track Lifetime Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Lifetime (frames)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 検出ソースの分布
        source_counts = {'yolo_detection': 0, 'lstm_prediction': 0, 'reused_id': 0, 'new_id': 0}
        for positions in self.position_history.values():
            for pos in positions:
                source = pos[3]
                if source in source_counts:
                    source_counts[source] += 1
        
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        colors = ['gold', 'lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax4.bar(sources, counts, alpha=0.7, color=colors, edgecolor='black')
        ax4.set_title('Detection Source Distribution', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # カウント値をバーの上に表示
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(results_folder, 'advanced_tracking_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"高度なトラッキング分析図を保存しました: {plot_path}")
        
        plt.show()

    def run_complete_evaluation(self, video_path):
        """完全な評価を実行"""
        print("=== トラッキング評価を開始 ===")
        
        # ビデオ処理
        self.process_video(video_path)
        
        # 結果フォルダを作成
        results_folder = self.visualize_tracking_quality(save_plots=True)
        
        # 詳細レポート作成
        evaluation_data = self.create_detailed_evaluation_report(results_folder)
        
        # 高度な可視化
        self.create_advanced_visualization(results_folder)
        
        print(f"\n=== 評価完了 ===")
        print(f"結果フォルダ: {results_folder}")
        print(f"総合スコア: {evaluation_data['tracking_performance']['overall_score']:.3f}")
        
        return results_folder, evaluation_data

if __name__ == "__main__":
    # モデルのパスを指定
    model_path = "./train_results/weights/best.pt"
    lstm_model_path = "best_lstm_model.pth"  # 学習済みLSTMモデルのパス
    
    # トラッカーの初期化
    tracker = ObjectTracker(model_path, lstm_model_path)
    
    # 動画のパスを指定
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3D_left.mp4"
    
    # 完全な評価を実行（Ground Truthデータ不要）
    results_folder, evaluation_data = tracker.run_complete_evaluation(video_path)
    
    print(f"\n=== 評価結果サマリー ===")
    print(f"結果フォルダ: {results_folder}")
    print(f"総合スコア: {evaluation_data['tracking_performance']['overall_score']:.3f}")
    print(f"追跡成功率: {evaluation_data['additional_metrics']['tracking_success_rate']:.3f}")
    print(f"平均追跡時間: {evaluation_data['additional_metrics']['average_tracking_duration']:.1f} フレーム") 