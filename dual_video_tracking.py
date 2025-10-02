import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import os

class LSTMTracker(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(LSTMTracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # x, y, w, h の予測

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class DualObjectTracker:
    def __init__(self, model_path, lstm_model_path=None, sequence_length=10):
        # YOLOモデルの読み込み
        self.yolo = YOLO(model_path)
        
        # LSTMモデルの初期化
        self.lstm = LSTMTracker()
        if lstm_model_path and os.path.exists(lstm_model_path):
            self.lstm.load_state_dict(torch.load(lstm_model_path))
            print(f"Loaded LSTM model from {lstm_model_path}")
        self.sequence_length = sequence_length
        
        # 左と右の動画の追跡履歴を保存
        self.left_track_history = {}
        self.right_track_history = {}
        
        # ID管理システム（左から順番にIDを振る）
        self.next_id = 1
        self.left_active_tracks = {}
        self.right_active_tracks = {}
        self.left_missed_frames = {}
        self.right_missed_frames = {}
        
    def get_new_id(self):
        """新しいIDを取得"""
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def release_id(self, obj_id, is_left=True):
        """IDを解放"""
        if is_left:
            if obj_id in self.left_active_tracks:
                del self.left_active_tracks[obj_id]
            if obj_id in self.left_missed_frames:
                del self.left_missed_frames[obj_id]
        else:
            if obj_id in self.right_active_tracks:
                del self.right_active_tracks[obj_id]
            if obj_id in self.right_missed_frames:
                del self.right_missed_frames[obj_id]
    
    def preprocess_detection(self, detection):
        # YOLOの検出結果をLSTMの入力形式に変換
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections, is_left=True):
        # 現在のフレームで検出された物体のIDを記録
        current_detections = set()
        active_tracks = self.left_active_tracks if is_left else self.right_active_tracks
        missed_frames = self.left_missed_frames if is_left else self.right_missed_frames
        track_history = self.left_track_history if is_left else self.right_track_history
        
        for det in detections:
            # 物体の位置情報を取得
            bbox = det.xywh[0].cpu().numpy()  # x, y, w, h
            
            # 既存の追跡とマッチング
            matched = False
            for track_id, track_info in list(active_tracks.items()):
                if track_id in current_detections:
                    continue
                    
                # 既存の追跡と新しい検出の距離を計算
                old_bbox = track_history[track_id][-1]
                distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                
                # 距離が閾値以下の場合、同じ物体とみなす
                if distance < 200:
                    current_detections.add(track_id)
                    matched = True
                    missed_frames[track_id] = 0
                    track_history[track_id].append(self.preprocess_detection(bbox))
                    break
            
            # 新しい物体の場合
            if not matched:
                new_id = self.get_new_id()
                current_detections.add(new_id)
                active_tracks[new_id] = bbox
                track_history[new_id] = deque(maxlen=self.sequence_length)
                track_history[new_id].append(self.preprocess_detection(bbox))
                missed_frames[new_id] = 0
        
        # 見失った物体の処理
        for track_id in list(active_tracks.keys()):
            if track_id not in current_detections:
                missed_frames[track_id] = missed_frames.get(track_id, 0) + 1
                if missed_frames[track_id] > 30:  # 30フレーム以上見失った場合
                    self.release_id(track_id, is_left)

    def process_dual_videos(self, left_video_path, right_video_path):
        # 左と右の動画を同時に処理
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        
        # 動画の設定を取得
        width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(left_cap.get(cv2.CAP_PROP_FPS))
        
        # 出力用のVideoWriterを設定
        left_output_path = "video/tracking_video_left.mp4"
        right_output_path = "video/tracking_video_right.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        left_out = cv2.VideoWriter(left_output_path, fourcc, fps, (width, height))
        right_out = cv2.VideoWriter(right_output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while left_cap.isOpened() and right_cap.isOpened():
            ret_left, left_frame = left_cap.read()
            ret_right, right_frame = right_cap.read()
            
            if not ret_left or not ret_right:
                break
                
            # 左動画の処理
            left_results = self.yolo.track(left_frame, persist=True)
            if left_results[0].boxes is not None:
                self.update_tracking(frame_count, left_results[0].boxes, is_left=True)
                self.visualize_tracking(left_frame, left_results[0].boxes, is_left=True)
            
            # 右動画の処理
            right_results = self.yolo.track(right_frame, persist=True)
            if right_results[0].boxes is not None:
                self.update_tracking(frame_count, right_results[0].boxes, is_left=False)
                self.visualize_tracking(right_frame, right_results[0].boxes, is_left=False)
            
            # 座標差の計算と表示
            self.calculate_coordinate_differences(frame_count)
            
            # フレームを出力ファイルに書き込み
            left_out.write(left_frame)
            right_out.write(right_frame)
            
            # 結果の表示
            cv2.imshow("Left Tracking", left_frame)
            cv2.imshow("Right Tracking", right_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        left_cap.release()
        right_cap.release()
        left_out.release()
        right_out.release()
        cv2.destroyAllWindows()
        print(f"Left tracking result saved to {left_output_path}")
        print(f"Right tracking result saved to {right_output_path}")

    def visualize_tracking(self, frame, boxes, is_left=True):
        """追跡結果を可視化"""
        active_tracks = self.left_active_tracks if is_left else self.right_active_tracks
        used_ids = set()
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 最も近い追跡IDを探す
            min_distance = float('inf')
            closest_id = None
            
            for track_id, track_info in active_tracks.items():
                if track_id in used_ids:
                    continue
                distance = np.linalg.norm(np.array([center_x, center_y]) - track_info[:2])
                if distance < min_distance:
                    min_distance = distance
                    closest_id = track_id
            
            if closest_id is not None:
                used_ids.add(closest_id)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {closest_id}", 
                          (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def calculate_coordinate_differences(self, frame_count):
        """同じIDの物体の座標差を計算"""
        common_ids = set(self.left_active_tracks.keys()) & set(self.right_active_tracks.keys())
        
        if common_ids:
            print(f"\nFrame {frame_count} - Coordinate differences:")
            for obj_id in common_ids:
                left_bbox = self.left_active_tracks[obj_id]
                right_bbox = self.right_active_tracks[obj_id]
                
                # 中心座標の差を計算
                left_center_x, left_center_y = left_bbox[:2]
                right_center_x, right_center_y = right_bbox[:2]
                
                diff_x = abs(left_center_x - right_center_x)
                diff_y = abs(left_center_y - right_center_y)
                
                print(f"ID {obj_id}: Δx = {diff_x:.2f}, Δy = {diff_y:.2f}")
        else:
            print(f"\nFrame {frame_count}: No common objects found")

if __name__ == "__main__":
    # モデルのパスを指定
    model_path = "./train_results/weights/best.pt"
    lstm_model_path = "best_lstm_model.pth"
    
    # トラッカーの初期化
    tracker = DualObjectTracker(model_path, lstm_model_path)
    
    # 動画のパスを指定
    left_video_path = "./video/processed_train_video_left.mp4"
    right_video_path = "./video/processed_train_video_right.mp4"
    
    # ファイルの存在確認
    if not os.path.exists(left_video_path):
        print(f"Error: Left video file {left_video_path} not found!")
        exit(1)
    
    if not os.path.exists(right_video_path):
        print(f"Error: Right video file {right_video_path} not found!")
        exit(1)
    
    # 二つの動画の同時処理開始
    tracker.process_dual_videos(left_video_path, right_video_path) 