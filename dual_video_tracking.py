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
    def __init__(self, model_path, lstm_model_path=None, sequence_length=10, max_fish=20):
        # YOLOモデルの読み込み（model_pathがNoneの場合はスキップ）
        if model_path is not None:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = None
        
        # LSTMモデルの初期化
        self.lstm = LSTMTracker()
        if lstm_model_path and os.path.exists(lstm_model_path):
            self.lstm.load_state_dict(torch.load(lstm_model_path))
            print(f"Loaded LSTM model from {lstm_model_path}")
        self.sequence_length = sequence_length
        self.max_fish = max_fish
        
        # 左と右の動画の追跡履歴を保存
        self.left_track_history = {}
        self.right_track_history = {}
        
        # 改善されたID管理システム
        self.next_id = 1
        self.left_active_tracks = {}
        self.right_active_tracks = {}
        self.left_missed_frames = {}
        self.right_missed_frames = {}
        
        # 見失った魚の情報を記憶するシステム（左右それぞれ）
        self.left_lost_fish = {}
        self.right_lost_fish = {}
        self.reuse_distance_threshold = 150
        self.max_lost_frames = 60
        
        # 破棄されたIDを記録するシステム（左右共通）
        self.discarded_ids = set()  # 破棄されたIDのセット
        self.max_discarded_ids = 50  # 記録する最大破棄ID数
        
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
    
    def release_id(self, obj_id, current_frame, is_left=True):
        """IDを解放し、見失った魚として記録"""
        if is_left:
            if obj_id in self.left_active_tracks:
                # 見失った魚の情報を保存
                self.left_lost_fish[obj_id] = {
                    'last_position': self.left_active_tracks[obj_id].copy(),
                    'lost_frames': 0,
                    'last_seen_frame': current_frame
                }
                del self.left_active_tracks[obj_id]
            if obj_id in self.left_missed_frames:
                del self.left_missed_frames[obj_id]
            if obj_id in self.left_track_history:
                del self.left_track_history[obj_id]
        else:
            if obj_id in self.right_active_tracks:
                # 見失った魚の情報を保存
                self.right_lost_fish[obj_id] = {
                    'last_position': self.right_active_tracks[obj_id].copy(),
                    'lost_frames': 0,
                    'last_seen_frame': current_frame
                }
                del self.right_active_tracks[obj_id]
            if obj_id in self.right_missed_frames:
                del self.right_missed_frames[obj_id]
            if obj_id in self.right_track_history:
                del self.right_track_history[obj_id]
    
    def find_reusable_id(self, new_position, current_frame, is_left=True):
        """見失った魚の中で再利用可能なIDを探す"""
        lost_fish = self.left_lost_fish if is_left else self.right_lost_fish
        reusable_id = None
        min_distance = float('inf')
        
        for fish_id, fish_info in list(lost_fish.items()):
            # フレーム数が上限を超えている場合は破棄されたIDとして記録
            if fish_info['lost_frames'] > self.max_lost_frames:
                del lost_fish[fish_id]
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
        """破棄されたIDを記録"""
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
    
    def update_tracking(self, frame_id, detections, is_left=True):
        # 見失った魚のフレーム数を更新
        lost_fish = self.left_lost_fish if is_left else self.right_lost_fish
        for fish_id in lost_fish:
            lost_fish[fish_id]['lost_frames'] += 1
        
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
            min_distance = float('inf')
            best_match_id = None
            
            for track_id, track_info in list(active_tracks.items()):
                if track_id in current_detections:
                    continue
                    
                # 既存の追跡と新しい検出の距離を計算
                if track_id in track_history and len(track_history[track_id]) > 0:
                    old_bbox = track_history[track_id][-1]
                    distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                    
                    # 距離が閾値以下の場合、候補として記録
                    if distance < 200:
                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = track_id
            
            # 最適なマッチを見つけた場合
            if best_match_id is not None:
                current_detections.add(best_match_id)
                matched = True
                missed_frames[best_match_id] = 0
                track_history[best_match_id].append(self.preprocess_detection(bbox))
                active_tracks[best_match_id] = bbox
            
            # 既存の追跡にマッチしなかった場合、見失った魚のIDを再利用
            if not matched:
                reusable_id = self.find_reusable_id(bbox, frame_id, is_left)
                if reusable_id is not None:
                    # 見失った魚のIDを再利用
                    current_detections.add(reusable_id)
                    active_tracks[reusable_id] = bbox
                    track_history[reusable_id] = deque(maxlen=self.sequence_length)
                    track_history[reusable_id].append(self.preprocess_detection(bbox))
                    missed_frames[reusable_id] = 0
                    # 見失った魚リストから削除
                    del lost_fish[reusable_id]
                    side = "left" if is_left else "right"
                    print(f"Reused ID {reusable_id} for fish near position {bbox[:2]} ({side})")
                else:
                    # 新しいIDを作成
                    new_id = self.get_new_id()
                    current_detections.add(new_id)
                    active_tracks[new_id] = bbox
                    track_history[new_id] = deque(maxlen=self.sequence_length)
                    track_history[new_id].append(self.preprocess_detection(bbox))
                    missed_frames[new_id] = 0
                    side = "left" if is_left else "right"
                    print(f"Created new ID {new_id} for fish at position {bbox[:2]} ({side})")
        
        # 見失った物体の処理
        for track_id in list(active_tracks.keys()):
            if track_id not in current_detections:
                missed_frames[track_id] = missed_frames.get(track_id, 0) + 1
                if missed_frames[track_id] > 30:  # 30フレーム以上見失った場合
                    missed_count = missed_frames[track_id]
                    self.release_id(track_id, frame_id, is_left)
                    side = "left" if is_left else "right"
                    print(f"Lost fish ID {track_id} after {missed_count} frames ({side})")

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