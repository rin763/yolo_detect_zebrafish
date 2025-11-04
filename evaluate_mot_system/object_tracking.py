# 以下はobject_tracking.pyのコードです。
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import torch
from lstm_kalman_tracker import LSTMKalmanTracker
from mot_evaluator import MOTEvaluator
import json

class ObjectTracker:
    def __init__(self, model_path, sequence_length=10, max_fish=10, use_lstm=True, enable_evaluation=False):
        if model_path is not None:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = None
        
        self.sequence_length = sequence_length
        self.max_fish = max_fish  # 最大トラッキング数（デフォルト10）
        self.use_lstm = use_lstm
        
        # ===== 評価機能 =====
        self.enable_evaluation = enable_evaluation
        if self.enable_evaluation:
            self.evaluator = MOTEvaluator(iou_threshold=0.5, distance_threshold=50)
            print("MOT Evaluator initialized")
        
        # LSTM+カルマンフィルタ強化トラッカーの初期化
        if self.use_lstm:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.lstm_kalman_tracker = LSTMKalmanTracker(
                sequence_length=sequence_length,
                device=device
            )
            print(f"LSTM+Kalman tracker initialized on {device}")
        
        # 物体の追跡履歴を保存
        self.track_history = {}
        self.position_history = {}
        
        # 改善されたID管理システム
        self.next_id = 1
        self.active_tracks = {}
        self.missed_frames = {}
        
        # ===== 改善1: 見失った魚の管理を緩和 =====
        self.lost_fish = {}
        self.reuse_distance_threshold = 400  # 300 → 400に拡大（より遠くても再検出）
        self.max_lost_frames = 120  # 90 → 120に延長（より長く記憶）
        
        # ===== 改善2: マッチングスコアの閾値を大幅緩和 =====
        self.reuse_score_threshold = 0.02
        self.lstm_prediction_range = 200
        self.distance_score_weight = 0.7
        self.lstm_score_weight = 0.3
        
        # 破棄されたIDを記録するシステム
        self.discarded_ids = {}
        self.max_discarded_ids = 10
        self.discarded_id_reuse_distance = 200  # 150 → 200に拡大
        
        # ===== 改善3: マッチングパラメータの調整 =====
        self.use_direction_matching = True
        self.direction_weight = 0.1  # 0.2 → 0.1に減少（方向の影響を減らす）
        self.distance_weight = 0.85  # 0.8 → 0.85に増加（距離を最重視）
        self.direction_threshold = 1.5  # 1.0 → 1.5に緩和（約86度まで許容）
        
        # ===== 改善4: LSTM影響の緩和 =====
        self.lstm_matching_threshold = 0.4  # 0.6 → 0.4に緩和
        self.prediction_weight = 0.05  # 0.5 → 0.05に大幅減少（LSTMの影響を最小化）
        
        # ===== 改善5: 最大検出数の制限追加 =====
        self.max_detections = max_fish  # 最大検出数を設定
        self.max_active_tracks = max_fish  # 最大同時トラッキング数を設定
        
        # ===== 改善6: 信頼度フィルタリング追加 =====
        self.min_detection_confidence = 0.3  # YOLO検出の最小信頼度
        
    
    def update_position_history(self, track_id, frame_id, x, y, source="detection"):
        """IDごとの位置履歴を更新（動きの方向も計算）"""
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        direction_rad = 0.0
        if len(self.position_history[track_id]) > 0:
            prev_frame, prev_x, prev_y, prev_source, prev_direction = self.position_history[track_id][-1]
            dx = x - prev_x
            dy = y - prev_y
            
            if dx != 0 or dy != 0:
                direction_rad = np.arctan2(dy, dx)
        
        self.position_history[track_id].append((frame_id, x, y, source, direction_rad))
    
        
    def get_new_id(self):
        """新しいIDを取得（破棄されたIDを優先的に再利用）"""
        if self.discarded_ids:
            reused_id = min(self.discarded_ids.keys())
            del self.discarded_ids[reused_id]
            print(f"Reusing discarded ID: {reused_id}")
            return reused_id
        
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def add_discarded_id(self, fish_id, position, frame_id):
        """破棄されたIDを記録（位置情報も含む）"""
        self.discarded_ids[fish_id] = {
            'position': position[:2].copy(),
            'discarded_frame': frame_id
        }
        
        if len(self.discarded_ids) > self.max_discarded_ids:
            oldest_id = min(self.discarded_ids.keys(), 
                          key=lambda x: self.discarded_ids[x]['discarded_frame'])
            del self.discarded_ids[oldest_id]
        
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
        
        last_positions = self.position_history[track_id][-2:]
        prev_frame, prev_x, prev_y, prev_source, prev_direction = last_positions[0]
        curr_frame, curr_x, curr_y, curr_source, curr_direction = last_positions[1]
        
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
        
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        return abs(diff)
    
    def compute_enhanced_matching_cost(self, detection, track_id):
        """距離と方向を考慮したマッチングコストを計算（LSTM予測も含む）"""
        if track_id not in self.track_history or len(self.track_history[track_id]) == 0:
            return float('inf')
        
        # 基本的な距離コスト
        old_bbox = self.track_history[track_id][-1]
        distance = np.linalg.norm(detection[:2] - old_bbox[:2])
        
        # ===== 改善7: 距離閾値を拡大 =====
        if distance > 300:  # 250 → 300に拡大（より遠くてもマッチング可能）
            return float('inf')
        
        # 方向コスト
        direction_cost = 0.0
        if self.use_direction_matching:
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
                        
                        # ===== 改善8: 方向差のペナルティを緩和 =====
                        if direction_diff > self.direction_threshold:
                            direction_cost = direction_diff * 5  # 10 → 5に減少
                        else:
                            direction_cost = direction_diff * 0.5  # 軽微なペナルティ
        
        # LSTM+カルマンフィルタ予測コスト
        lstm_cost = 0.0
        lstm_confidence = 0.0
        if self.use_lstm and track_id in self.position_history:
            self.lstm_kalman_tracker.update_track_sequence(track_id, self.position_history[track_id])
            detection_position = detection[:2]
            hybrid_score = self.lstm_kalman_tracker.compute_hybrid_reid_score(track_id, detection_position)
            lstm_confidence = hybrid_score
            
            # ===== 改善9: LSTMペナルティを大幅緩和 =====
            if hybrid_score < self.lstm_matching_threshold:
                lstm_cost = (self.lstm_matching_threshold - hybrid_score) * 10  # 20 → 10に減少
        
        # 総合コスト（距離を最重視）
        total_cost = (self.distance_weight * distance + 
                     self.direction_weight * direction_cost + 
                     self.prediction_weight * lstm_cost)
        
        return total_cost, distance, direction_cost, lstm_cost, lstm_confidence
        
    def release_id(self, obj_id, current_frame):
        """IDを解放し、見失った魚として記録"""
        if obj_id in self.active_tracks:
            self.lost_fish[obj_id] = {
                'last_position': self.active_tracks[obj_id].copy(),
                'lost_frames': 0,
                'last_seen_frame': current_frame,
                'predicted_positions': []
            }
            
            if self.use_lstm and obj_id in self.position_history:
                predicted_positions = self.lstm_kalman_tracker.predict_lost_track_positions(obj_id, frames_ahead=15)  # 10 → 15に増加
                self.lost_fish[obj_id]['predicted_positions'] = predicted_positions
                print(f"Generated {len(predicted_positions)} hybrid predicted positions for lost fish {obj_id}")
            
            self.add_discarded_id(obj_id, self.active_tracks[obj_id], current_frame)
            
            del self.active_tracks[obj_id]
        if obj_id in self.missed_frames:
            del self.missed_frames[obj_id]
        if obj_id in self.track_history:
            del self.track_history[obj_id]
    
    def find_reusable_id(self, new_position, current_frame):
        """見失った魚の中で再利用可能なIDを探す（大幅緩和版）"""
        reusable_id = None
        min_distance = float('inf')
        best_total_score = 0.0
        
        for fish_id, fish_info in list(self.lost_fish.items()):
            # 保持期間チェック
            if fish_info['lost_frames'] > self.max_lost_frames:
                del self.lost_fish[fish_id]
                self.add_discarded_id(fish_id, fish_info['last_position'], fish_info['last_seen_frame'])
                continue
            
            # 距離計算
            last_pos = fish_info['last_position']
            distance = np.linalg.norm(np.array(new_position[:2]) - np.array(last_pos[:2]))
            
            # ===== 改善10: 距離スコアの計算を大幅緩和 =====
            if distance < self.reuse_distance_threshold:
                distance_score = 1 - (distance / self.reuse_distance_threshold)
            else:
                # 閾値の3倍までスコアを与える
                distance_score = max(0, 0.5 * (1 - distance / (self.reuse_distance_threshold * 3)))
            
            # LSTM予測スコア
            lstm_score = 0.0
            if self.use_lstm and 'predicted_positions' in fish_info and fish_info['predicted_positions']:
                min_pred_distance = float('inf')
                for pred_pos in fish_info['predicted_positions']:
                    pred_distance = np.linalg.norm(np.array(new_position[:2]) - pred_pos)
                    min_pred_distance = min(min_pred_distance, pred_distance)
                
                if min_pred_distance < self.lstm_prediction_range:
                    lstm_score = max(0, 1 - min_pred_distance / self.lstm_prediction_range)
            
            # 総合スコア
            total_score = (self.distance_score_weight * distance_score + 
                        self.lstm_score_weight * lstm_score)
            
            # ===== 改善11: 閾値判定を大幅緩和 =====
            if total_score > self.reuse_score_threshold:
                if reusable_id is None or total_score > best_total_score:
                    min_distance = distance
                    reusable_id = fish_id
                    best_total_score = total_score
        
        if reusable_id is not None:
            print(f"✓ Found reusable ID {reusable_id} with distance {min_distance:.2f} and score {best_total_score:.3f}")
        
        return reusable_id
    
    def preprocess_detection(self, detection):
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections, ground_truth=None):
        """追跡を更新"""
        # 見失った魚のフレーム数を更新
        for fish_id in self.lost_fish:
            self.lost_fish[fish_id]['lost_frames'] += 1
        
        # ===== 改善12: 信頼度フィルタリング追加 =====
        filtered_detections = []
        for det in detections:
            if det.conf[0] >= self.min_detection_confidence:
                filtered_detections.append(det)
            else:
                print(f"⚠ Filtered out low confidence detection: {det.conf[0]:.2f}")
        
        # ===== 改善13: 最大検出数の制限 =====
        if len(filtered_detections) > self.max_detections:
            # 信頼度順にソート
            filtered_detections = sorted(filtered_detections, key=lambda x: x.conf[0], reverse=True)
            filtered_detections = filtered_detections[:self.max_detections]
            print(f"⚠ Limited detections to max {self.max_detections}")
        
        detections = filtered_detections
        
        # ===== 追加: 最大トラッキング数のチェック =====
        if len(self.active_tracks) >= self.max_active_tracks:
            print(f"⚠ Already tracking maximum {self.max_active_tracks} objects")
        
        current_detections = set()
        
        for det in detections:
            bbox = det.xywh[0].cpu().numpy()
            
            # 既存の追跡とマッチング
            matched = False
            min_cost = float('inf')
            best_match_id = None
            
            for track_id, track_info in list(self.active_tracks.items()):
                if track_id in current_detections:
                    continue
                
                if self.use_direction_matching:
                    cost_result = self.compute_enhanced_matching_cost(bbox, track_id)
                    if isinstance(cost_result, tuple) and len(cost_result) >= 5:
                        total_cost, distance, direction_cost, lstm_cost, lstm_confidence = cost_result
                    else:
                        total_cost = cost_result
                        distance = float('inf')
                        direction_cost = 0.0
                        lstm_cost = 0.0
                        lstm_confidence = 0.0
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_match_id = track_id
                else:
                    if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                        old_bbox = self.track_history[track_id][-1]
                        distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                
                        if distance < 300 and distance < min_cost:  # 250 → 300に拡大
                            min_cost = distance
                            best_match_id = track_id
            
            # ===== 改善14: マッチングコスト閾値を緩和 =====
            if best_match_id is not None and min_cost < 250:  # 200 → 250に拡大（マッチングしやすく）
                current_detections.add(best_match_id)
                matched = True
                self.missed_frames[best_match_id] = 0
                self.track_history[best_match_id].append(self.preprocess_detection(bbox))
                self.active_tracks[best_match_id] = bbox
                self.update_position_history(best_match_id, frame_id, bbox[0], bbox[1], "detection")
                print(f"✓ Matched detection to track {best_match_id} (cost: {min_cost:.2f})")
            elif best_match_id is not None:
                print(f"✗ Match cost too high: {min_cost:.2f} > 250")
            
            # 既存の追跡にマッチしなかった場合
            if not matched:
                # ===== 追加: 最大トラッキング数に達している場合は新規追加をスキップ =====
                if len(self.active_tracks) >= self.max_active_tracks:
                    print(f"⛔ Cannot add new track: Already tracking {self.max_active_tracks} objects (max limit)")
                    continue  # 新しいIDを作成せずスキップ
                
                reusable_id = self.find_reusable_id(bbox, frame_id)
                if reusable_id is not None:
                    current_detections.add(reusable_id)
                    self.active_tracks[reusable_id] = bbox
                    self.track_history[reusable_id] = deque(maxlen=self.sequence_length)
                    self.track_history[reusable_id].append(self.preprocess_detection(bbox))
                    self.missed_frames[reusable_id] = 0
                    del self.lost_fish[reusable_id]
                    self.update_position_history(reusable_id, frame_id, bbox[0], bbox[1], "detection")
                    print(f"✓ Reused lost ID {reusable_id}")
                else:
                    result = self.find_nearest_discarded_id(bbox)
                    if result is not None:
                        nearest_discarded_id, distance = result
                        del self.discarded_ids[nearest_discarded_id]
                        
                        current_detections.add(nearest_discarded_id)
                        self.active_tracks[nearest_discarded_id] = bbox
                        self.track_history[nearest_discarded_id] = deque(maxlen=self.sequence_length)
                        self.track_history[nearest_discarded_id].append(self.preprocess_detection(bbox))
                        self.missed_frames[nearest_discarded_id] = 0
                        self.update_position_history(nearest_discarded_id, frame_id, bbox[0], bbox[1], "detection")
                        print(f"✓ Reused discarded ID {nearest_discarded_id}")
                    else:
                        new_id = self.get_new_id()
                        current_detections.add(new_id)
                        self.active_tracks[new_id] = bbox
                        self.track_history[new_id] = deque(maxlen=self.sequence_length)
                        self.track_history[new_id].append(self.preprocess_detection(bbox))
                        self.missed_frames[new_id] = 0
                        self.update_position_history(new_id, frame_id, bbox[0], bbox[1], "detection")
                        print(f"➕ Created new ID {new_id}")
        
        # 見失った物体の処理
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
                
                # ===== 改善15: 見失いフレーム数を延長 =====
                if self.missed_frames[track_id] > 60:  # 45 → 60に延長（より長く保持）
                    missed_count = self.missed_frames[track_id]
                    self.release_id(track_id, frame_id)
                    print(f"❌ Lost fish ID {track_id} after {missed_count} frames")
                    
                    if self.use_lstm:
                        self.lstm_kalman_tracker.cleanup_track(track_id)
        
        # 評価機能
        if self.enable_evaluation and ground_truth is not None:
            predictions = {}
            for track_id, bbox in self.active_tracks.items():
                predictions[track_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            
            # ===== デバッグ: 評価更新の詳細ログ =====
            if frame_id % 100 == 0:
                print(f"📊 Frame {frame_id}: GT={len(ground_truth)}, Predictions={len(predictions)}")
                # 中間評価結果を表示
                intermediate_summary = self.evaluator.get_summary()
                print(f"   Intermediate - IDF1: {intermediate_summary.get('IDF1', 0):.4f}, "
                      f"ID_Switches: {intermediate_summary.get('ID_Switches', 0)}")
            
            self.evaluator.update_frame(ground_truth, predictions)
        elif self.enable_evaluation and ground_truth is None:
            if frame_id % 100 == 0:
                print(f"⚠️ Frame {frame_id}: No Ground Truth data available - SKIPPING EVALUATION UPDATE")
                print(f"   This will cause evaluation metrics to be unreliable!")

    def load_ground_truth(self, gt_path, frame_id):
        """MOTChallenge形式のGround Truthを読み込み"""
        gt_data = {}
        
        if not os.path.exists(gt_path):
            print(f"⚠️ Ground Truth file not found: {gt_path}")
            return None
        
        try:
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"⚠️ Ground Truth file is empty: {gt_path}")
                    return None
                
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    try:
                        frame_num = int(parts[0])
                        if frame_num == frame_id:
                            obj_id = int(parts[1])
                            x, y, w, h = map(float, parts[2:6])
                            gt_data[obj_id] = [x + w/2, y + h/2, w, h]
                    except ValueError as ve:
                        print(f"⚠️ Invalid data format in line: {line.strip()}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return gt_data if gt_data else None

    def process_video(self, video_path, ground_truth_path=None):
        """動画を処理"""
        print("\n" + "="*60)
        print("🎬 Video Processing & Tracking")
        print("="*60)
        print(f"📹 Video: {video_path}")
        print(f"🔧 LSTM Enabled: {self.use_lstm}")
        print(f"📊 Evaluation Enabled: {self.enable_evaluation}")
        
        if self.enable_evaluation:
            if ground_truth_path:
                if os.path.exists(ground_truth_path):
                    print(f"✅ Ground Truth: {ground_truth_path}")
                    try:
                        with open(ground_truth_path, 'r') as f:
                            lines = f.readlines()
                            print(f"   Total lines in GT file: {len(lines)}")
                            if lines:
                                first_frame = int(lines[0].strip().split(',')[0])
                                last_frame = int(lines[-1].strip().split(',')[0])
                                print(f"   Frame range: {first_frame} - {last_frame}")
                    except Exception as e:
                        print(f"⚠️ Error reading GT file: {e}")
                else:
                    print(f"❌ Ground Truth file not found: {ground_truth_path}")
                    print("   Evaluation will be disabled!")
                    self.enable_evaluation = False
            else:
                print("❌ No Ground Truth path provided")
                print("   Evaluation will be disabled!")
                self.enable_evaluation = False
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = "video/tracking_video_right.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
                
            results = self.yolo.track(frame, persist=True)
            
            detections = []
            if results[0].boxes is not None:
                detections.extend(results[0].boxes)
            
            ground_truth = None
            if self.enable_evaluation and ground_truth_path is not None:
                ground_truth = self.load_ground_truth(ground_truth_path, frame_count)
                
                # ===== デバッグ: Ground Truthの状態を確認 =====
                if frame_count % 500 == 0:  # 500フレームごとに表示
                    if ground_truth:
                        print(f"🔍 Frame {frame_count}: GT has {len(ground_truth)} objects")
                    else:
                        print(f"⚠️ Frame {frame_count}: GT is None or empty")
            
            if detections:
                self.update_tracking(frame_count, detections, ground_truth)
                
                used_ids = set()
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    min_distance = float('inf')
                    closest_id = None
                    
                    for track_id, track_info in self.active_tracks.items():
                        if track_id in used_ids:
                            continue
                        distance = np.linalg.norm(np.array([center_x, center_y]) - track_info[:2])
                        if distance < min_distance:
                            min_distance = distance
                            closest_id = track_id
                    
                    if closest_id is not None:
                        used_ids.add(closest_id)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{closest_id}", 
                                  (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            if self.enable_evaluation:
                summary = self.evaluator.get_summary()
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"ID Switches: {summary['ID_Switches']}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Active Tracks: {len(self.active_tracks)}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            out.write(frame)
            
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Tracking result saved to {output_path}")
        
        # ===== デバッグ: 最終評価前の状態確認 =====
        if self.enable_evaluation:
            print("\n" + "="*60)
            print("🔍 Pre-Final Evaluation Debug Info")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            
            # Evaluatorの内部状態を確認（可能な場合）
            if hasattr(self.evaluator, 'total_frames'):
                print(f"Evaluator total frames: {self.evaluator.total_frames}")
            if hasattr(self.evaluator, 'IDTP'):
                print(f"IDTP (ID True Positives): {self.evaluator.IDTP}")
            if hasattr(self.evaluator, 'IDFP'):
                print(f"IDFP (ID False Positives): {self.evaluator.IDFP}")
            if hasattr(self.evaluator, 'IDFN'):
                print(f"IDFN (ID False Negatives): {self.evaluator.IDFN}")
            
            print("="*60 + "\n")
        
        if self.enable_evaluation:
            print("\n" + "="*60)
            print("MOT Evaluation Results")
            print("="*60)
            summary = self.evaluator.print_summary()
            
            result_dir = os.path.dirname(output_path)
            if not result_dir:
                result_dir = "."
            json_path = os.path.join(result_dir, "evaluation_results.json")
            self.evaluator.save_results(json_path)
            
            return summary
        
        return None

    def collect_training_data(self, video_path, output_dir):
        """動画から教師データを自動生成"""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        track_data = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.yolo.track(frame, persist=True)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    obj_id = int(box.id.item()) if box.id is not None else None
                    
                    if obj_id is not None:
                        if obj_id not in track_data:
                            track_data[obj_id] = []
                        
                        track_data[obj_id].append([frame_count, center_x, center_y, width, height])
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        for obj_id, data in track_data.items():
            if len(data) >= self.sequence_length:
                data_array = np.array(data)
                output_path = os.path.join(output_dir, f"track_{obj_id}.npy")
                np.save(output_path, data_array)
                print(f"Saved tracking data for object {obj_id} with {len(data)} frames")
        
        cap.release()
        print(f"Data collection completed. Saved {len(track_data)} object tracks.")


if __name__ == "__main__":
    model_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/train_results/weights/best.pt"
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/test/9min_3D_left.mp4"
    ground_truth_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt"
    
    enable_evaluation = ground_truth_path is not None and os.path.exists(ground_truth_path)
    
    if not enable_evaluation and ground_truth_path:
        print(f"\n⚠️ WARNING: Ground Truth file not found: {ground_truth_path}")
        print("   Evaluation mode will be disabled.")
        print("   Please generate Ground Truth using ground_truth.py first.\n")
    
    tracker = ObjectTracker(
        model_path=model_path,
        sequence_length=10,
        max_fish=10,  # 最大10匹に制限
        use_lstm=True,
        enable_evaluation=enable_evaluation
    )
    
    print("Starting LSTM+Kalman Filter enhanced object tracking...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔢 Maximum tracking limit: 10 objects")
    print("Hybrid prediction features:")
    print("  - LSTM neural network for pattern learning")
    print("  - Kalman filter for motion prediction")
    print("  - Hybrid Re-ID scoring system")
    print("  - Enhanced lost fish tracking")
    
    if enable_evaluation:
        print("\n=== MOT Evaluation Mode Enabled ===")
        print("Evaluation metrics will be calculated:")
        print("  - MOTA (Multiple Object Tracking Accuracy)")
        print("  - MOTP (Multiple Object Tracking Precision)")
        print("  - IDF1 (ID F1 Score)")
        print("  - ID Switches")
        print("  - Fragmentations")
    
    evaluation_results = tracker.process_video(
        video_path,
        ground_truth_path=ground_truth_path
    )
    
    if evaluation_results:
        print("\n" + "="*60)
        print("Final Evaluation Summary")
        print("="*60)
        print(f"MOTA: {evaluation_results['MOTA']:.4f}")
        print(f"MOTP: {evaluation_results['MOTP']:.2f} pixels")
        print(f"IDF1: {evaluation_results['IDF1']:.4f}")
        print(f"ID Switches: {evaluation_results['ID_Switches']}")
        print(f"Fragmentations: {evaluation_results['Fragmentations']}")
        print("="*60)