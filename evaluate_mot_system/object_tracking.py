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
    def __init__(self, model_path, sequence_length=10, max_fish=20, use_lstm=True, enable_evaluation=False):
        # YOLOモデルの読み込み（model_pathがNoneの場合はスキップ）
        if model_path is not None:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = None
        
        self.sequence_length = sequence_length
        self.max_fish = max_fish
        self.use_lstm = use_lstm
        
        # ===== 追加: 評価機能 =====
        self.enable_evaluation = enable_evaluation
        if self.enable_evaluation:
            self.evaluator = MOTEvaluator(iou_threshold=0.5, distance_threshold=50)
            print("MOT Evaluator initialized")
        # ==========================
        
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
        
        # IDごとの位置履歴を保存（x, y座標、動きの方向）
        self.position_history = {}  # {track_id: [(frame_id, x, y, source, direction_rad), ...]}
        
        # 改善されたID管理システム
        self.next_id = 1  # 次の新しいID
        self.active_tracks = {}  # 現在追跡中の物体とそのID
        self.missed_frames = {}  # 物体を見失ったフレーム数を記録
        
        # 見失った魚の情報を記憶するシステム
        self.lost_fish = {}  # {id: {'last_position': [x, y, w, h], 'lost_frames': count, 'last_seen_frame': frame_id}}
        self.reuse_distance_threshold = 250  # ID再利用の距離閾値
        self.max_lost_frames = 60  # IDを保持する最大フレーム数（60から30に短縮）
        
        self.reuse_score_threshold = 0.15  # スコア閾値（緩い）
        self.lstm_prediction_range = 100  # LSTM予測範囲
        self.distance_score_weight = 0.6  # 距離の重み
        self.lstm_score_weight = 0.4  # LSTMの重み
        
        # 破棄されたIDを記録するシステム（位置情報も含む）
        self.discarded_ids = {}  # {id: {'position': [x, y], 'discarded_frame': frame_id}}
        self.max_discarded_ids = 20  # 記録する最大破棄ID数
        self.discarded_id_reuse_distance = 100  # 破棄ID再利用の距離閾値
        
        
        # 動きの方向ベースマッチングの設定
        self.use_direction_matching = True  # 動きの方向を考慮したマッチングを使用するか
        self.direction_weight = 0.2  # 方向の重み（0.0-1.0）
        self.distance_weight = 0.8  # 距離の重み（0.0-1.0）
        self.direction_threshold = 1.0  # 方向差の閾値（ラジアン）
        
        # LSTM強化マッチングの設定
        self.lstm_matching_threshold = 0.6  # LSTMマッチングの閾値
        self.prediction_weight = 0.5  # 予測の重み
        
    
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
        # 評価用にサイズ制限を無効化
        # if len(self.position_history[track_id]) > 1000:  # 最大1000フレーム分保持
        #     self.position_history[track_id] = self.position_history[track_id][-500:]  # 最新500フレーム分のみ保持
    
        
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
        """距離と方向を考慮したマッチングコストを計算（LSTM予測も含む）"""
        if track_id not in self.track_history or len(self.track_history[track_id]) == 0:
            return float('inf')
        
        # 基本的な距離コスト
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
        
        # LSTM+カルマンフィルタ予測コスト（LSTMが有効な場合）
        lstm_cost = 0.0
        lstm_confidence = 0.0
        if self.use_lstm and track_id in self.position_history:
            # LSTM+カルマンフィルタトラッカーのシーケンスを更新
            self.lstm_kalman_tracker.update_track_sequence(track_id, self.position_history[track_id])
            
            # ハイブリッドRe-IDスコアを計算
            detection_position = detection[:2]
            hybrid_score = self.lstm_kalman_tracker.compute_hybrid_reid_score(track_id, detection_position)
            lstm_confidence = hybrid_score
            
            # スコアが低い場合はペナルティ（係数を下げて緩和）
            if hybrid_score < self.lstm_matching_threshold:
                lstm_cost = (self.lstm_matching_threshold - hybrid_score) * 20  # 50から20に減少
        
        # 総合コスト（距離、方向、LSTM予測の重み付き和）
        total_cost = (self.distance_weight * distance + 
                     self.direction_weight * direction_cost + 
                     self.prediction_weight * lstm_cost)
        
        return total_cost, distance, direction_cost, lstm_cost, lstm_confidence
        
    def release_id(self, obj_id, current_frame):
        """IDを解放し、見失った魚として記録（30フレーム以上見失った場合）"""
        if obj_id in self.active_tracks:
            # 見失った魚の情報を保存
            self.lost_fish[obj_id] = {
                'last_position': self.active_tracks[obj_id].copy(),
                'lost_frames': 0,
                'last_seen_frame': current_frame,
                'predicted_positions': []  # LSTM予測位置を保存
            }
            
            # LSTM+カルマンフィルタ予測位置を計算（見失った魚の追跡継続）
            if self.use_lstm and obj_id in self.position_history:
                predicted_positions = self.lstm_kalman_tracker.predict_lost_track_positions(obj_id, frames_ahead=10)
                self.lost_fish[obj_id]['predicted_positions'] = predicted_positions
                print(f"Generated {len(predicted_positions)} hybrid predicted positions for lost fish {obj_id}")
            
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
        """見失った魚の中で再利用可能なIDを探す（緩和版）"""
        reusable_id = None
        min_distance = float('inf')
        best_total_score = 0.0
        
        for fish_id, fish_info in list(self.lost_fish.items()):
            # 保持期間チェック（60フレームに延長）
            if fish_info['lost_frames'] > self.max_lost_frames:
                del self.lost_fish[fish_id]
                self.add_discarded_id(fish_id, fish_info['last_position'], fish_info['last_seen_frame'])
                continue
            
            # 距離計算
            last_pos = fish_info['last_position']
            distance = np.linalg.norm(np.array(new_position[:2]) - np.array(last_pos[:2]))
            
            # 距離スコア（緩和版：閾値を超えても除外しない）
            if distance < self.reuse_distance_threshold:
                distance_score = 1 - (distance / self.reuse_distance_threshold)
            else:
                # 閾値の2倍までは少しスコアを与える
                distance_score = max(0, 0.3 * (1 - distance / (self.reuse_distance_threshold * 2)))
            
            # LSTM予測スコア
            lstm_score = 0.0
            if self.use_lstm and 'predicted_positions' in fish_info and fish_info['predicted_positions']:
                min_pred_distance = float('inf')
                for pred_pos in fish_info['predicted_positions']:
                    pred_distance = np.linalg.norm(np.array(new_position[:2]) - pred_pos)
                    min_pred_distance = min(min_pred_distance, pred_distance)
                
                # LSTM予測範囲を拡大（50 → 100）
                if min_pred_distance < self.lstm_prediction_range:
                    lstm_score = max(0, 1 - min_pred_distance / self.lstm_prediction_range)
            
            # 総合スコア（重み調整）
            total_score = (self.distance_score_weight * distance_score + 
                        self.lstm_score_weight * lstm_score)
            
            # 閾値判定（0.3 → 0.15に緩和）
            if total_score > self.reuse_score_threshold:
                if reusable_id is None or total_score > best_total_score:
                    min_distance = distance
                    reusable_id = fish_id
                    best_total_score = total_score
        
        if reusable_id is not None:
            print(f"Found reusable ID {reusable_id} with distance {min_distance:.2f} and score {best_total_score:.3f}")
        
        return reusable_id
    
    def preprocess_detection(self, detection):
        # YOLOの検出結果をLSTMの入力形式に変換
        x, y, w, h = detection
        return np.array([x, y, w, h])
    
    def update_tracking(self, frame_id, detections, ground_truth=None):
        """
        追跡を更新（既存のロジックは変更なし、評価機能のみ追加）
        
        Args:
            frame_id: フレームID
            detections: YOLO検出結果
            ground_truth: Ground Truthデータ（評価用、オプション）
        """
        # 見失った魚のフレーム数を更新（30フレーム以上見失った場合）
        for fish_id in self.lost_fish:
            self.lost_fish[fish_id]['lost_frames'] += 1
        
        # 現在のフレームで検出された物体のIDを記録
        current_detections = set()
        
        for det in detections:
            # 物体の位置情報を取得
            bbox = det.xywh[0].cpu().numpy()  # x, y, w, h
            
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
                        print(f"Enhanced matching: Track {track_id}, cost={total_cost:.2f}, distance={distance:.2f}, direction_cost={direction_cost:.3f}, LSTM_cost={lstm_cost:.3f}, LSTM_confidence={lstm_confidence:.3f}")
                else:
                    # 従来の距離ベースマッチング
                    if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                        old_bbox = self.track_history[track_id][-1]
                        distance = np.linalg.norm(bbox[:2] - old_bbox[:2])
                
                        if distance < 200 and distance < min_cost:
                            min_cost = distance
                            best_match_id = track_id
            
            # 最適なマッチを見つけた場合（距離と方向を考慮、コストが閾値以内の場合のみ）
            if best_match_id is not None and min_cost < 150:  # コスト閾値を追加
                current_detections.add(best_match_id)
                matched = True
                self.missed_frames[best_match_id] = 0
                self.track_history[best_match_id].append(self.preprocess_detection(bbox))
                self.active_tracks[best_match_id] = bbox
                # 位置履歴に追加（動きの方向も計算される）
                self.update_position_history(best_match_id, frame_id, bbox[0], bbox[1], "detection")
                print(f"Matched detection to track {best_match_id} with enhanced matching (cost: {min_cost:.2f})")
            elif best_match_id is not None:
                print(f"Match found but cost too high: {min_cost:.2f}, skipping match")
            
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
                        print(f"Created new ID {new_id} for fish at position {bbox[:2]}")
        
        # 見失った物体の処理（30フレーム以上見失った場合） 
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
                
                
                if self.missed_frames[track_id] > 30:  # 30フレーム以上見失った場合
                    missed_count = self.missed_frames[track_id]
                    self.release_id(track_id, frame_id)
                    print(f"Lost fish ID {track_id} after {missed_count} frames")
                    
                    # LSTM+カルマンフィルタトラッカーのクリーンアップ
                    if self.use_lstm:
                        self.lstm_kalman_tracker.cleanup_track(track_id)
        
        # ===== 追加: 評価機能 =====
        # Ground Truthがある場合、評価器を更新
        if self.enable_evaluation and ground_truth is not None:
            # 現在のトラッキング結果を評価フォーマットに変換
            predictions = {}
            for track_id, bbox in self.active_tracks.items():
                # bbox形式: [x_center, y_center, width, height]
                predictions[track_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            
            # デバッグ情報
            if frame_id % 100 == 0:  # 100フレームごとに表示
                print(f"📊 Frame {frame_id}: GT objects={len(ground_truth)}, Predictions={len(predictions)}")
            
            # 評価器を更新
            self.evaluator.update_frame(ground_truth, predictions)
        elif self.enable_evaluation and ground_truth is None:
            if frame_id % 100 == 0:
                print(f"⚠️ Frame {frame_id}: No Ground Truth data available")
        # ==========================

    # ===== 追加: Ground Truth読み込み関数 =====
    def load_ground_truth(self, gt_path, frame_id):
        """
        MOTChallenge形式のGround Truthを読み込み
        形式: frame, id, x, y, w, h, conf, -1, -1, -1
        
        Args:
            gt_path: Ground Truthファイルのパス
            frame_id: 現在のフレームID
            
        Returns:
            dict: {obj_id: [x_center, y_center, width, height]}
        """
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
                            # 左上角座標から中心座標に変換
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
    # ==========================================

    def process_video(self, video_path, ground_truth_path=None):
        """
        動画を処理（既存のロジックは変更なし、評価機能のみ追加）
        
        Args:
            video_path: 入力動画のパス
            ground_truth_path: Ground Truthファイルのパス（オプション）
        """
        # ===== 評価設定の確認 =====
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
                    # Ground Truthファイルの内容をチェック
                    try:
                        with open(ground_truth_path, 'r') as f:
                            lines = f.readlines()
                            print(f"   Total lines in GT file: {len(lines)}")
                            if lines:
                                # 最初と最後のフレームIDを表示
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
        # ==========================
        
        cap = cv2.VideoCapture(video_path)
        
        # 動画の設定を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 出力用のVideoWriterを設定
        output_path = "video/tracking_video_right.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
                
            # YOLOで物体検出
            results = self.yolo.track(frame, persist=True)
            
            # YOLO検出結果を処理
            detections = []
            if results[0].boxes is not None:
                detections.extend(results[0].boxes)
            
            # ===== 追加: Ground Truth読み込み =====
            ground_truth = None
            if self.enable_evaluation and ground_truth_path is not None:
                ground_truth = self.load_ground_truth(ground_truth_path, frame_count)
            # ======================================
            
            if detections:
                # 追跡の更新（Ground Truthも渡す）
                self.update_tracking(frame_count, detections, ground_truth)
                
                # 結果の可視化
                used_ids = set()  # このフレームで使用済みのIDを記録
                for box in detections:
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
                        # BBoxを緑色で表示
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # IDを青色で表示
                        cv2.putText(frame, f"{closest_id}", 
                                  (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # ===== 追加: 評価情報を画面に表示 =====
            if self.enable_evaluation:
                summary = self.evaluator.get_summary()
                # フレーム情報
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # ID切り替わり回数
                cv2.putText(frame, f"ID Switches: {summary['ID_Switches']}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                # アクティブなトラック数
                cv2.putText(frame, f"Active Tracks: {len(self.active_tracks)}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # =========================================
            
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
        
        # ===== 追加: 評価結果の表示と保存 =====
        if self.enable_evaluation:
            print("\n" + "="*60)
            print("MOT Evaluation Results")
            print("="*60)
            summary = self.evaluator.print_summary()
            
            # JSONファイルとして保存
            result_dir = os.path.dirname(output_path)
            if not result_dir:
                result_dir = "."
            json_path = os.path.join(result_dir, "evaluation_results.json")
            self.evaluator.save_results(json_path)
            
            return summary
        # ======================================
        
        return None

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
    model_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/train_results/weights/best.pt"
    
    # 動画のパスを指定
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3min_3D_left.mp4"
    
    # Ground Truthのパス（MOTChallenge形式の.txtファイル）
    # 例: "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt"
    # ground_truth.pyはGround Truth生成ツールで、Ground Truthデータファイルではありません
    ground_truth_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt"
    
    # ===== 評価モードの設定 =====
    # Ground Truthファイルが存在する場合のみ評価モードを有効化
    enable_evaluation = ground_truth_path is not None and os.path.exists(ground_truth_path)
    
    if not enable_evaluation and ground_truth_path:
        print(f"\n⚠️ WARNING: Ground Truth file not found: {ground_truth_path}")
        print("   Evaluation mode will be disabled.")
        print("   Please generate Ground Truth using ground_truth.py first.\n")
    # ===========================
    
    # LSTM強化トラッカーの初期化
    tracker = ObjectTracker(
        model_path=model_path,
        sequence_length=10,
        max_fish=20,
        use_lstm=True,  # LSTM機能を有効化
        enable_evaluation=enable_evaluation  # 評価機能を有効化
    )
    
    print("Starting LSTM+Kalman Filter enhanced object tracking...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
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
    
    # 動画の処理開始
    evaluation_results = tracker.process_video(
        video_path,
        ground_truth_path=ground_truth_path
    )
    
    # ===== 評価結果の表示 =====
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
    # ==========================