import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
import os
# from sklearn.metrics.pairwise import cosine_similarity  # 使用しないためコメントアウト

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

class AppearanceFeatureExtractor(nn.Module):
    """外観特徴抽出器（簡易版CNN）"""
    def __init__(self, input_channels=3, feature_dim=256):
        super(AppearanceFeatureExtractor, self).__init__()
        
        # 簡易CNN（実際の実装ではより複雑なCNNを使用）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, height, width] - 画像パッチ
        Returns:
            features: [batch_size, feature_dim] - 外観特徴量
        """
        conv_out = self.conv_layers(x)
        features = self.fc(conv_out.view(conv_out.size(0), -1))
        return features

class LSTMReIDTracker:
    """LSTMベースのRe-IDトラッキングシステム"""
    
    def __init__(self, model_path, lstm_model_path=None, sequence_length=10):
        # YOLOモデルの読み込み
        if model_path is not None:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = None
            
        self.sequence_length = sequence_length
        
        # LSTMモデルの初期化
        self.lstm_model = LSTMMotionPredictor()
        self.lstm_available = False
        
        if lstm_model_path and os.path.exists(lstm_model_path):
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_model_path))
                self.lstm_model.eval()
                self.lstm_available = True
                print(f"LSTMモデルを読み込みました: {lstm_model_path}")
            except Exception as e:
                print(f"LSTMモデルの読み込みに失敗: {e}")
                self.lstm_available = False
        
        # 外観特徴抽出器の初期化
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.appearance_extractor.eval()
        
        # トラッキング関連のデータ構造
        self.next_id = 1
        self.active_tracks = {}  # {track_id: {'bbox': [x,y,w,h], 'last_seen': frame_id, 'missed_frames': count}}
        self.track_history = {}  # {track_id: deque([x,y,w,h], ...)}
        self.motion_features = {}  # {track_id: [feature_vector]}
        self.appearance_features = {}  # {track_id: [feature_vector]}
        
        # 失われたトラックの管理
        self.lost_tracks = {}  # {track_id: {'last_bbox': [x,y,w,h], 'lost_frames': count, 'last_features': [motion_feat, appearance_feat]}}
        self.max_lost_frames = 30  # 最大失跡フレーム数
        
        # Re-ID設定
        self.motion_similarity_threshold = 0.7  # 移動特徴の類似度閾値
        self.appearance_similarity_threshold = 0.6  # 外観特徴の類似度閾値
        self.distance_threshold = 150  # 距離閾値（ピクセル）
        self.motion_weight = 0.4  # 移動特徴の重み
        self.appearance_weight = 0.6  # 外観特徴の重み
        
    def extract_appearance_features(self, frame, bbox):
        """画像から外観特徴を抽出"""
        try:
            x, y, w, h = bbox
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # 画像パッチを抽出
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                return None
                
            # リサイズしてテンソルに変換
            patch_resized = cv2.resize(patch, (64, 64))
            patch_tensor = torch.FloatTensor(patch_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                features = self.appearance_extractor(patch_tensor)
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"外観特徴抽出エラー: {e}")
            return None
    
    def predict_motion_and_features(self, track_id):
        """LSTMで移動予測と特徴抽出"""
        if not self.lstm_available or track_id not in self.track_history:
            return None
            
        if len(self.track_history[track_id]) < self.sequence_length:
            return None
        
        # 履歴からシーケンスを取得
        history = list(self.track_history[track_id])
        sequence = np.array(history[-self.sequence_length:])
        
        # テンソルに変換
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            output = self.lstm_model(sequence_tensor)
            
            return {
                'motion_pred': output['motion_pred'].cpu().numpy()[0],
                'features': output['features'].cpu().numpy()[0],
                'confidence': output['confidence'].cpu().numpy()[0][0]
            }
    
    def compute_similarity(self, features1, features2):
        """特徴量の類似度を計算（コサイン類似度）"""
        if features1 is None or features2 is None:
            return 0.0
            
        # 正規化
        features1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
        features2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # コサイン類似度
        similarity = np.dot(features1_norm, features2_norm)
        return max(0.0, similarity)
    
    def find_best_reid_match(self, detection_bbox, detection_appearance_feat, frame):
        """Re-IDによる最適なマッチング"""
        best_match_id = None
        best_score = 0.0
        
        # 1. アクティブトラックとのマッチング
        for track_id in self.active_tracks.keys():
            if track_id in self.motion_features:
                # 距離チェック
                track_bbox = self.active_tracks[track_id]['bbox']
                distance = np.linalg.norm(np.array(detection_bbox[:2]) - np.array(track_bbox[:2]))
                
                if distance > self.distance_threshold:
                    continue
                
                # LSTM予測を取得
                lstm_output = self.predict_motion_and_features(track_id)
                if lstm_output is None:
                    continue
                
                # 移動特徴の類似度
                motion_sim = self.compute_similarity(
                    detection_appearance_feat,  # 簡易的に外観特徴を使用
                    lstm_output['features']
                )
                
                # 外観特徴の類似度
                if track_id in self.appearance_features:
                    appearance_sim = self.compute_similarity(
                        detection_appearance_feat,
                        self.appearance_features[track_id]
                    )
                else:
                    appearance_sim = 0.0
                
                # 総合スコア
                total_score = (self.motion_weight * motion_sim + 
                             self.appearance_weight * appearance_sim)
                
                if total_score > best_score and total_score > 0.5:  # 閾値
                    best_score = total_score
                    best_match_id = track_id
        
        # 2. 失われたトラックとのマッチング
        for track_id, lost_info in self.lost_tracks.items():
            # 距離チェック
            last_bbox = lost_info['last_bbox']
            distance = np.linalg.norm(np.array(detection_bbox[:2]) - np.array(last_bbox[:2]))
            
            if distance > self.distance_threshold:
                continue
            
            # 特徴量の類似度
            if 'last_features' in lost_info:
                last_motion_feat, last_appearance_feat = lost_info['last_features']
                
                # 移動特徴の類似度
                motion_sim = self.compute_similarity(
                    detection_appearance_feat,  # 簡易的に外観特徴を使用
                    last_motion_feat
                )
                
                # 外観特徴の類似度
                appearance_sim = self.compute_similarity(
                    detection_appearance_feat,
                    last_appearance_feat
                )
                
                # 総合スコア
                total_score = (self.motion_weight * motion_sim + 
                             self.appearance_weight * appearance_sim)
                
                if total_score > best_score and total_score > 0.6:  # 失われたトラックはより厳しい閾値
                    best_score = total_score
                    best_match_id = track_id
        
        return best_match_id, best_score
    
    def update_tracking(self, frame_id, detections, frame):
        """Re-ID統合トラッキングの更新"""
        current_detections = set()
        
        for detection in detections:
            # 検出結果を処理
            bbox = detection.xywh[0].cpu().numpy()  # [x, y, w, h]
            
            # 外観特徴を抽出
            appearance_feat = self.extract_appearance_features(frame, bbox)
            if appearance_feat is None:
                continue
            
            # Re-IDによるマッチング
            best_match_id, match_score = self.find_best_reid_match(bbox, appearance_feat, frame)
            
            if best_match_id is not None:
                # 既存トラックの更新
                current_detections.add(best_match_id)
                self.active_tracks[best_match_id]['bbox'] = bbox
                self.active_tracks[best_match_id]['last_seen'] = frame_id
                self.active_tracks[best_match_id]['missed_frames'] = 0
                
                # 履歴を更新
                self.track_history[best_match_id].append(bbox)
                
                # 特徴量を更新
                self.appearance_features[best_match_id] = appearance_feat
                
                # LSTM特徴量を更新
                lstm_output = self.predict_motion_and_features(best_match_id)
                if lstm_output is not None:
                    self.motion_features[best_match_id] = lstm_output['features']
                
                print(f"Re-ID成功: Track {best_match_id}, スコア: {match_score:.3f}")
                
                # 失われたトラックから削除
                if best_match_id in self.lost_tracks:
                    del self.lost_tracks[best_match_id]
                    
            else:
                # 新しいトラックの作成
                new_id = self.next_id
                self.next_id += 1
                
                current_detections.add(new_id)
                self.active_tracks[new_id] = {
                    'bbox': bbox,
                    'last_seen': frame_id,
                    'missed_frames': 0
                }
                
                self.track_history[new_id] = deque(maxlen=self.sequence_length)
                self.track_history[new_id].append(bbox)
                
                self.appearance_features[new_id] = appearance_feat
                
                print(f"新しいトラック作成: ID {new_id}")
        
        # 失われたトラックの処理
        tracks_to_remove = []
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.active_tracks[track_id]['missed_frames'] += 1
                
                if self.active_tracks[track_id]['missed_frames'] > self.max_lost_frames:
                    # トラックを失われたリストに移動
                    self.lost_tracks[track_id] = {
                        'last_bbox': self.active_tracks[track_id]['bbox'],
                        'lost_frames': 0,
                        'last_features': (
                            self.motion_features.get(track_id),
                            self.appearance_features.get(track_id)
                        )
                    }
                    
                    tracks_to_remove.append(track_id)
                    print(f"トラック {track_id} を失われたリストに移動")
        
        # 削除対象のトラックを削除
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            if track_id in self.motion_features:
                del self.motion_features[track_id]
            if track_id in self.appearance_features:
                del self.appearance_features[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
        
        # 失われたトラックのフレーム数を更新
        for track_id in self.lost_tracks:
            self.lost_tracks[track_id]['lost_frames'] += 1
            
            # 長期間失われたトラックは削除
            if self.lost_tracks[track_id]['lost_frames'] > self.max_lost_frames * 2:
                del self.lost_tracks[track_id]
    
    def process_video(self, video_path):
        """動画処理メイン関数"""
        cap = cv2.VideoCapture(video_path)
        
        # 動画設定を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 出力設定
        output_path = "video/lstm_reid_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLOで物体検出
            results = self.yolo.track(frame, persist=True)
            
            if results[0].boxes is not None:
                # トラッキング更新
                self.update_tracking(frame_count, results[0].boxes, frame)
                
                # 現在のフレームで検出されたトラックのみを可視化
                for detection in results[0].boxes:
                    bbox = detection.xywh[0].cpu().numpy()
                    x, y, w, h = bbox
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    # 最も近いアクティブトラックのIDを探す
                    min_distance = float('inf')
                    closest_id = None
                    
                    for track_id, track_info in self.active_tracks.items():
                        track_bbox = track_info['bbox']
                        distance = np.linalg.norm(np.array([x, y]) - np.array(track_bbox[:2]))
                        if distance < min_distance and distance < 50:  # 50ピクセル以内
                            min_distance = distance
                            closest_id = track_id
                    
                    if closest_id is not None:
                        # バウンディングボックスを描画
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {closest_id}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 失われたトラックの予測位置を表示（最大5フレームまで）
                if self.lstm_available:
                    for track_id, lost_info in self.lost_tracks.items():
                        if (lost_info['lost_frames'] <= 5 and  # 最大5フレームまで表示
                            track_id in self.track_history and 
                            len(self.track_history[track_id]) >= self.sequence_length):
                            lstm_output = self.predict_motion_and_features(track_id)
                            if lstm_output is not None and lstm_output['confidence'] > 0.7:  # より高い信頼度
                                pred_bbox = lstm_output['motion_pred']
                                pred_x, pred_y = pred_bbox[:2]
                                
                                # 画面内かチェック
                                if 0 <= pred_x < width and 0 <= pred_y < height:
                                    # 予測位置を赤色で表示
                                    cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (0, 0, 255), 2)
                                    cv2.putText(frame, f"Pred-{track_id}", 
                                              (int(pred_x)+10, int(pred_y)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # フレームを出力
            out.write(frame)
            
            # 表示
            cv2.imshow("LSTM Re-ID Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # 進捗表示とデバッグ情報
            if frame_count % 100 == 0:
                print(f"処理済みフレーム: {frame_count}")
                print(f"  アクティブトラック数: {len(self.active_tracks)}")
                print(f"  失われたトラック数: {len(self.lost_tracks)}")
                if self.active_tracks:
                    active_ids = list(self.active_tracks.keys())
                    print(f"  アクティブID: {active_ids}")
                if self.lost_tracks:
                    lost_ids = list(self.lost_tracks.keys())
                    print(f"  失われたID: {lost_ids}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"LSTM Re-IDトラッキング結果を保存しました: {output_path}")
        
        # 統計情報を表示
        print(f"\n=== トラッキング統計 ===")
        print(f"総フレーム数: {frame_count}")
        print(f"最終アクティブトラック数: {len(self.active_tracks)}")
        print(f"失われたトラック数: {len(self.lost_tracks)}")
        print(f"LSTM利用可能: {self.lstm_available}")

if __name__ == "__main__":
    # モデルパス
    model_path = "./train_results/weights/best.pt"
    lstm_model_path = "best_lstm_model.pth"
    
    # トラッカー初期化
    tracker = LSTMReIDTracker(model_path, lstm_model_path)
    
    # 動画パス
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3D_left.mp4"
    
    print("=== LSTM Re-IDトラッキングシステム ===")
    print(f"移動特徴重み: {tracker.motion_weight}")
    print(f"外観特徴重み: {tracker.appearance_weight}")
    print(f"距離閾値: {tracker.distance_threshold}")
    print(f"LSTM利用可能: {tracker.lstm_available}")
    
    # 動画処理開始
    tracker.process_video(video_path)
