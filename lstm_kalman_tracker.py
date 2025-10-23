import numpy as np
import torch
import torch.nn as nn
from collections import deque
import cv2

class KalmanFilter:
    """カルマンフィルタによる位置予測"""
    
    def __init__(self, dt=1.0):
        """
        カルマンフィルタの初期化
        dt: 時間間隔（フレーム間隔）
        """
        self.dt = dt
        
        # 状態ベクトル: [x, y, vx, vy] (位置と速度)
        self.state_size = 4
        self.measurement_size = 2  # 観測: [x, y]
        
        # 状態遷移行列 F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 観測行列 H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # プロセスノイズ共分散行列 Q
        q = 0.1  # プロセスノイズの強度
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=np.float32) * q
        
        # 観測ノイズ共分散行列 R
        self.R = np.eye(2, dtype=np.float32) * 0.1
        
        # 初期状態共分散行列 P
        self.P = np.eye(4, dtype=np.float32) * 100
        
        # 初期状態
        self.x = np.zeros(4, dtype=np.float32)
        self.is_initialized = False
    
    def initialize(self, x, y):
        """カルマンフィルタの初期化"""
        self.x = np.array([x, y, 0, 0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100
        self.is_initialized = True
    
    def predict(self):
        """予測ステップ"""
        if not self.is_initialized:
            return None
        
        # 状態予測
        self.x = self.F @ self.x
        
        # 共分散予測
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:2]  # 位置のみ返す
    
    def update(self, x, y):
        """更新ステップ"""
        if not self.is_initialized:
            self.initialize(x, y)
            return self.x[:2]
        
        # 観測ベクトル
        z = np.array([x, y], dtype=np.float32)
        
        # カルマンゲイン
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状態更新
        y = z - self.H @ self.x  # イノベーション
        self.x = self.x + K @ y
        
        # 共分散更新
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:2]  # 位置のみ返す
    
    def get_velocity(self):
        """現在の速度を取得"""
        if not self.is_initialized:
            return np.array([0, 0], dtype=np.float32)
        return self.x[2:4]
    
    def get_state(self):
        """現在の状態を取得"""
        if not self.is_initialized:
            return None
        return self.x.copy()

class LSTMKalmanPredictor(nn.Module):
    """LSTMとカルマンフィルタを組み合わせた予測モデル"""
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        super(LSTMKalmanPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 出力層（位置と速度の修正値）
        self.position_correction = nn.Linear(hidden_size, 2)  # 位置修正値
        self.velocity_correction = nn.Linear(hidden_size, 2)   # 速度修正値
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 最後の出力のみを使用
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 修正値を予測
        pos_correction = self.position_correction(last_output)
        vel_correction = self.velocity_correction(last_output)
        
        return pos_correction, vel_correction

class LSTMKalmanTracker:
    """LSTMとカルマンフィルタを組み合わせた追跡システム"""
    
    def __init__(self, sequence_length=10, device='cpu'):
        self.sequence_length = sequence_length
        self.device = device
        
        # LSTMモデル
        self.lstm_model = LSTMKalmanPredictor().to(device)
        
        # 各トラックのカルマンフィルタとLSTM状態
        self.track_filters = {}  # {track_id: KalmanFilter}
        self.track_lstm_states = {}  # {track_id: {'model': LSTMPredictor, 'sequence': deque}}
        
        # 予測結果の保存
        self.predictions = {}  # {track_id: {'kalman_pos': [], 'lstm_correction': [], 'final_pos': []}}
        
        # ハイブリッド予測の重み
        self.kalman_weight = 0.7  # カルマンフィルタの重み
        self.lstm_weight = 0.3    # LSTM修正の重み
        
    def normalize_sequence(self, sequence):
        """シーケンスを正規化（位置と速度を分離）"""
        if len(sequence) < 2:
            return None
            
        positions = []
        velocities = []
        
        for i in range(len(sequence)):
            frame_id, x, y, source, direction = sequence[i]
            positions.append([x, y])
            
            # 速度を計算
            if i > 0:
                prev_frame, prev_x, prev_y, _, _ = sequence[i-1]
                vx = x - prev_x
                vy = y - prev_y
                velocities.append([vx, vy])
            else:
                velocities.append([0, 0])
        
        return np.array(positions), np.array(velocities)
    
    def create_track_models(self, track_id):
        """新しいトラック用のカルマンフィルタとLSTMモデルを作成"""
        # カルマンフィルタを作成
        kalman_filter = KalmanFilter(dt=1.0)
        
        # LSTMモデルを作成
        lstm_model = LSTMKalmanPredictor().to(self.device)
        sequence = deque(maxlen=self.sequence_length)
        
        self.track_filters[track_id] = kalman_filter
        self.track_lstm_states[track_id] = {
            'model': lstm_model,
            'sequence': sequence,
            'last_position': None,
            'last_velocity': None
        }
        
        self.predictions[track_id] = {
            'kalman_pos': [],
            'lstm_correction': [],
            'final_pos': []
        }
    
    def update_track_sequence(self, track_id, position_history):
        """トラックのシーケンスを更新"""
        if track_id not in self.track_filters:
            self.create_track_models(track_id)
        
        # 位置履歴からシーケンスを作成
        if len(position_history) >= 2:
            positions, velocities = self.normalize_sequence(position_history)
            
            if positions is not None:
                # カルマンフィルタを更新
                kalman_filter = self.track_filters[track_id]
                latest_pos = kalman_filter.update(positions[-1, 0], positions[-1, 1])
                
                # LSTMシーケンスを更新
                track_state = self.track_lstm_states[track_id]
                
                # 位置と速度を結合してLSTM入力を作成
                for i in range(len(positions)):
                    combined_input = np.concatenate([positions[i], velocities[i]])
                    track_state['sequence'].append(combined_input)
                
                # 最新の位置と速度を保存
                track_state['last_position'] = positions[-1]
                track_state['last_velocity'] = velocities[-1]
    
    def predict_next_positions(self, track_id, steps=1):
        """次の位置を予測（カルマンフィルタ + LSTM修正）"""
        if track_id not in self.track_filters:
            return None, None, 0.0
        
        kalman_filter = self.track_filters[track_id]
        track_state = self.track_lstm_states[track_id]
        sequence = track_state['sequence']
        
        if len(sequence) < self.sequence_length:
            # LSTMが利用できない場合はカルマンフィルタのみ使用
            kalman_pos = kalman_filter.predict()
            if kalman_pos is not None:
                return kalman_pos, np.array([0, 0]), 0.5
            return None, None, 0.0
        
        # カルマンフィルタで予測
        kalman_pos = kalman_filter.predict()
        
        # LSTMで修正値を予測
        sequence_tensor = torch.FloatTensor(list(sequence)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lstm_pos_correction, lstm_vel_correction = track_state['model'](sequence_tensor)
        
        # CPUに移動してnumpy配列に変換
        lstm_correction = lstm_pos_correction.cpu().numpy()[0]
        
        # ハイブリッド予測（カルマンフィルタ + LSTM修正）
        if kalman_pos is not None:
            final_pos = (self.kalman_weight * kalman_pos + 
                        self.lstm_weight * (kalman_pos + lstm_correction))
            
            # 信頼度を計算
            confidence = min(len(sequence) / self.sequence_length, 1.0)
            
            return final_pos, kalman_pos, confidence
        
        return None, None, 0.0
    
    def compute_hybrid_reid_score(self, track_id, detection_position, detection_velocity=None):
        """ハイブリッドRe-Identificationスコアを計算"""
        if track_id not in self.track_filters:
            return 0.0
        
        # ハイブリッド予測位置を取得
        pred_position, kalman_pos, confidence = self.predict_next_positions(track_id)
        
        if pred_position is None:
            return 0.0
        
        # 位置スコア（距離に基づく）
        position_diff = np.linalg.norm(np.array(detection_position) - pred_position)
        position_score = max(0, 1 - position_diff / 100)  # 100ピクセル以内で正規化
        
        # カルマンフィルタの速度情報も考慮
        kalman_filter = self.track_filters[track_id]
        kalman_velocity = kalman_filter.get_velocity()
        
        velocity_score = 0.0
        if detection_velocity is not None and np.linalg.norm(kalman_velocity) > 0:
            velocity_diff = np.linalg.norm(np.array(detection_velocity) - kalman_velocity)
            velocity_score = max(0, 1 - velocity_diff / 50)  # 50ピクセル/フレーム以内で正規化
        
        # 総合スコア
        total_score = (0.6 * position_score + 0.4 * velocity_score) * confidence
        
        return total_score
    
    def predict_lost_track_positions(self, track_id, frames_ahead=5):
        """見失ったトラックの位置を予測（ハイブリッド）"""
        if track_id not in self.track_filters:
            return []
        
        kalman_filter = self.track_filters[track_id]
        predicted_positions = []
        
        # カルマンフィルタで予測
        for i in range(frames_ahead):
            kalman_pos = kalman_filter.predict()
            if kalman_pos is not None:
                predicted_positions.append(kalman_pos)
            else:
                break
        
        return predicted_positions
    
    def cleanup_track(self, track_id):
        """トラックのクリーンアップ"""
        if track_id in self.track_filters:
            del self.track_filters[track_id]
        if track_id in self.track_lstm_states:
            del self.track_lstm_states[track_id]
        if track_id in self.predictions:
            del self.predictions[track_id]
    
    def get_track_statistics(self, track_id):
        """トラックの統計情報を取得"""
        if track_id not in self.track_filters:
            return None
        
        kalman_filter = self.track_filters[track_id]
        track_state = self.track_lstm_states[track_id]
        sequence_length = len(track_state['sequence'])
        
        return {
            'sequence_length': sequence_length,
            'last_position': track_state['last_position'],
            'last_velocity': track_state['last_velocity'],
            'kalman_velocity': kalman_filter.get_velocity(),
            'is_ready_for_prediction': sequence_length >= self.sequence_length,
            'kalman_initialized': kalman_filter.is_initialized
        }
