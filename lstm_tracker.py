import torch
import torch.nn as nn
import numpy as np
from collections import deque
import math

class LSTMPredictor(nn.Module):
    """LSTMベースの位置予測モデル"""
    
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 出力層（位置と速度）
        self.position_head = nn.Linear(hidden_size, 2)  # x, y
        self.velocity_head = nn.Linear(hidden_size, 2)   # vx, vy
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 最後の出力のみを使用
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 位置と速度を予測
        position = self.position_head(last_output)
        velocity = self.velocity_head(last_output)
        
        return position, velocity

class LSTMEnhancedTracker:
    """LSTMを使った強化された物体追跡システム"""
    
    def __init__(self, sequence_length=10, prediction_horizon=5, device='cpu'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        # LSTMモデル
        self.lstm_model = LSTMPredictor().to(device)
        
        # 各トラックのLSTM状態
        self.track_lstm_states = {}  # {track_id: {'model': LSTMPredictor, 'sequence': deque}}
        
        # 予測結果の保存
        self.predictions = {}  # {track_id: {'positions': [], 'velocities': [], 'confidence': []}}
        
        # Re-IDスコアの設定
        self.position_weight = 0.6
        self.velocity_weight = 0.4
        self.confidence_threshold = 0.7
        
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
    
    def create_track_model(self, track_id):
        """新しいトラック用のLSTMモデルを作成"""
        model = LSTMPredictor().to(self.device)
        sequence = deque(maxlen=self.sequence_length)
        
        self.track_lstm_states[track_id] = {
            'model': model,
            'sequence': sequence,
            'last_position': None,
            'last_velocity': None
        }
        
        self.predictions[track_id] = {
            'positions': [],
            'velocities': [],
            'confidence': []
        }
    
    def update_track_sequence(self, track_id, position_history):
        """トラックのシーケンスを更新"""
        if track_id not in self.track_lstm_states:
            self.create_track_model(track_id)
        
        # 位置履歴からシーケンスを作成
        if len(position_history) >= 2:
            positions, velocities = self.normalize_sequence(position_history)
            
            if positions is not None:
                # シーケンスを更新
                track_state = self.track_lstm_states[track_id]
                
                # 位置と速度を結合してLSTM入力を作成
                for i in range(len(positions)):
                    combined_input = np.concatenate([positions[i], velocities[i]])
                    track_state['sequence'].append(combined_input)
                
                # 最新の位置と速度を保存
                track_state['last_position'] = positions[-1]
                track_state['last_velocity'] = velocities[-1]
    
    def predict_next_positions(self, track_id, steps=1):
        """次の位置を予測"""
        if track_id not in self.track_lstm_states:
            return None, None, 0.0
        
        track_state = self.track_lstm_states[track_id]
        sequence = track_state['sequence']
        
        if len(sequence) < self.sequence_length:
            return None, None, 0.0
        
        # シーケンスをテンソルに変換
        sequence_tensor = torch.FloatTensor(list(sequence)).unsqueeze(0).to(self.device)
        
        # 予測実行
        with torch.no_grad():
            predicted_position, predicted_velocity = track_state['model'](sequence_tensor)
        
        # CPUに移動してnumpy配列に変換
        pred_pos = predicted_position.cpu().numpy()[0]
        pred_vel = predicted_velocity.cpu().numpy()[0]
        
        # 信頼度を計算（シーケンスの長さと一貫性に基づく）
        confidence = min(len(sequence) / self.sequence_length, 1.0)
        
        return pred_pos, pred_vel, confidence
    
    def compute_reid_score(self, track_id, detection_position, detection_velocity=None):
        """Re-Identificationスコアを計算"""
        if track_id not in self.track_lstm_states:
            return 0.0
        
        # 予測位置を取得
        pred_position, pred_velocity, confidence = self.predict_next_positions(track_id)
        
        if pred_position is None:
            return 0.0
        
        # 位置スコア（距離に基づく）
        position_diff = np.linalg.norm(np.array(detection_position) - pred_position)
        position_score = max(0, 1 - position_diff / 100)  # 100ピクセル以内で正規化
        
        # 速度スコア
        velocity_score = 0.0
        if detection_velocity is not None and pred_velocity is not None:
            velocity_diff = np.linalg.norm(np.array(detection_velocity) - pred_velocity)
            velocity_score = max(0, 1 - velocity_diff / 50)  # 50ピクセル/フレーム以内で正規化
        
        # 総合スコア
        total_score = (self.position_weight * position_score + 
                      self.velocity_weight * velocity_score) * confidence
        
        return total_score
    
    def predict_lost_track_positions(self, track_id, frames_ahead=5):
        """見失ったトラックの位置を予測"""
        if track_id not in self.track_lstm_states:
            return []
        
        track_state = self.track_lstm_states[track_id]
        predicted_positions = []
        
        # 現在の位置と速度を取得
        current_pos = track_state['last_position']
        current_vel = track_state['last_velocity']
        
        if current_pos is None:
            return []
        
        # 単純な線形予測（LSTMが利用できない場合のフォールバック）
        for i in range(frames_ahead):
            predicted_pos = current_pos + current_vel * (i + 1)
            predicted_positions.append(predicted_pos)
        
        return predicted_positions
    
    def cleanup_track(self, track_id):
        """トラックのクリーンアップ"""
        if track_id in self.track_lstm_states:
            del self.track_lstm_states[track_id]
        if track_id in self.predictions:
            del self.predictions[track_id]
    
    def get_track_statistics(self, track_id):
        """トラックの統計情報を取得"""
        if track_id not in self.track_lstm_states:
            return None
        
        track_state = self.track_lstm_states[track_id]
        sequence_length = len(track_state['sequence'])
        
        return {
            'sequence_length': sequence_length,
            'last_position': track_state['last_position'],
            'last_velocity': track_state['last_velocity'],
            'is_ready_for_prediction': sequence_length >= self.sequence_length
        }
