import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import csv

class LSTMTracker(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(LSTMTracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # x, y, w, h の予測

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class ObjectTracker:
    def __init__(self, model_path, sequence_length=10):
        # YOLOモデルとLSTMモデルの読み込み
        self.yolo = YOLO(model_path)
        self.lstm = LSTMTracker()
        self.lstm.load_state_dict(torch.load(r"C:\Users\et439\OneDrive\桌面\project\Rin\best_lstm_model.pth", map_location='cpu'))
        self.lstm.eval()

        self.sequence_length = sequence_length
        self.track_history = {}
        self.available_ids = set(range(1, 6))
        self.active_tracks = {}
        self.missed_frames = {}

        # 🔸ログ保持リスト
        self.tracking_log = []

        self.cooling_tracks = {}  # 冷卻期中的軌跡
        self.cooldown_frames = 60  # 冷卻期長度

    def get_new_id(self):
        if self.available_ids:
            return self.available_ids.pop()
        return None

    def release_id(self, obj_id):
        if 1 <= obj_id <= 5:
            self.available_ids.add(obj_id)
            self.active_tracks.pop(obj_id, None)
            self.missed_frames.pop(obj_id, None)
            self.track_history.pop(obj_id, None)

    def preprocess_detection(self, detection):
        x, y, w, h = detection
        return np.array([x, y, w, h])

    def update_tracking(self, frame_id, detections):
        current_detections = set()
        predicted_positions = {}

        # --- 1. LSTMで予測 ---
        for track_id in list(self.active_tracks.keys()):
            if len(self.track_history[track_id]) >= self.sequence_length:
                seq = np.array(self.track_history[track_id])
                input_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_bbox = self.lstm(input_seq).cpu().numpy().flatten()
                predicted_positions[track_id] = pred_bbox
            else:
                predicted_positions[track_id] = self.track_history[track_id][-1]

        # --- 2. 検出結果と予測位置をマッチング ---
        unmatched_detections = list(detections)
        used_tracks = set()
        
        for i, det in enumerate(unmatched_detections[:]):
            bbox = det.xywh[0].cpu().numpy()

            min_dist = float('inf')
            matched_id = None

            for track_id, pred_bbox in predicted_positions.items():
                if track_id in used_tracks:
                    continue
                dist = np.linalg.norm(bbox[:2] - pred_bbox[:2])  # 中心点の距離
                if dist < min_dist and dist < 150:  # 閾値150（調整可能）
                    min_dist = dist
                    matched_id = track_id

            if matched_id is not None:
                current_detections.add(matched_id)
                used_tracks.add(matched_id)
                self.track_history[matched_id].append(self.preprocess_detection(bbox))
                self.active_tracks[matched_id] = bbox
                self.missed_frames[matched_id] = 0
                unmatched_detections.remove(det)  # マッチング済みは除外

        # --- 3. 残った検出に新しいIDを割り当て ---
        for det in unmatched_detections:
            bbox = det.xywh[0].cpu().numpy()
            new_id = self.get_new_id()
            if new_id is not None:
                current_detections.add(new_id)
                self.active_tracks[new_id] = bbox
                self.track_history[new_id] = deque(maxlen=self.sequence_length)
                self.track_history[new_id].append(self.preprocess_detection(bbox))
                self.missed_frames[new_id] = 0

        # --- 4. 長期間未検出な物体のIDを解放 ---
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_detections:
                self.missed_frames[track_id] = self.missed_frames.get(track_id, 0) + 1
###################### 修改部分 #########################
                if self.missed_frames[track_id] > 30:
                # 把這個 track_id 暫時進入冷卻期，不馬上釋放
                    self.cooling_tracks[track_id] = {
                        "last_bbox": self.active_tracks[track_id],
                        "cooling_count": 0
                    }
                    self.active_tracks.pop(track_id, None)
                    self.missed_frames.pop(track_id, None)
        # --- 5. 冷卻中的 track ID 處理 ---
        cooling_to_release = []

        for track_id, info in self.cooling_tracks.items():
            info["cooling_count"] += 1
            # 嘗試看看目前的 detection 有沒有一個靠近 last_bbox，可以復活
            for det in detections:
                bbox = det.xywh[0].cpu().numpy()
                dist = np.linalg.norm(bbox[:2] - info["last_bbox"][:2])
                if dist < 100:  # 嘗試復活此 ID
                    self.active_tracks[track_id] = bbox
                    self.track_history[track_id] = deque(maxlen=self.sequence_length)
                    self.track_history[track_id].append(self.preprocess_detection(bbox))
                    self.missed_frames[track_id] = 0
                    cooling_to_release.append(track_id)
                    break

            # 如果冷卻期超過上限，就真的刪掉
            if info["cooling_count"] > self.cooldown_frames:
                cooling_to_release.append(track_id)
                self.release_id(track_id)

        # 移除已復活或已刪除的冷卻 ID
        for tid in cooling_to_release:
            self.cooling_tracks.pop(tid, None)

###############################################################

    def save_tracking_log(self, filepath):
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'object_id', 'x', 'y', 'w', 'h'])
            writer.writerows(self.tracking_log)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\video\upt_tracking_result.mp4"
        csv_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\video\box\upt_lstm_tracking_log.csv"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

####### 把 bounding box 的邊框顏色做成不同顏色以便區分  ##########
        # define ID color（BGR form）
        id_colors = {
            1: (0, 255, 0),      # green
            2: (255, 0, 0),      # blue
            3: (0, 255, 255),    # yellow
            4: (255, 0, 255),    # purple
            5: (0, 128, 255),    # orange
        }
        default_color = (0, 0, 255)  # red, use for "when ID > 5"


###############################################################
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo.track(frame, persist=True)
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if results[0].boxes is not None:
                self.update_tracking(frame_id, results[0].boxes)

                used_ids = set()
                for box in results[0].boxes:
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
                        color = id_colors.get(closest_id, default_color)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID: {closest_id}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if results[0].boxes is None or len(results[0].boxes) == 0:
                for track_id, bbox in self.active_tracks.items():
                    x, y, w, h = bbox
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {track_id} (pred)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            out.write(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Tracking result saved to: {output_path}")

        # 🔸ログ保存
        self.save_tracking_log(csv_path)
        print(f"Tracking log saved to: {csv_path}")

if __name__ == "__main__":
    model_path =r"C:\Users\et439\OneDrive\桌面\project\Rin\train_results\weights\best.pt"

    tracker = ObjectTracker(model_path)
    video_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\video\processed_video.mp4"
    tracker.process_video(video_path)