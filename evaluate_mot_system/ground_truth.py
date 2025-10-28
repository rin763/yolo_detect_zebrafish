"""
Ground Truth生成ツール - 3つの方法を提供
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from object_tracking import ObjectTracker

class GroundTruthGenerator:
    """
    方法1: 半自動生成（推奨）
    - 現在のトラッカーで初期予測を生成
    - 目視確認しながらID修正
    - 最も効率的
    
    方法2: 完全自動生成（簡易評価用）
    - 現在のトラッカー結果をそのまま保存
    - 別の設定で動かした時との比較用
    
    方法3: サンプリング生成（長時間動画用）
    - N秒ごとにフレームを抽出
    - 代表的なフレームのみ手動アノテーション
    """
    
    def __init__(self, model_path, use_lstm=True, enable_evaluation=False):
        """ObjectTrackerを使用してGround Truthを生成"""
        self.tracker = ObjectTracker(
            model_path=model_path,
            sequence_length=10,
            max_fish=20,
            use_lstm=use_lstm,
            enable_evaluation=enable_evaluation
        )
        self.ground_truth_data = []
        self.id_corrections = {}  # ユーザーによるID修正記録
        self.next_id = 1
        
        # UIの状態
        self.current_frame_idx = 0
        self.paused = True
        self.selected_box = None
        
        # マウスクリック時の位置
        self.mouse_x = None
        self.mouse_y = None
        self.mouse_clicked = False
        
    def method1_semi_automatic(self, video_path, output_path, review_interval=30):
        """
        方法1: 半自動生成
        
        使い方:
        1. トラッカーが自動でIDを割り当て
        2. N フレームごとに一時停止
        3. ID切り替わりがあれば修正
        4. 修正内容を保存
        
        Args:
            video_path: 入力動画
            output_path: Ground Truth保存先（.txt）
            review_interval: 何フレームごとにレビューするか
        """
        print("\n=== 方法1: 半自動Ground Truth生成（ObjectTracker使用）===")
        print("操作方法:")
        print("  マウス: ボックスをクリックして選択")
        print("  A: 現在の位置に新規物体を追加")
        print("  D: 選択した物体を削除")
        print("  0-9: 選択したボックスのIDを変更")
        print("  SPACE: 一時停止/再生")
        print("  →/N: 次のフレーム")
        print("  ←/P: 前のフレーム")
        print("  S: 現在の状態を保存")
        print("  Q: 終了")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        temp_detections = []  # 一時的な検出結果を保存
        manual_detections = {}  # {frame: [(track_id, bbox), ...]}
        
        # マウスイベントハンドラー
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_x = x
                self.mouse_y = y
                self.mouse_clicked = True
                print(f"マウスクリック: ({x}, {y})")
        
        cv2.namedWindow("Ground Truth Generator")
        cv2.setMouseCallback("Ground Truth Generator", mouse_callback)
        
        while cap.isOpened():
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            else:
                # 一時停止中は同じフレームを表示
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
            
            # ObjectTrackerでの検出と追跡
            results = self.tracker.yolo.track(frame, persist=True)
            
            # 検出結果を取得
            detections = []
            if results[0].boxes is not None:
                detections.extend(results[0].boxes)
            
            # ObjectTrackerで追跡を更新
            if detections:
                self.tracker.update_tracking(frame_count, detections)
            
            display_frame = frame.copy()
            boxes_info = []
            
            # ObjectTrackerのactive_tracksを使用
            for track_id, bbox in self.tracker.active_tracks.items():
                # 修正されたIDがあればそれを使用
                display_id = self.id_corrections.get((frame_count, track_id), track_id)
                
                # 削除マーク（None）が付いている場合は描画しない
                if display_id is None:
                    continue
                
                x, y, w, h = bbox
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                
                boxes_info.append({
                    'yolo_id': track_id,
                    'display_id': display_id,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'frame': frame_count
                })
                
                # ボックスとIDを描画
                # BBoxを緑色で表示（選択時は赤色）
                box_color = (0, 0, 255) if self.selected_box == track_id else (0, 255, 0)
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                # IDを青色で表示
                cv2.putText(display_frame, f"{display_id}", 
                          (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # マニュアルで追加された物体を描画
            if frame_count in manual_detections:
                for manual_track_id, manual_bbox in manual_detections[frame_count]:
                    x, y, w, h = manual_bbox
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    display_id = manual_track_id
                    boxes_info.append({
                        'yolo_id': manual_track_id,
                        'display_id': display_id,
                        'bbox': (x1, y1, x2, y2),
                        'frame': frame_count
                    })
                    
                    # BBoxを緑色で表示（選択時は赤色）
                    box_color = (0, 0, 255) if self.selected_box == manual_track_id else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                    # IDを青色で表示
                    cv2.putText(display_frame, f"{display_id}", 
                              (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # 情報表示
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "PAUSED" if self.paused else "PLAYING", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if self.selected_box is not None:
                cv2.putText(display_frame, f"Selected: {self.selected_box}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Ground Truth Generator", display_frame)
            
            # レビューインターバルで自動停止
            if not self.paused and frame_count % review_interval == 0:
                self.paused = True
                print(f"\n--- Frame {frame_count} でレビュー ---")
                print(f"検出数: {len(boxes_info)}")
            
            # マウスクリックでボックスを選択
            if self.mouse_clicked:
                closest_box = None
                min_distance = float('inf')
                
                # 既存のボックスをチェック
                for track_id, bbox in self.tracker.active_tracks.items():
                    x, y, w, h = bbox
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    if (x1 <= self.mouse_x <= x2) and (y1 <= self.mouse_y <= y2):
                        distance = np.sqrt((self.mouse_x - x)**2 + (self.mouse_y - y)**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_box = track_id
                
                # マニュアル追加されたボックスをチェック
                if frame_count in manual_detections:
                    for manual_track_id, manual_bbox in manual_detections[frame_count]:
                        x, y, w, h = manual_bbox
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        
                        if (x1 <= self.mouse_x <= x2) and (y1 <= self.mouse_y <= y2):
                            distance = np.sqrt((self.mouse_x - x)**2 + (self.mouse_y - y)**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_box = manual_track_id
                
                if closest_box is not None:
                    self.selected_box = closest_box
                    print(f"ボックスを選択: {closest_box}")
                
                self.mouse_clicked = False
            
            # キー入力処理
            key = cv2.waitKey(1 if not self.paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('s'):
                self._save_ground_truth(temp_detections, output_path)
                print(f"保存しました: {output_path}")
            elif key == ord('a'):
                # 新しい物体を追加（マウスクリックした位置に）
                if self.mouse_x is not None and self.mouse_y is not None:
                    if frame_count not in manual_detections:
                        manual_detections[frame_count] = []
                    
                    new_track_id = max([t for t in self.tracker.active_tracks.keys()] + 
                                      [t for t in (manual_detections.get(frame_count, []))] +
                                      [0]) + 1
                    
                    # デフォルトサイズのbboxを作成
                    default_w = 50
                    default_h = 50
                    new_bbox = [self.mouse_x, self.mouse_y, default_w, default_h]
                    manual_detections[frame_count].append((new_track_id, new_bbox))
                    self.selected_box = new_track_id
                    print(f"新しい物体を追加: ID {new_track_id} at ({self.mouse_x}, {self.mouse_y})")
            elif key == ord('d'):
                # 選択した物体を削除
                if self.selected_box is not None:
                    # マニュアル追加された物体を削除
                    if frame_count in manual_detections:
                        manual_detections[frame_count] = [(t, b) for t, b in manual_detections[frame_count] 
                                                         if t != self.selected_box]
                        print(f"マニュアル物体 {self.selected_box} を削除")
                    
                    # ObjectTrackerのトラックを一時的に無効化
                    # 現在のフレームでのマッチングをスキップするために
                    # id_correctionsを使って削除マークを付ける
                    self.id_corrections[(frame_count, self.selected_box)] = None  # Noneは削除を意味する
                    print(f"トラック {self.selected_box} をフレーム {frame_count} から削除")
                    
                    self.selected_box = None
            elif key == ord('n') or key == 83:  # 右矢印
                frame_count += 1
                self.paused = True
            elif key == ord('p') or key == 81:  # 左矢印
                frame_count = max(0, frame_count - 1)
                self.paused = True
            elif ord('0') <= key <= ord('9'):
                # 数字キーでIDを変更
                new_id = key - ord('0')
                if self.selected_box is not None:
                    if frame_count in manual_detections:
                        # マニュアル追加された物体の場合
                        for i, (track_id, bbox) in enumerate(manual_detections[frame_count]):
                            if track_id == self.selected_box:
                                manual_detections[frame_count][i] = (new_id, bbox)
                                print(f"ID {self.selected_box} を {new_id} に変更")
                                self.selected_box = new_id
                                break
                    else:
                        # ObjectTrackerの物体の場合
                        self.id_corrections[(frame_count, self.selected_box)] = new_id
                        print(f"ID {self.selected_box} を {new_id} に変更")
            
            # 検出結果を一時保存
            temp_detections.append({
                'frame': frame_count,
                'boxes': boxes_info
            })
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 最終保存
        self._save_ground_truth(temp_detections, output_path)
        print(f"\nGround Truth生成完了: {output_path}")
        return output_path
    
    def method2_full_automatic(self, video_path, output_path, confidence_threshold=0.5):
        """
        方法2: 完全自動生成（比較用）
        
        ObjectTrackerの結果をそのままGround Truthとして保存
        異なる設定で動かした時の比較基準として使用
        
        Args:
            video_path: 入力動画
            output_path: Ground Truth保存先
            confidence_threshold: 信頼度閾値
        """
        print("\n=== 方法2: 完全自動Ground Truth生成（ObjectTracker使用）===")
        print("ObjectTrackerの結果をそのまま保存します...")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLOで検出
            results = self.tracker.yolo.track(frame, persist=True)
            
            # 検出結果を取得
            detections = []
            if results[0].boxes is not None:
                detections.extend(results[0].boxes)
            
            # ObjectTrackerで追跡を更新
            if detections:
                self.tracker.update_tracking(frame_count, detections)
            
            # ObjectTrackerのactive_tracksを使用
            for track_id, bbox in self.tracker.active_tracks.items():
                x, y, w, h = bbox
                
                # MOTChallenge形式 (bboxは中心座標なので、左上角に変換)
                all_detections.append({
                    'frame': frame_count,
                    'id': track_id,
                    'x': x - w/2,
                    'y': y - h/2,
                    'w': w,
                    'h': h,
                    'conf': 1.0  # ObjectTrackerは信頼度を保持していないので1.0
                })
            
            if frame_count % 100 == 0:
                print(f"処理済み: {frame_count} フレーム、アクティブトラック: {len(self.tracker.active_tracks)}")
        
        cap.release()
        
        # MOTChallenge形式で保存
        self._save_mot_format(all_detections, output_path)
        print(f"完了: {len(all_detections)} 検出を保存")
        return output_path
    
    def method3_sampling(self, video_path, output_dir, sample_interval=30):
        """
        方法3: サンプリング生成（長時間動画用）
        
        N秒ごとにフレームを抽出し、画像として保存
        CVAT/LabelImgなどの外部ツールで手動アノテーション
        
        Args:
            video_path: 入力動画
            output_dir: サンプル画像の保存先
            sample_interval: サンプリング間隔（フレーム数）
        """
        print("\n=== 方法3: サンプリングGround Truth生成 ===")
        print(f"{sample_interval}フレームごとに画像を抽出します")
        print("これらの画像をCVATなどでアノテーションしてください")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_interval == 0:
                # フレームを保存
                output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
                # メタデータも保存
                metadata_path = os.path.join(output_dir, f"frame_{frame_count:06d}.json")
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'video_path': video_path
                    }, f, indent=2)
                
                print(f"保存: {output_path}")
        
        cap.release()
        
        print(f"\n完了: {saved_count} フレームを保存")
        print(f"保存先: {output_dir}")
        print("\n次のステップ:")
        print("1. CVATをインストール: pip install cvat-cli")
        print("2. 画像をアップロード")
        print("3. アノテーション作業")
        print("4. MOTChallenge形式でエクスポート")
        
        return output_dir
    
    def _save_ground_truth(self, detections, output_path):
        """検出結果をMOTChallenge形式で保存"""
        all_data = []
        for frame_data in detections:
            frame_num = frame_data['frame']
            for box_info in frame_data.get('boxes', []):
                x1, y1, x2, y2 = box_info['bbox']
                obj_id = self.id_corrections.get(
                    (frame_num, box_info['yolo_id']), 
                    box_info['display_id']
                )
                
                # 削除マーク（None）が付いている場合は保存しない
                if obj_id is None:
                    continue
                
                all_data.append({
                    'frame': frame_num,
                    'id': obj_id,
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1,
                    'conf': 1.0
                })
        
        self._save_mot_format(all_data, output_path)
    
    def _save_mot_format(self, detections, output_path):
        """MOTChallenge形式で保存"""
        with open(output_path, 'w') as f:
            for det in sorted(detections, key=lambda x: (x['frame'], x['id'])):
                # MOTChallenge形式: frame, id, x, y, w, h, conf, -1, -1, -1
                f.write(f"{det['frame']},{det['id']},{det['x']:.2f},{det['y']:.2f},"
                       f"{det['w']:.2f},{det['h']:.2f},{det.get('conf', 1.0):.2f},-1,-1,-1\n")


# 使用例
if __name__ == "__main__":
    model_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/train_results/weights/best.pt"
    generator = GroundTruthGenerator(model_path)
    
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3min_3D_left.mp4"
    
    # ========================================
    # 使用例1: 半自動生成（推奨）
    # ========================================
    print("\n【推奨】方法1: 半自動生成")
    print("トラッカーが予測 → あなたが確認・修正")
    
    gt_path = generator.method1_semi_automatic(
        video_path=video_path,
        output_path="/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt",
        review_interval=30  # 30フレームごとに確認
    )
    
    # # ========================================
    # # 使用例2: 完全自動生成（比較用）
    # # ========================================
    # print("\n方法2: 完全自動生成（別の設定との比較用）")
    
    # baseline_gt = generator.method2_full_automatic(
    #     video_path=video_path,
    #     output_path="/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/baseline.txt",
    #     confidence_threshold=0.5
    # )
    
    # # ========================================
    # # 使用例3: サンプリング（長時間動画用）
    # # ========================================
    # print("\n方法3: サンプリング生成（長時間動画向け）")
    
    # sample_dir = generator.method3_sampling(
    #     video_path=video_path,
    #     output_dir="/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/samples",
    #     sample_interval=30  # 30フレーム = 1秒ごと（30fps想定）
    # )