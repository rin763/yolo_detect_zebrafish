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
    
    # 最大トラッキング数の定数（object_tracking.pyと同じ）
    MAX_FISH = 10  # IDは1〜10の範囲で割り当てられます
    
    def __init__(self, model_path, max_fish=None, use_lstm=True, enable_evaluation=False, lstm_model_path=None):
        """
        ObjectTrackerを使用してGround Truthを生成
        
        Args:
            model_path: YOLOモデルのパス
            max_fish: 最大トラッキング数（Noneの場合はMAX_FISHを使用）
            use_lstm: LSTM機能を使用するか
            enable_evaluation: 評価機能を有効にするか
        """
        if max_fish is None:
            max_fish = self.MAX_FISH
        
        self.max_fish = max_fish
        self.tracker = ObjectTracker(
            model_path=model_path,
            sequence_length=10,
            max_fish=max_fish,  # object_tracking.pyと同じ設定
            use_lstm=use_lstm,
            enable_evaluation=enable_evaluation,
            lstm_model_path=lstm_model_path
        )
        self.ground_truth_data = []
        self.id_corrections = {}  # ユーザーによるID修正記録
        self.deleted_ids_per_frame = {}  # {frame: [deleted_ids, ...]} 削除されたIDを記録
        self.last_mouse_x = None  # 最後のマウスクリック位置
        self.last_mouse_y = None
        
        print(f"Ground Truth Generator initialized with max_fish={max_fish}")
        print(f"IDs will be assigned from 1 to {max_fish}")
        
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
        print(f"最大トラッキング数: {self.max_fish} (IDは1-{self.max_fish}で割り当て)")
        print("操作方法:")
        print("  マウス: ボックスをクリックして選択")
        print("  A: 新規物体を追加（最後のマウスクリック位置、または画面中央）")
        print("     - 削除されたIDを優先的に再利用します")
        print("  D: 選択した物体を削除（削除されたIDは再利用可能になります）")
        print(f"  1-{min(9, self.max_fish)}: 選択したボックスのIDを変更（キーボードの数字キー）")
        if self.max_fish == 10:
            print("  0: 選択したボックスのIDを10に変更")
        print("  SPACE: 一時停止/再生")
        print("  →/N: 次のフレーム")
        print("  ←/P: 前のフレーム")
        print("  S: 現在の状態を保存")
        print("  Q: 終了")
        
        cap = cv2.VideoCapture(video_path)
        # object_tracking.pyと同じフレームカウント方式（フレーム1から始まる）
        frame_count = 0  # 初期化は0（object_tracking.pyと同じ）
        temp_detections = []  # 一時的な検出結果を保存
        manual_detections = {}  # {frame: [(track_id, bbox), ...]}
        
        # マウスイベントハンドラー
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_x = x
                self.mouse_y = y
                self.last_mouse_x = x  # 最後のクリック位置を記録
                self.last_mouse_y = y
                self.mouse_clicked = True
                print(f"マウスクリック: ({x}, {y})")
        
        cv2.namedWindow("Ground Truth Generator")
        cv2.setMouseCallback("Ground Truth Generator", mouse_callback)
        
        # 動画の総フレーム数を取得（デバッグ用）
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"動画の総フレーム数: {total_frames}")
        
        while cap.isOpened():
            # object_tracking.pyと同じ方式：フレームを読み込む
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1  # object_tracking.pyと同じ：読み込み後にカウント
            else:
                # 一時停止中：現在のフレーム位置を再読み込み
                # CAP_PROP_POS_FRAMESは0ベースなので、frame_count-1を指定
                if frame_count == 0:
                    # 最初のフレームの場合
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                else:
                    # frame_count > 0の場合、frame_count-1の位置を指定
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                    ret, frame = cap.read()
                    if not ret:
                        break
            
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
            
            # トラッキング数の表示
            total_tracks = len(self.tracker.active_tracks) + (len(manual_detections.get(frame_count, [])) if frame_count in manual_detections else 0)
            cv2.putText(display_frame, f"Tracks: {total_tracks}/{self.max_fish}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # 削除されたIDの表示（再利用可能）
            if frame_count in self.deleted_ids_per_frame and len(self.deleted_ids_per_frame[frame_count]) > 0:
                deleted_ids = self.deleted_ids_per_frame[frame_count]
                deleted_str = f"Deleted IDs: {', '.join(map(str, deleted_ids))}"
                cv2.putText(display_frame, deleted_str, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos = 185
            else:
                y_pos = 150
            
            if self.selected_box is not None:
                cv2.putText(display_frame, f"Selected: {self.selected_box}", (10, y_pos),
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
                # 新しい物体を追加（マウスクリックした位置または最後のクリック位置を使用）
                # 位置を決定（最後のクリック位置を使用、なければ画面中央）
                if self.last_mouse_x is not None and self.last_mouse_y is not None:
                    add_x = self.last_mouse_x
                    add_y = self.last_mouse_y
                else:
                    # 画面中央を使用
                    add_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 else 320
                    add_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0 else 240
                    print(f"⚠️ マウスクリック位置がないため、画面中央 ({add_x}, {add_y}) を使用します")
                
                if frame_count not in manual_detections:
                    manual_detections[frame_count] = []
                
                # 使用中のIDを取得
                used_ids = set(self.tracker.active_tracks.keys())
                if frame_count in manual_detections:
                    used_ids.update([t for t, _ in manual_detections[frame_count]])
                
                # 削除されたIDを優先的に再利用（現在のフレームで削除されたID）
                new_track_id = None
                if frame_count in self.deleted_ids_per_frame:
                    deleted_ids = self.deleted_ids_per_frame[frame_count]
                    for deleted_id in deleted_ids:
                        if deleted_id not in used_ids and 1 <= deleted_id <= self.max_fish:
                            new_track_id = deleted_id
                            # 再利用したIDを削除リストから削除
                            self.deleted_ids_per_frame[frame_count].remove(deleted_id)
                            if not self.deleted_ids_per_frame[frame_count]:
                                del self.deleted_ids_per_frame[frame_count]
                            print(f"♻️ 削除されたID {deleted_id} を再利用")
                            break
                
                # 削除されたIDがない場合、未使用のIDを探す
                if new_track_id is None:
                    for candidate_id in range(1, self.max_fish + 1):
                        if candidate_id not in used_ids:
                            new_track_id = candidate_id
                            break
                
                if new_track_id is None:
                    print(f"❌ 新しい物体を追加できません: すべてのID (1-{self.max_fish}) が使用中です")
                else:
                    # デフォルトサイズのbboxを作成
                    default_w = 50
                    default_h = 50
                    new_bbox = [add_x, add_y, default_w, default_h]
                    manual_detections[frame_count].append((new_track_id, new_bbox))
                    self.selected_box = new_track_id
                    print(f"✓ 新しい物体を追加: ID {new_track_id} at ({add_x}, {add_y})")
            elif key == ord('d'):
                # 選択した物体を削除
                if self.selected_box is not None:
                    deleted_id = self.selected_box
                    
                    # マニュアル追加された物体を削除
                    if frame_count in manual_detections:
                        manual_detections[frame_count] = [(t, b) for t, b in manual_detections[frame_count] 
                                                         if t != deleted_id]
                        print(f"✓ マニュアル物体 {deleted_id} を削除")
                    
                    # ObjectTrackerのトラックを一時的に無効化
                    # 現在のフレームでのマッチングをスキップするために
                    # id_correctionsを使って削除マークを付ける
                    self.id_corrections[(frame_count, deleted_id)] = None  # Noneは削除を意味する
                    
                    # 削除されたIDを記録（再利用用）
                    if frame_count not in self.deleted_ids_per_frame:
                        self.deleted_ids_per_frame[frame_count] = []
                    if deleted_id not in self.deleted_ids_per_frame[frame_count]:
                        self.deleted_ids_per_frame[frame_count].append(deleted_id)
                        print(f"✓ 削除されたID {deleted_id} を記録（再利用可能）")
                    
                    print(f"✓ トラック {deleted_id} をフレーム {frame_count} から削除")
                    
                    self.selected_box = None
                else:
                    print("⚠️ 削除する物体が選択されていません")
            elif key == ord('n') or key == 83:  # 右矢印
                # 次のフレームへ（ただし動画の総フレーム数を超えない）
                if frame_count < total_frames:
                    frame_count += 1
                    self.paused = True
                    print(f"フレーム {frame_count} に移動")
                else:
                    print(f"フレーム {frame_count} は最後のフレームです")
            elif key == ord('p') or key == 81:  # 左矢印
                # 前のフレームへ（ただし1未満にならない）
                if frame_count > 1:
                    frame_count -= 1
                    self.paused = True
                    print(f"フレーム {frame_count} に移動")
                else:
                    print(f"フレーム {frame_count} は最初のフレームです")
            elif ord('0') <= key <= ord('9'):
                # 数字キーでIDを変更
                new_id = key - ord('0')
                
                # 0キーが押された場合、max_fishが10の場合は10として扱う
                if new_id == 0 and self.max_fish == 10:
                    new_id = 10
                
                # IDの範囲チェック（1〜max_fishの範囲内）
                if new_id < 1 or new_id > self.max_fish:
                    print(f"⚠️ 無効なID: {new_id}。IDは1-{self.max_fish}の範囲で指定してください")
                elif self.selected_box is not None:
                    if frame_count in manual_detections:
                        # マニュアル追加された物体の場合
                        for i, (track_id, bbox) in enumerate(manual_detections[frame_count]):
                            if track_id == self.selected_box:
                                manual_detections[frame_count][i] = (new_id, bbox)
                                print(f"✓ ID {self.selected_box} を {new_id} に変更")
                                self.selected_box = new_id
                                break
                    else:
                        # ObjectTrackerの物体の場合
                        self.id_corrections[(frame_count, self.selected_box)] = new_id
                        print(f"✓ ID {self.selected_box} を {new_id} に変更")
            
            # 検出結果を一時保存
            temp_detections.append({
                'frame': frame_count,
                'boxes': boxes_info
            })
        
        cap.release()
        cv2.destroyAllWindows()
        
        # フレーム範囲の確認（デバッグ用）
        if temp_detections:
            frames_recorded = set(d['frame'] for d in temp_detections)
            min_frame = min(frames_recorded)
            max_frame = max(frames_recorded)
            print(f"\n📊 Ground Truth統計:")
            print(f"   記録されたフレーム範囲: {min_frame} - {max_frame}")
            print(f"   記録された総フレーム数: {len(frames_recorded)}")
            print(f"   動画の総フレーム数: {total_frames}")
            print(f"   最終frame_count: {frame_count}")
            
            if max_frame != total_frames:
                print(f"   ⚠️ 警告: 最後のフレーム({total_frames})が記録されていません")
                print(f"   記録された最大フレーム: {max_frame}")
            
            # フレーム1が記録されているか確認
            if 1 not in frames_recorded:
                print(f"   ⚠️ 警告: 最初のフレーム(1)が記録されていません")
            
            # 連続するフレームが欠けているか確認
            missing_frames = []
            for f in range(1, total_frames + 1):
                if f not in frames_recorded:
                    missing_frames.append(f)
            
            if missing_frames:
                print(f"   ⚠️ 警告: {len(missing_frames)}フレームが欠けています")
                if len(missing_frames) <= 10:
                    print(f"   欠けているフレーム: {missing_frames}")
                else:
                    print(f"   欠けているフレーム（最初の10個）: {missing_frames[:10]}...")
        
        # 最終保存
        self._save_ground_truth(temp_detections, output_path)
        print(f"\nGround Truth生成完了: {output_path}")
        return output_path
    
    
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
    
    def add_single_frame(self, video_path, frame_number, existing_gt_path=None, output_path=None):
        """
        特定のフレームだけを処理して、既存のGround Truthファイルに追加
        
        Args:
            video_path: 入力動画
            frame_number: 処理するフレーム番号（1ベース）
            existing_gt_path: 既存のGround Truthファイルのパス（Noneの場合は新規作成）
            output_path: 出力ファイルのパス（Noneの場合はexisting_gt_pathと同じ）
        
        Returns:
            保存されたファイルのパス
        """
        print(f"\n=== 単一フレーム処理: Frame {frame_number} ===")
        
        if output_path is None:
            if existing_gt_path:
                output_path = existing_gt_path
            else:
                output_path = "ground_truth/frame_{frame_number}.txt"
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_number < 1 or frame_number > total_frames:
            print(f"❌ エラー: フレーム番号 {frame_number} は範囲外です（1-{total_frames}）")
            cap.release()
            return None
        
        # 指定フレームに移動（0ベース）
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ エラー: フレーム {frame_number} を読み込めませんでした")
            cap.release()
            return None
        
        print(f"✅ フレーム {frame_number} を読み込みました")
        
        # ObjectTrackerでの検出と追跡
        results = self.tracker.yolo.track(frame, persist=True)
        detections = []
        if results[0].boxes is not None:
            detections.extend(results[0].boxes)
        
        # ObjectTrackerで追跡を更新
        if detections:
            self.tracker.update_tracking(frame_number, detections)
        
        # フレームデータを収集
        boxes_info = []
        
        # ObjectTrackerのactive_tracksを使用
        for track_id, bbox in self.tracker.active_tracks.items():
            display_id = self.id_corrections.get((frame_number, track_id), track_id)
            
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
                'frame': frame_number
            })
        
        cap.release()
        
        # 既存のGround Truthを読み込む（ある場合）
        existing_data = []
        if existing_gt_path and os.path.exists(existing_gt_path):
            print(f"📖 既存のGround Truthを読み込み中: {existing_gt_path}")
            with open(existing_gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    try:
                        existing_frame = int(parts[0])
                        # 欠けているフレームのデータは除外
                        if existing_frame != frame_number:
                            existing_data.append({
                                'frame': existing_frame,
                                'id': int(parts[1]),
                                'x': float(parts[2]),
                                'y': float(parts[3]),
                                'w': float(parts[4]),
                                'h': float(parts[5]),
                                'conf': float(parts[6]) if len(parts) > 6 else 1.0
                            })
                    except ValueError:
                        continue
            print(f"   既存データ: {len(existing_data)} エントリ")
        
        # 新しいフレームのデータを追加
        new_frame_data = []
        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info['bbox']
            obj_id = self.id_corrections.get(
                (frame_number, box_info['yolo_id']), 
                box_info['display_id']
            )
            
            if obj_id is None:
                continue
            
            new_frame_data.append({
                'frame': frame_number,
                'id': obj_id,
                'x': x1,
                'y': y1,
                'w': x2 - x1,
                'h': y2 - y1,
                'conf': 1.0
            })
        
        print(f"✅ フレーム {frame_number} のデータ: {len(new_frame_data)} 物体")
        
        # 既存データと新しいデータを結合
        all_data = existing_data + new_frame_data
        
        # 保存
        self._save_mot_format(all_data, output_path)
        
        print(f"\n✅ 完了！Ground Truthを保存しました: {output_path}")
        print(f"   総エントリ数: {len(all_data)}")
        print(f"   フレーム範囲: {min(d['frame'] for d in all_data)} - {max(d['frame'] for d in all_data)}")
        
        return output_path


# 使用例
if __name__ == "__main__":
    # 最大トラッキング数を設定（object_tracking.pyと同じ）
    MAX_FISH = 10  # この値を変更することで最大トラッキング数を調整できます
    
    model_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/train_results/weights/best.pt"
    lstm_model_path = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/best_reid_lstm_model.pth"
    
    # GroundTruthGeneratorを初期化（max_fishを指定）
    generator = GroundTruthGenerator(
        model_path=model_path,
        max_fish=MAX_FISH,
        lstm_model_path=lstm_model_path
    )
    
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/test/9min_3D_left.mp4"
    
    print("トラッカーが予測 → あなたが確認・修正")
    print(f"最大トラッキング数: {MAX_FISH} (IDs: 1-{MAX_FISH})")
    
    gt_path = generator.method1_semi_automatic(
        video_path=video_path,
        output_path="/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt",
        review_interval=30  # 30フレームごとに確認
    )
    
    # ========================================
    # 欠けているフレームを追加
    # ========================================
    # print("\n【追加機能】欠けているフレームを補完")
    # print("既存のGround Truthに特定フレームのデータを追加します")
    
    # existing_gt = "/Users/rin/Documents/畢業專題/yolo_detect_zebrafish/evaluate_mot_system/ground_truth/semi_auto.txt"
    # missing_frame = 6870  # 欠けているフレーム番号
    
    # generator.add_single_frame(
    #     video_path=video_path,
    #     frame_number=missing_frame,
    #     existing_gt_path=existing_gt,
    #     output_path=existing_gt  # 既存ファイルを上書き
    # )
   