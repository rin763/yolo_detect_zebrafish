"""
MOT (Multiple Object Tracking) 評価システム - 修正版
IDF1計算のバグを修正し、より正確な評価を実現
"""

import numpy as np
from collections import defaultdict
import json

class MOTEvaluator:
    """
    Multiple Object Tracking (MOT) 評価システム
    リアルタイムで評価指標を計算し、メモリ効率的にデータを管理
    """
    
    def __init__(self, iou_threshold=0.5, distance_threshold=50):
        """
        Args:
            iou_threshold: IoUマッチングの閾値
            distance_threshold: 距離ベースマッチングの閾値（ピクセル）
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        # フレームごとの評価データ（リアルタイム計算用）
        self.frame_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'id_switches': 0,
            'fragmentations': 0,
            'total_distance_error': 0.0,
            'matched_count': 0,
        }
        
        # ID管理用
        self.prev_frame_matches = {}  # {gt_id: pred_id}
        self.track_status = {}  # {gt_id: {'active': bool, 'last_matched_frame': int}}
        
        # ===== 修正: IDF1計算用の詳細なマッチング履歴 =====
        # フレームレベルのID対応を記録
        self.id_matches_per_frame = []  # [(gt_id, pred_id), ...]
        
        # 各予測IDがどのGT IDに何回マッチしたか
        self.pred_to_gt_counts = defaultdict(lambda: defaultdict(int))  # {pred_id: {gt_id: count}}
        
        # トラックレベルの統計
        self.track_lengths = defaultdict(int)
        self.track_gt_matches = defaultdict(set)
        # ===================================================
        
        # 現在のフレーム番号
        self.current_frame = 0
        
        # 軽量な位置履歴（最新N個のみ保持）
        self.position_history_size = 100
        self.recent_positions = defaultdict(lambda: defaultdict(list))
        
    def compute_iou(self, box1, box2):
        """IoU (Intersection over Union) を計算"""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def compute_center_distance(self, box1, box2):
        """中心点間の距離を計算"""
        return np.sqrt((box1[0] - box2[0])**2 + (box1[1] - box2[1])**2)
    
    def match_detections(self, ground_truth, predictions):
        """Ground TruthとPredictionをマッチング"""
        if not ground_truth or not predictions:
            return {}, set(ground_truth.keys()), set(predictions.keys())
        
        gt_ids = list(ground_truth.keys())
        pred_ids = list(predictions.keys())
        
        cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                iou = self.compute_iou(ground_truth[gt_id], predictions[pred_id])
                if iou >= self.iou_threshold:
                    cost_matrix[i, j] = 1 - iou
                else:
                    cost_matrix[i, j] = np.inf
        
        # グリーディマッチング
        matches = {}
        unmatched_gt = set(gt_ids)
        unmatched_pred = set(pred_ids)
        
        while True:
            if np.all(np.isinf(cost_matrix)):
                break
            
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            i, j = min_idx
            
            if cost_matrix[i, j] == np.inf:
                break
            
            gt_id = gt_ids[i]
            pred_id = pred_ids[j]
            
            matches[gt_id] = pred_id
            unmatched_gt.discard(gt_id)
            unmatched_pred.discard(pred_id)
            
            cost_matrix[i, :] = np.inf
            cost_matrix[:, j] = np.inf
        
        return matches, unmatched_gt, unmatched_pred
    
    def update_frame(self, ground_truth, predictions):
        """1フレーム分の評価を更新"""
        self.current_frame += 1
        
        # デバッグ情報（最初の数フレームのみ）
        if self.current_frame <= 3:
            print(f"\n🔍 Debug Frame {self.current_frame}:")
            print(f"   GT objects: {len(ground_truth)} - IDs: {list(ground_truth.keys())[:5]}")
            print(f"   Predictions: {len(predictions)} - IDs: {list(predictions.keys())[:5]}")
        
        # マッチングを実行
        matches, unmatched_gt, unmatched_pred = self.match_detections(
            ground_truth, predictions
        )
        
        if self.current_frame <= 3:
            print(f"   Matches: {len(matches)}")
            if matches:
                print(f"   Sample matches: {list(matches.items())[:3]}")
            print(f"   Unmatched GT: {len(unmatched_gt)}, Unmatched Pred: {len(unmatched_pred)}")
        
        # マッチング警告
        if len(matches) == 0 and len(ground_truth) > 0 and len(predictions) > 0:
            if self.current_frame % 100 == 1:
                print(f"⚠️ Frame {self.current_frame}: No matches! GT={len(ground_truth)}, Pred={len(predictions)}")
        
        # メトリクスを更新
        self.frame_metrics['true_positives'] += len(matches)
        self.frame_metrics['false_positives'] += len(unmatched_pred)
        self.frame_metrics['false_negatives'] += len(unmatched_gt)
        self.frame_metrics['matched_count'] += len(matches)
        
        # 距離誤差を計算
        for gt_id, pred_id in matches.items():
            distance = self.compute_center_distance(
                ground_truth[gt_id], predictions[pred_id]
            )
            self.frame_metrics['total_distance_error'] += distance
        
        # ===== 修正: IDF1用のマッチング履歴を記録 =====
        for gt_id, pred_id in matches.items():
            # フレームレベルのマッチングを記録
            self.id_matches_per_frame.append((gt_id, pred_id))
            
            # 各予測IDがどのGT IDに何回マッチしたか記録
            self.pred_to_gt_counts[pred_id][gt_id] += 1
        # ================================================
        
        # ID切り替わりとフラグメンテーションを検出
        for gt_id, pred_id in matches.items():
            if gt_id in self.prev_frame_matches:
                if self.prev_frame_matches[gt_id] != pred_id:
                    self.frame_metrics['id_switches'] += 1
                    if self.current_frame <= 10:  # デバッグ
                        print(f"Frame {self.current_frame}: ID switch GT {gt_id}: {self.prev_frame_matches[gt_id]} -> {pred_id}")
            
            if gt_id in self.track_status:
                if not self.track_status[gt_id]['active']:
                    self.frame_metrics['fragmentations'] += 1
                    if self.current_frame <= 10:
                        print(f"Frame {self.current_frame}: Fragmentation GT {gt_id}")
            
            self.track_status[gt_id] = {
                'active': True,
                'last_matched_frame': self.current_frame
            }
            
            # トラック統計を更新
            self.track_lengths[pred_id] += 1
            self.track_gt_matches[pred_id].add(gt_id)
        
        # 見逃されたGT IDはトラックが途切れた
        for gt_id in unmatched_gt:
            if gt_id in self.track_status:
                self.track_status[gt_id]['active'] = False
        
        # 次フレームのためにマッチングを保存
        self.prev_frame_matches = matches.copy()
        
        # 位置履歴を更新
        self.recent_positions[self.current_frame] = {
            'gt': ground_truth.copy(),
            'pred': predictions.copy()
        }
        
        # 古い位置履歴を削除
        old_frames = [f for f in self.recent_positions.keys() 
                      if f < self.current_frame - self.position_history_size]
        for frame in old_frames:
            del self.recent_positions[frame]
        
        # 可視化用のマッチング結果を返す
        return {
            'matches': matches,
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred,
            'ground_truth': ground_truth,
            'predictions': predictions
        }
    
    def calculate_mota(self):
        """MOTA (Multiple Object Tracking Accuracy) を計算"""
        total_gt = (self.frame_metrics['true_positives'] + 
                   self.frame_metrics['false_negatives'])
        
        if total_gt == 0:
            return 0.0
        
        mota = 1 - (
            (self.frame_metrics['false_negatives'] + 
             self.frame_metrics['false_positives'] + 
             self.frame_metrics['id_switches']) / total_gt
        )
        
        return mota
    
    def calculate_motp(self):
        """MOTP (Multiple Object Tracking Precision) を計算"""
        if self.frame_metrics['matched_count'] == 0:
            return 0.0
        
        motp = self.frame_metrics['total_distance_error'] / self.frame_metrics['matched_count']
        return motp
    
    def calculate_idf1(self):
        """
        IDF1 (ID F1 Score) を計算 - 修正版
        標準的なMOTChallenge評価方式に準拠
        """
        print(f"\n📊 IDF1 Calculation (Fixed Version):")
        
        # ===== 修正: 正確なIDTP、IDFP、IDFNの計算 =====
        total_idtp = 0  # ID True Positives
        total_idfp = 0  # ID False Positives
        total_idfn = 0  # ID False Negatives
        
        # デバッグ情報
        if len(self.pred_to_gt_counts) == 0:
            print("   ⚠️ No prediction-to-GT mappings found!")
            print("   This means no matches were recorded during tracking.")
            return 0.0
        
        print(f"   Total prediction tracks: {len(self.pred_to_gt_counts)}")
        print(f"   Total frame-level matches: {len(self.id_matches_per_frame)}")
        
        # 各予測トラックについて、最も多くマッチしたGT IDを特定
        for pred_id, gt_counts in self.pred_to_gt_counts.items():
            if not gt_counts:
                continue
            
            # このPred IDが最も多くマッチしたGT ID
            most_matched_gt = max(gt_counts.items(), key=lambda x: x[1])
            best_gt_id, best_count = most_matched_gt
            
            # IDTP: 最もマッチしたGT IDとの一致数
            total_idtp += best_count
            
            # IDFP: その他のGT IDとの誤マッチ数
            for gt_id, count in gt_counts.items():
                if gt_id != best_gt_id:
                    total_idfp += count
        
        # IDFN: False Negatives（見逃し）
        total_idfn = self.frame_metrics['false_negatives']
        
        # デバッグ情報を表示
        print(f"   IDTP (ID True Positives): {total_idtp}")
        print(f"   IDFP (ID False Positives): {total_idfp}")
        print(f"   IDFN (ID False Negatives): {total_idfn}")
        
        # Precision と Recall
        if (total_idtp + total_idfp) == 0:
            print(f"   ⚠️ ID Precision = 0 (no predictions matched)")
            precision = 0.0
        else:
            precision = total_idtp / (total_idtp + total_idfp)
        
        if (total_idtp + total_idfn) == 0:
            print(f"   ⚠️ ID Recall = 0 (no GT objects)")
            recall = 0.0
        else:
            recall = total_idtp / (total_idtp + total_idfn)
        
        print(f"   ID Precision: {precision:.4f}")
        print(f"   ID Recall: {recall:.4f}")
        
        # IDF1
        if precision + recall == 0:
            print(f"   ⚠️ IDF1 = 0 (precision + recall = 0)")
            return 0.0
        
        idf1 = 2 * precision * recall / (precision + recall)
        print(f"   IDF1: {idf1:.4f}")
        
        return idf1
    
    def get_summary(self):
        """評価サマリーを取得（カウンターはリセットしない）"""
        mota = self.calculate_mota()
        motp = self.calculate_motp()
        idf1 = self.calculate_idf1()
        
        summary = {
            'MOTA': mota,
            'MOTP': motp,
            'IDF1': idf1,
            'ID_Switches': self.frame_metrics['id_switches'],
            'Fragmentations': self.frame_metrics['fragmentations'],
            'True_Positives': self.frame_metrics['true_positives'],
            'False_Positives': self.frame_metrics['false_positives'],
            'False_Negatives': self.frame_metrics['false_negatives'],
            'Total_Frames': self.current_frame,
            'Avg_Track_Length': np.mean(list(self.track_lengths.values())) if self.track_lengths else 0,
        }
        
        return summary
    
    def print_summary(self):
        """評価サマリーを表示"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("MOT Evaluation Summary")
        print("="*60)
        print(f"Total Frames: {summary['Total_Frames']}")
        print(f"\nCore Metrics:")
        print(f"  MOTA (↑):  {summary['MOTA']:.4f}")
        print(f"  MOTP (↓):  {summary['MOTP']:.2f} pixels")
        print(f"  IDF1 (↑):  {summary['IDF1']:.4f}")
        print(f"\nDetection Metrics:")
        print(f"  True Positives:  {summary['True_Positives']}")
        print(f"  False Positives: {summary['False_Positives']}")
        print(f"  False Negatives: {summary['False_Negatives']}")
        print(f"\nTracking Quality:")
        print(f"  ID Switches:     {summary['ID_Switches']}")
        print(f"  Fragmentations:  {summary['Fragmentations']}")
        print(f"  Avg Track Length: {summary['Avg_Track_Length']:.1f} frames")
        print("="*60 + "\n")
        
        return summary
    
    def save_results(self, output_path):
        """評価結果をJSON形式で保存"""
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")
    
    def reset(self):
        """評価器をリセット（新しい評価を開始する場合）"""
        self.frame_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'id_switches': 0,
            'fragmentations': 0,
            'total_distance_error': 0.0,
            'matched_count': 0,
        }
        self.prev_frame_matches = {}
        self.track_status = {}
        self.id_matches_per_frame = []
        self.pred_to_gt_counts = defaultdict(lambda: defaultdict(int))
        self.track_lengths = defaultdict(int)
        self.track_gt_matches = defaultdict(set)
        self.current_frame = 0
        self.recent_positions = defaultdict(lambda: defaultdict(list))
        print("📌 Evaluator reset completed")