"""
MOT (Multiple Object Tracking) 評価システム
MOTA, MOTP, IDF1などの標準指標を計算

使い方:
    from mot_evaluator import MOTEvaluator
    
    evaluator = MOTEvaluator()
    
    # フレームごとに評価を更新
    evaluator.update_frame(ground_truth, predictions)
    
    # 最終結果を表示
    evaluator.print_summary()
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
            'true_positives': 0,      # 正しく検出・追跡できた数
            'false_positives': 0,     # 誤検出数
            'false_negatives': 0,     # 見逃し数
            'id_switches': 0,         # ID切り替わり回数
            'fragmentations': 0,      # トラック断片化回数
            'total_distance_error': 0.0,  # 総距離誤差
            'matched_count': 0,       # マッチング成功数
        }
        
        # ID管理用
        self.prev_frame_matches = {}  # {gt_id: pred_id}
        self.track_status = {}  # {gt_id: {'active': bool, 'last_matched_frame': int}}
        
        # トラックレベルの統計（メモリ効率的）
        self.track_lengths = defaultdict(int)  # {pred_id: length}
        self.track_gt_matches = defaultdict(set)  # {pred_id: set of gt_ids}
        
        # 現在のフレーム番号
        self.current_frame = 0
        
        # 軽量な位置履歴（最新N個のみ保持）
        self.position_history_size = 100
        self.recent_positions = defaultdict(lambda: defaultdict(list))  # {frame: {id: position}}
        
    def compute_iou(self, box1, box2):
        """
        IoU (Intersection over Union) を計算
        box形式: [x_center, y_center, width, height]
        """
        # 中心座標形式からコーナー座標形式に変換
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # 交差領域を計算
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        # 各ボックスの面積
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # Union面積
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def compute_center_distance(self, box1, box2):
        """中心点間の距離を計算"""
        return np.sqrt((box1[0] - box2[0])**2 + (box1[1] - box2[1])**2)
    
    def match_detections(self, ground_truth, predictions):
        """
        Ground TruthとPredictionをマッチング（ハンガリアンアルゴリズム的）
        
        Args:
            ground_truth: {gt_id: [x, y, w, h], ...}
            predictions: {pred_id: [x, y, w, h], ...}
        
        Returns:
            matches: {gt_id: pred_id}
            unmatched_gt: set of gt_ids
            unmatched_pred: set of pred_ids
        """
        if not ground_truth or not predictions:
            return {}, set(ground_truth.keys()), set(predictions.keys())
        
        # コストマトリックスを構築（IoUベース）
        gt_ids = list(ground_truth.keys())
        pred_ids = list(predictions.keys())
        
        cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                iou = self.compute_iou(ground_truth[gt_id], predictions[pred_id])
                # IoUが閾値以上ならコストは (1 - IoU)、未満なら無限大
                if iou >= self.iou_threshold:
                    cost_matrix[i, j] = 1 - iou
                else:
                    cost_matrix[i, j] = np.inf
        
        # 簡易的なグリーディマッチング（本格実装ではlinear_sum_assignment使用推奨）
        matches = {}
        unmatched_gt = set(gt_ids)
        unmatched_pred = set(pred_ids)
        
        while True:
            # 最小コストを探す
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
            
            # マッチした行と列を無効化
            cost_matrix[i, :] = np.inf
            cost_matrix[:, j] = np.inf
        
        return matches, unmatched_gt, unmatched_pred
    
    def update_frame(self, ground_truth, predictions):
        """
        1フレーム分の評価を更新
        
        Args:
            ground_truth: {gt_id: [x, y, w, h], ...}
            predictions: {pred_id: [x, y, w, h], ...}
        """
        self.current_frame += 1
        
        # マッチングを実行
        matches, unmatched_gt, unmatched_pred = self.match_detections(
            ground_truth, predictions
        )
        
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
        
        # ID切り替わりとフラグメンテーションを検出
        for gt_id, pred_id in matches.items():
            # 前フレームでこのGT IDが異なるPred IDにマッチしていた場合
            if gt_id in self.prev_frame_matches:
                if self.prev_frame_matches[gt_id] != pred_id:
                    self.frame_metrics['id_switches'] += 1
                    print(f"Frame {self.current_frame}: ID switch detected for GT {gt_id}: {self.prev_frame_matches[gt_id]} -> {pred_id}")
            
            # トラックステータスを更新
            if gt_id in self.track_status:
                if not self.track_status[gt_id]['active']:
                    # 一度途切れたトラックが再開 = フラグメンテーション
                    self.frame_metrics['fragmentations'] += 1
                    print(f"Frame {self.current_frame}: Fragmentation detected for GT {gt_id}")
            
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
        
        # 次フレームのために現在のマッチングを保存
        self.prev_frame_matches = matches.copy()
        
        # 軽量な位置履歴を更新（最新100フレームのみ）
        self.recent_positions[self.current_frame] = {
            'gt': ground_truth.copy(),
            'pred': predictions.copy()
        }
        
        # 古い位置履歴を削除
        old_frames = [f for f in self.recent_positions.keys() 
                      if f < self.current_frame - self.position_history_size]
        for frame in old_frames:
            del self.recent_positions[frame]
    
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
        IDF1 (ID F1 Score) を計算
        簡易版: 各予測トラックが主にマッチするGT IDとの一致度
        """
        total_idtp = 0  # ID True Positives
        total_idfp = 0  # ID False Positives
        total_idfn = 0  # ID False Negatives
        
        for pred_id, gt_ids in self.track_gt_matches.items():
            if len(gt_ids) == 0:
                continue
            
            # このトラックが最も多くマッチしたGT ID
            # 簡易版では、1つのGT IDに最もマッチした場合をIDTPとする
            if len(gt_ids) == 1:
                total_idtp += self.track_lengths[pred_id]
            else:
                # 複数のGT IDにマッチ = ID切り替わりがあった
                total_idfp += self.track_lengths[pred_id]
        
        # 見逃されたフレーム
        total_idfn = self.frame_metrics['false_negatives']
        
        if (total_idtp + total_idfp) == 0 or (total_idtp + total_idfn) == 0:
            return 0.0
        
        precision = total_idtp / (total_idtp + total_idfp)
        recall = total_idtp / (total_idtp + total_idfn)
        
        if precision + recall == 0:
            return 0.0
        
        idf1 = 2 * precision * recall / (precision + recall)
        return idf1
    
    def get_summary(self):
        """評価サマリーを取得"""
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