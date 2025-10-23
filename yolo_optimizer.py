#!/usr/bin/env python3
"""
YOLOモデル比較・最適化スクリプト
物体検出精度向上のためのモデル選択とパラメータ調整
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

class YOLOModelOptimizer:
    def __init__(self):
        self.models = {
            'yolo11n': 'yolo11n.pt',  # 最小・最速
            'yolo11s': 'yolo11s.pt',  # 現在使用中
            'yolo11m': 'yolo11m.pt',  # 中サイズ・高精度
            'yolo11l': 'yolo11l.pt',  # 大サイズ・最高精度
            'yolo11x': 'yolo11x.pt'   # 最大・最高精度
        }
        
    def test_model_performance(self, model_name, video_path, sample_frames=100):
        """モデルの性能をテスト"""
        print(f"\n=== Testing {model_name} ===")
        
        model = YOLO(self.models[model_name])
        cap = cv2.VideoCapture(video_path)
        
        total_detections = 0
        total_time = 0
        frame_count = 0
        
        while cap.isOpened() and frame_count < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # 検出実行
            results = model.track(
                frame,
                conf=0.3,
                iou=0.5,
                max_det=50,
                verbose=False
            )
            
            end_time = time.time()
            
            # 検出数をカウント
            if results[0].boxes is not None:
                detections = len(results[0].boxes)
                total_detections += detections
            
            total_time += (end_time - start_time)
            frame_count += 1
            
            if frame_count % 20 == 0:
                print(f"Processed {frame_count} frames, avg detections: {total_detections/frame_count:.2f}")
        
        cap.release()
        
        avg_detections = total_detections / frame_count
        avg_time = total_time / frame_count
        fps = 1.0 / avg_time
        
        print(f"Results for {model_name}:")
        print(f"  Average detections per frame: {avg_detections:.2f}")
        print(f"  Average processing time: {avg_time:.3f}s")
        print(f"  FPS: {fps:.1f}")
        
        return {
            'model': model_name,
            'avg_detections': avg_detections,
            'avg_time': avg_time,
            'fps': fps
        }
    
    def optimize_detection_parameters(self, model_path, video_path):
        """検出パラメータの最適化"""
        print(f"\n=== Optimizing detection parameters for {model_path} ===")
        
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        # パラメータの組み合わせをテスト
        param_combinations = [
            {'conf': 0.2, 'iou': 0.4, 'max_det': 30},
            {'conf': 0.3, 'iou': 0.5, 'max_det': 50},
            {'conf': 0.4, 'iou': 0.6, 'max_det': 30},
            {'conf': 0.25, 'iou': 0.45, 'max_det': 40},
        ]
        
        best_params = None
        best_score = 0
        
        for params in param_combinations:
            print(f"\nTesting params: {params}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # フレームをリセット
            total_detections = 0
            frame_count = 0
            
            while cap.isOpened() and frame_count < 50:  # 50フレームでテスト
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.track(frame, **params, verbose=False)
                
                if results[0].boxes is not None:
                    total_detections += len(results[0].boxes)
                
                frame_count += 1
            
            avg_detections = total_detections / frame_count
            score = avg_detections  # より多くの検出を優先
            
            print(f"  Average detections: {avg_detections:.2f}")
            
            if score > best_score:
                best_score = score
                best_params = params
        
        cap.release()
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.2f}")
        
        return best_params

def main():
    optimizer = YOLOModelOptimizer()
    
    # 動画パス
    video_path = "/Users/rin/Documents/畢業專題/YOLO/video/3D_left.mp4"
    
    print("=== YOLO Model Performance Comparison ===")
    
    # 各モデルの性能をテスト
    results = []
    for model_name in ['yolo11s', 'yolo11m', 'yolo11l']:  # 主要モデルのみテスト
        try:
            result = optimizer.test_model_performance(model_name, video_path, sample_frames=50)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # 結果を比較
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<10} {'Detections':<12} {'FPS':<8} {'Score':<8}")
    print("-" * 45)
    
    for result in results:
        score = result['avg_detections'] * result['fps'] / 10  # 正規化スコア
        print(f"{result['model']:<10} {result['avg_detections']:<12.2f} {result['fps']:<8.1f} {score:<8.2f}")
    
    # 最適なパラメータをテスト
    print("\n=== Parameter Optimization ===")
    best_params = optimizer.optimize_detection_parameters('yolo11s.pt', video_path)
    
    print("\n=== Recommendations ===")
    print("1. より大きなモデル（yolo11m, yolo11l）を使用することを検討")
    print("2. 検出パラメータを最適化")
    print("3. 画像前処理を追加")
    print("4. データ拡張を強化")

if __name__ == "__main__":
    main()
