# References and Related Papers

This document lists the key papers and references for the technologies **actually used** in this multi-object tracking system.

## Technologies Used in This System

### 1. YOLO (You Only Look Once) - Object Detection

**Implementation Location:** `object_tracking.py` (line 15: `self.yolo = YOLO(model_path)`)

**YOLO v8 (Ultralytics)**
- The system uses Ultralytics YOLO v8 for object detection
- Official repository: https://github.com/ultralytics/ultralytics

**Original YOLO Papers:**
- **YOLO v1**: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *CVPR 2016*.
  - Paper: https://arxiv.org/abs/1506.02640
  
- **YOLO v2 (YOLO9000)**: Redmon, J., & Farhadi, A. (2017). "YOLO9000: Better, Faster, Stronger." *CVPR 2017*.
  - Paper: https://arxiv.org/abs/1612.08242
  
- **YOLO v3**: Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." *arXiv preprint*.
  - Paper: https://arxiv.org/abs/1804.02767

### 2. Kalman Filter - State Estimation and Prediction

**Implementation Location:** `lstm_kalman_tracker.py` (lines 8-107: `KalmanFilter` class)

**Original Kalman Filter Paper:**
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*, 82(1), 35-45.
  - DOI: 10.1115/1.3662552

**Kalman Filter for Tracking:**
- Bar-Shalom, Y., & Fortmann, T. E. (1988). "Tracking and Data Association." *Academic Press*.
  - Classic textbook on tracking with Kalman filters

**Kalman Filter in Computer Vision:**
- Welch, G., & Bishop, G. (2006). "An Introduction to the Kalman Filter." *UNC Chapel Hill TR 95-041*.
  - Available online: http://www.cs.unc.edu/~welch/kalman/

### 3. LSTM (Long Short-Term Memory) - Sequence Prediction

**Implementation Location:** `lstm_kalman_tracker.py` (lines 109-139: `LSTMKalmanPredictor` class)

**Original LSTM Paper:**
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
  - DOI: 10.1162/neco.1997.9.8.1735

### 4. IoU (Intersection over Union) - Matching Metric

**Implementation Location:** `mot_evaluator.py` (lines 60-86: `compute_iou` method)

**Standard IoU:**
- Standard computer vision metric for bounding box overlap calculation
- Used in `mot_evaluator.py` for matching ground truth and predictions (line 104: `iou = self.compute_iou(...)`)

### 5. MOT Evaluation Metrics

**Implementation Location:** `mot_evaluator.py` (lines 229-333: `calculate_mota`, `calculate_motp`, `calculate_idf1` methods)

**MOT Evaluation Metrics:**
- **MOTA, MOTP**: Bernardin, K., & Stiefelhagen, R. (2008). "Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics." *EURASIP Journal on Image and Video Processing*.
  - Paper: https://link.springer.com/article/10.1155/2008/246309
  - Used in: `mot_evaluator.py` lines 229-243 (MOTA), 245-252 (MOTP)

- **IDF1**: Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016). "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking." *ECCV 2016*.
  - Paper: https://arxiv.org/abs/1609.01775
  - Used in: `mot_evaluator.py` lines 254-320 (IDF1 calculation)

- **ID Switches and Fragmentations**: Part of CLEAR MOT metrics
  - Used in: `mot_evaluator.py` lines 31-32, 176-203 (tracking and counting)

**MOT Challenge:**
- Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). "MOT16: A Benchmark for Multi-Object Tracking." *arXiv preprint*.
  - Paper: https://arxiv.org/abs/1603.00831
  - MOT Challenge website: https://motchallenge.net/

### 6. Distance-Based Matching

**Implementation Location:** 
- `object_tracking.py` (lines 177-230: `compute_enhanced_matching_cost` method)
- `mot_evaluator.py` (lines 88-90: `compute_center_distance` method)

- Uses Euclidean distance for matching detections to tracks
- Standard distance-based association method

### 7. Direction-Based Matching

**Implementation Location:** `object_tracking.py` (lines 146-175: `get_last_direction`, `compute_direction_difference` methods)

- Uses motion direction (arctan2) for enhanced matching
- Calculates direction difference between detection and track movement
- Used in: `object_tracking.py` lines 191-210 (direction cost calculation)

## System Architecture

### Hybrid LSTM + Kalman Filter Approach

**Implementation Location:** `lstm_kalman_tracker.py` (lines 141-347: `LSTMKalmanTracker` class)

- Combines Kalman Filter for short-term motion prediction
- Uses LSTM for long-term trajectory pattern learning
- Implements hybrid Re-ID scoring (lines 214-250: `compute_hybrid_reid_score`)

**Related Papers:**
- Xu, Y., Osep, A., Ban, Y., Horaud, R., Leal-Taixé, L., & Rosenhahn, B. (2019). "How to Train Your Deep Multi-Object Tracker." *CVPR 2019*.
  - Paper: https://arxiv.org/abs/1906.06618

## Implementation Details

### Custom ID Management
- **Location:** `object_tracking.py` (lines 42-61, 98-144, 232-304)
- Custom implementation with:
  - Active tracks (`active_tracks`)
  - Lost fish tracking (`lost_fish`)
  - Discarded ID reuse (`discarded_ids`)

### Greedy Matching (Not Hungarian Algorithm)
- **Location:** `mot_evaluator.py` (lines 110-135: greedy matching in `match_detections`)
- Uses greedy matching (not Hungarian algorithm) for cost matrix minimization
- Finds minimum cost pairs iteratively

## Additional Resources

### MOT Challenge Datasets
- MOTChallenge: https://motchallenge.net/
  - Provides standard datasets and evaluation protocols for MOT

### Online Resources
- **Computer Vision Foundation (CVF)**: https://openaccess.thecvf.com/
  - Open access papers from CVPR, ICCV, ECCV

- **arXiv**: https://arxiv.org/list/cs.CV/recent
  - Latest computer vision papers

