# stereo_yolo_z_from_labels.py
import os
import csv
import glob
import math
from typing import List, Tuple, Optional, Dict
import cv2 as cv

# ====== ここを「はかりの実測値」に合わせて設定してください ======
F_PX = 1908.0   # 焦点距離 [px] 例: Arducam 1/2.7" + 8mm -> 約1900px
B_MM = 60.0     # 基線長 [mm] 実測（はかり）
# ============================================================

IMAGE_DIR = "processed_datasets/images"   # 左右の元画像があるフォルダ
LABEL_DIR = "processed_datasets/labels"   # YOLOラベル(.txt)があるフォルダ
OUTPUT_CSV = "z_results.csv"

IMG_EXTS = [".jpg"]
LEFT_SUFFIX = "_L"
RIGHT_SUFFIX = "_R"

def find_image_path(stem: str) -> Optional[str]:
    """stem（拡張子抜き）の画像ファイルを探す"""
    for ext in IMG_EXTS:
        p = os.path.join(IMAGE_DIR, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def read_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO txt: one line per object -> class x y w h (normalized 0-1)
    return: list of (cls, x, y, w, h) (normalized floats)
    """
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            out.append((cls, x, y, w, h))
    return out

def yolo_to_pixels(yolo_box, img_w: int, img_h: int) -> Dict:
    """正規化 YOLO ボックス中心を px に変換"""
    cls, x, y, w, h = yolo_box
    return {
        "cls": cls,
        "x_px": x * img_w,
        "y_px": y * img_h,
        "w_px": w * img_w,
        "h_px": h * img_h,
    }

def load_size(image_path: str) -> Tuple[int, int]:
    """画像サイズ (w,h) を取得"""
    im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"画像が読めません: {image_path}")
    h, w = im.shape[:2]
    return w, h

def greedy_match_by_class_and_y(
    left_objs: List[Dict], right_objs: List[Dict], y_weight: float = 1.0
) -> List[Tuple[Dict, Dict]]:
    """
    左右の検出を対応付ける。
    - 同じ class のペアのみ
    - |yL - yR| が最小の組を貪欲にマッチ
    """
    matches = []
    used_r = set()
    for i, lo in enumerate(left_objs):
        best = None
        best_cost = 1e18
        for j, ro in enumerate(right_objs):
            if j in used_r:
                continue
            if lo["cls"] != ro["cls"]:
                continue
            cost = abs(lo["y_px"] - ro["y_px"]) * y_weight
            if cost < best_cost:
                best_cost = cost
                best = (j, ro)
        if best is not None:
            used_r.add(best[0])
            matches.append((lo, best[1]))
    return matches

def compute_z_mm_from_pair(xL_px: float, xR_px: float, f_px: float, b_mm: float) -> Optional[float]:
    d = xL_px - xR_px  # disparity [px] （Rectify前提：水平方向）
    if d <= 0:
        return None
    return (f_px * b_mm) / d

def main():
    # 左右ラベルの stem を収集（*_L.txt のみ列挙し、対応する *_R.txt を探す）
    left_label_paths = glob.glob(os.path.join(LABEL_DIR, f"*{LEFT_SUFFIX}.txt"))
    pairs = []
    for lp in sorted(left_label_paths):
        stem = os.path.splitext(os.path.basename(lp))[0]  # e.g., frame0001_L
        if not stem.endswith(LEFT_SUFFIX):
            continue
        stem_base = stem[:-len(LEFT_SUFFIX)]  # e.g., frame0001
        rp = os.path.join(LABEL_DIR, stem_base + RIGHT_SUFFIX + ".txt")
        if os.path.isfile(rp):
            pairs.append((stem_base, lp, rp))

    if not pairs:
        print("対応する L/R のラベルが見つかりません。ファイル名末尾の _L/_R と .txt を確認してください。")
        return

    rows = []
    for stem_base, l_txt, r_txt in pairs:
        # 対応する画像も探す（サイズ取得用）
        l_img = find_image_path(stem_base + LEFT_SUFFIX)
        r_img = find_image_path(stem_base + RIGHT_SUFFIX)
        if l_img is None or r_img is None:
            print(f"[WARN] 画像が見つかりません: {stem_base} (L:{l_img}, R:{r_img})")
            continue

        wL, hL = load_size(l_img)
        wR, hR = load_size(r_img)
        if (wL != wR) or (hL != hR):
            print(f"[WARN] 左右の解像度が異なります: {stem_base} ({wL}x{hL} vs {wR}x{hR})")
        W, H = wL, hL

        # ラベル読み込み
        left_yolo = read_yolo_labels(l_txt)
        right_yolo = read_yolo_labels(r_txt)
        if not left_yolo or not right_yolo:
            print(f"[WARN] ラベルが空: {stem_base}")
            continue

        # 正規化座標 → px 座標へ
        left_objs = [yolo_to_pixels(b, W, H) for b in left_yolo]
        right_objs = [yolo_to_pixels(b, W, H) for b in right_yolo]

        # 対応付け
        matches = greedy_match_by_class_and_y(left_objs, right_objs)

        if not matches:
            print(f"[WARN] 対応付けができませんでした: {stem_base}")
            continue

        # 各ペアで Z を計算
        for k, (lo, ro) in enumerate(matches):
            Zmm = compute_z_mm_from_pair(lo["x_px"], ro["x_px"], F_PX, B_MM)
            disparity = lo["x_px"] - ro["x_px"]
            rows.append({
                "stem": stem_base,
                "pair_id": k,
                "class": lo["cls"],
                "img_w": W,
                "img_h": H,
                "xL_px": round(lo["x_px"], 3),
                "xR_px": round(ro["x_px"], 3),
                "yL_px": round(lo["y_px"], 3),
                "yR_px": round(ro["y_px"], 3),
                "disparity_px": round(disparity, 3),
                "Z_mm": None if Zmm is None else round(Zmm, 3)
            })

    # CSV 出力
    if rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"完了: {OUTPUT_CSV} に {len(rows)} 行を書き出しました。")
    else:
        print("出力なし。入力や命名規則を確認してください。")

if __name__ == "__main__":
    main()
