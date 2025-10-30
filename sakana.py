# manual_align_overlay.py
# 左右フレームを半透明重ね表示しながら手動調整＋パラメータ保存

import cv2 as cv
import numpy as np
import json
import os

LEFT_VIDEO  = "./video/new_output_left.mp4"
RIGHT_VIDEO = "./video/new_output_right.mp4"
FRAME_ID = 100
OUT_JSON = "align_params.json"

GRID_STEP = 60  # グリッド線間隔(px)
ALPHA = 0.5     # 右画像の透明度（0.0〜1.0）

def grab_frame(video_path, frame_id):
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Frame {frame_id} not found in {video_path}")
    return frame

def warp_affine(img, angle, dx, dy, scale):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR)

def draw_grid(img, step=60, color=(255, 255, 255), thickness=1):
    g = img.copy()
    h, w = g.shape[:2]

    # 横線
    for y in range(0, h, step):
        cv.line(g, (0, y), (w, y), color, thickness, cv.LINE_AA)
    # 縦線（← 追加）
    for x in range(0, w, step):
        cv.line(g, (x, 0), (x, h), color, thickness, cv.LINE_AA)

    return g

def main():
    left  = grab_frame(LEFT_VIDEO, FRAME_ID)
    right = grab_frame(RIGHT_VIDEO, FRAME_ID)
    H, W = left.shape[:2]

    side = "right"  # 調整対象（初期は右）
    params = {
        "left":  {"angle": 0.0, "dx": 0, "dy": 0, "scale": 1.0},
        "right": {"angle": 0.0, "dx": 0, "dy": 0, "scale": 1.0}
    }

    show_grid = True
    step_rot, step_move, step_zoom = 0.2, 2, 0.01

    print(f"""
[ 操作方法 ]
Tab : 左右の調整対象を切替
j/l : 回転 (-/+ {step_rot}°)
i/k : 上下移動 (-/+ {step_move}px)
a/d : 左右移動 (-/+ {step_move}px)
z/x : 縮小/拡大 (-/+ {step_zoom})
g   : グリッド ON/OFF
s   : 保存 ({OUT_JSON})
r   : 現在側リセット
q/Esc : 終了
""")

    while True:
        pL = params["left"];  pR = params["right"]
        left_adj  = warp_affine(left,  pL["angle"], pL["dx"], pL["dy"], pL["scale"])
        right_adj = warp_affine(right, pR["angle"], pR["dx"], pR["dy"], pR["scale"])

        # グリッド（基準：左）
        visL = draw_grid(left_adj, GRID_STEP) if show_grid else left_adj

        # 半透明重ね合わせ（右を透明に）
        overlay = cv.addWeighted(visL, 1.0, right_adj, ALPHA, 0)

        # 情報表示
        infoL = f"[LEFT]  ang={pL['angle']:.2f} dx={pL['dx']} dy={pL['dy']} sc={pL['scale']:.3f}"
        infoR = f"[RIGHT] ang={pR['angle']:.2f} dx={pR['dx']} dy={pR['dy']} sc={pR['scale']:.3f}"
        cv.putText(overlay, infoL, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv.putText(overlay, infoR, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv.putText(overlay, f"Target: {side.upper()} (Tabで切替) | Overlay α={ALPHA}", 
                   (10, H-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv.imshow("Manual Overlay Align", overlay)
        k = cv.waitKey(10) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == 9:
            side = "right" if side == "left" else "left"
        elif k == ord('g'):
            show_grid = not show_grid
        elif k == ord('r'):
            params[side] = {"angle":0.0, "dx":0, "dy":0, "scale":1.0}
        elif k == ord('s'):
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved -> {OUT_JSON}\n{json.dumps(params, indent=2)}")

        # パラメータ変更
        tgt = params[side]
        if   k == ord('j'): tgt["angle"] -= step_rot
        elif k == ord('l'): tgt["angle"] += step_rot
        elif k == ord('i'): tgt["dy"] -= step_move
        elif k == ord('k'): tgt["dy"] += step_move
        elif k == ord('a'): tgt["dx"] -= step_move
        elif k == ord('d'): tgt["dx"] += step_move
        elif k == ord('z'): tgt["scale"] = max(0.90, tgt["scale"] - step_zoom)
        elif k == ord('x'): tgt["scale"] = min(1.10, tgt["scale"] + step_zoom)

    cv.destroyAllWindows()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 最終パラメータ保存 -> {OUT_JSON}\n{json.dumps(params, indent=2)}")

if __name__ == "__main__":
    main()
