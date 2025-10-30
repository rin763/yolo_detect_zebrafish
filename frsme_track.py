# estimate_frame_lag.py
import cv2 as cv
import numpy as np
import random

LEFT  = "./video/aligned_left_manual.mp4"
RIGHT = "./video/aligned_right_manual.mp4"

SEARCH_RANGE = 30     # ±何フレームまで探索するか
N_SAMPLES    = 400    # 何フレームサンプリングするか（多いほど安定・時間↑）
SEED         = 42

def read_total_frames(path):
    cap = cv.VideoCapture(path)
    assert cap.isOpened(), f"cannot open: {path}"
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def sample_indices(total, n, margin=30):
    random.seed(SEED)
    idxs = sorted(random.sample(range(margin, total - margin), min(n, total - 2*margin)))
    return idxs

def score_pair(imgL, imgR):
    # ORB→BF→RANSACで F に対するインライア数をスコアに
    orb = cv.ORB_create(2000)
    k1,d1 = orb.detectAndCompute(imgL, None)
    k2,d2 = orb.detectAndCompute(imgR, None)
    if d1 is None or d2 is None: return 0
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1,d2)
    if not m: return 0
    m = sorted(m, key=lambda x:x.distance)[:800]
    p1 = np.float32([k1[mm.queryIdx].pt for mm in m])
    p2 = np.float32([k2[mm.trainIdx].pt for mm in m])
    F,mask = cv.findFundamentalMat(p1,p2, cv.FM_RANSAC, 1.5, 0.999)
    if F is None: return 0
    return int(mask.ravel().sum())

def estimate_lag(left_path, right_path, search_range=SEARCH_RANGE, n_samples=N_SAMPLES):
    capL = cv.VideoCapture(left_path)
    capR = cv.VideoCapture(right_path)
    assert capL.isOpened() and capR.isOpened()
    total = int(min(capL.get(cv.CAP_PROP_FRAME_COUNT), capR.get(cv.CAP_PROP_FRAME_COUNT)))
    idxs = sample_indices(total, n_samples)

    # 低解像度化で高速化
    def grab_gray(cap, idx):
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok: return None
        g = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        g = cv.resize(g, (640, 360))
        return g

    best = (None, -1)  # (lag, score)
    for lag in range(-search_range, search_range+1):
        s = 0; n = 0
        for i in idxs:
            iL = i + lag
            iR = i
            if iL < 0 or iL >= total: continue
            gL = grab_gray(capL, iL); gR = grab_gray(capR, iR)
            if gL is None or gR is None: continue
            s += score_pair(gL, gR); n += 1
        if n > 0 and s > best[1]:
            best = (lag, s)
        print(f"lag {lag:+d}: score={s} (n={n})")
    capL.release(); capR.release()
    print(f"\n>>> Estimated FRAME_LAG = {best[0]} (score={best[1]})")
    return best[0]

if __name__ == "__main__":
    estimate_lag(LEFT, RIGHT)
