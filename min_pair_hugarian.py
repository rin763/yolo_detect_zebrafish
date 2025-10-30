# -*- coding: utf-8 -*-
"""
(1) フレーム内 左右Hungarian（相対座標 |Δy|+|Δx|, gate=40px）
    -> ./box/min_xy_diff_pairs_per_left_hungarian_absdiff.csv
(2) Z計算（x_abs 視差; f_px は 640px と sensor_width=5.37mm から厳密換算; B=0.13m; scale=2.0）
    -> ./box/min_xy_pairs_with_z_scaled.csv   ※pairedのみ出力
(3) 時間トラッキング（TrackManager: ID上限10・出生/休止/再利用・固定ゲート）
    -> ./box/min_xy_and_temporal_trackmanager_cap10.csv

入力:
  ./box/re3d_upt_lstm_tracking_log_left.csv
  ./box/re3d_upt_lstm_tracking_log_right.csv
  必須列: frame_id, object_id, x_center_rel, y_center_rel, x_center_abs, y_center_abs
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import deque

# ========= 入出力 =========
LEFT_CSV  = "./box/re3d_upt_lstm_tracking_log_left.csv"
RIGHT_CSV = "./box/re3d_upt_lstm_tracking_log_right.csv"

OUT_PAIRS            = "./box/min_xy_diff_pairs_per_left_hungarian_absdiff2.csv"
OUT_PAIRS_WITH_Z     = "./box/min_xy_pairs_with_z_scaled.csv"
OUT_TEMPORAL_TRACKS  = "./box/min_xy_and_temporal_trackmanager_cap10.csv"

# ========= ステレオ/幾何パラメータ =========
IMAGE_WIDTH_PX    = 640.0
SENSOR_WIDTH_MM   = 5.37          # ご指定
FOCAL_MM          = 8.0           # レンズ焦点距離
# f_px = (f_mm / sensor_width_mm) * image_width_px
FOCAL_PX          = (FOCAL_MM / SENSOR_WIDTH_MM) * IMAGE_WIDTH_PX   # ≈ 953.27 px
BASELINE_M        = 0.13          # B[m]
Z_SCALE           = 2.0           # 実測域(≈1.0-1.3m)へ合わせるための倍率
DISPARITY_MIN_PX  = 1.0           # 極小視差のNaN化

# ========= (1) フレーム内左右Hungarian =========
SPATIAL_MAX_SCORE = 40.0          # |Δy_rel|+|Δx_rel| <= 40px で採用（ご指定）

# ========= (3) Track Manager（ID上限10） =========
MAX_TRACKS             = 10
TEMPORAL_GATE_PX       = 30.0      # 時間整合のXY L1 ゲート（固定）
USE_Z_IN_COST          = False      # Trueで |Δz| に弱重みを加算
W_Z                    = 0.2
REID_WINDOW_FRAMES     = 30         # 休止IDの再利用を許す最大ギャップ
MAX_AGE_MISSES         = 10         # アクティブで未観測許容
MIN_HITS_TO_CONFIRM    = 2          # 誕生昇格の最小ヒット
CONF_GATE_FOR_BIRTH_PX = 18.0       # 誕生用の厳しめ距離（小さめに）
# ※簡易実装：予測は“前回位置”に等速0（速度未使用）。必要なら拡張可。

REQ_COLS = ["frame_id", "object_id", "x_center_rel", "y_center_rel", "x_center_abs", "y_center_abs"]

# ---------- (1) Framewise 左右マッチ ----------
def per_frame_left_right_hungarian(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid in sorted(left["frame_id"].unique()):
        Lf = left[left["frame_id"] == fid].reset_index(drop=True)
        Rf = right[right["frame_id"] == fid].reset_index(drop=True)

        L_ids  = Lf["object_id"].to_numpy(int)
        Lxr, Lyr = Lf["x_center_rel"].to_numpy(float), Lf["y_center_rel"].to_numpy(float)
        Lxa, Lya = Lf["x_center_abs"].to_numpy(float), Lf["y_center_abs"].to_numpy(float)

        if len(Rf) == 0:
            for i in range(len(Lf)):
                rows.append([int(fid), int(L_ids[i]),
                             float(Lyr[i]), float(Lxr[i]), float(Lya[i]), float(Lxa[i]),
                             -1, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan])
            continue

        R_ids  = Rf["object_id"].to_numpy(int)
        Rxr, Ryr = Rf["x_center_rel"].to_numpy(float), Rf["y_center_rel"].to_numpy(float)
        Rxa, Rya = Rf["x_center_abs"].to_numpy(float), Rf["y_center_abs"].to_numpy(float)

        dy = np.abs(Lyr[:, None] - Ryr[None, :])
        dx = np.abs(Lxr[:, None] - Rxr[None, :])
        C  = dy + dx

        ri, cj = linear_sum_assignment(C)
        assign = {int(r): int(c) for r, c in zip(ri, cj)}

        for li in range(len(Lf)):
            l_id = int(L_ids[li])
            ly_r, lx_r = float(Lyr[li]), float(Lxr[li])
            ly_a, lx_a = float(Lya[li]), float(Lxa[li])

            if li in assign:
                rj = assign[li]
                r_id = int(R_ids[rj])
                ry_r, rx_r = float(Ryr[rj]), float(Rxr[rj])
                ry_a, rx_a = float(Rya[rj]), float(Rxa[rj])

                diff_y, diff_x = float(dy[li, rj]), float(dx[li, rj])
                score = diff_y + diff_x
                abs_diff_x = abs(lx_a - rx_a)

                if score > SPATIAL_MAX_SCORE:
                    # 未マッチ化
                    rows.append([int(fid), l_id, ly_r, lx_r, ly_a, lx_a,
                                 -1, np.nan, np.nan, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, np.nan])
                else:
                    rows.append([int(fid), l_id, ly_r, lx_r, ly_a, lx_a,
                                 r_id, ry_r, rx_r, ry_a, rx_a,
                                 diff_y, diff_x, score, abs_diff_x])
            else:
                rows.append([int(fid), l_id, ly_r, lx_r, ly_a, lx_a,
                             -1, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan])

    df = pd.DataFrame(rows, columns=[
        "frame_id",
        "left_id",
        "left_y_center_rel",  "left_x_center_rel",
        "left_y_center_abs",  "left_x_center_abs",
        "right_id",
        "right_y_center_rel", "right_x_center_rel",
        "right_y_center_abs", "right_x_center_abs",
        "diff_y", "diff_x", "score_xy", "abs_diff_x"
    ])
    return df

# ---------- (2) Z 計算 ----------
def compute_z(pairs_df: pd.DataFrame) -> pd.DataFrame:
    df = pairs_df.copy()
    both = (df["right_id"] != -1)
    disp = np.where(both, np.abs(df["left_x_center_abs"] - df["right_x_center_abs"]), np.nan)
    disp = np.where(disp < DISPARITY_MIN_PX, np.nan, disp)
    df["disparity_px"] = disp
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Z_raw_m"] = (FOCAL_PX * BASELINE_M) / df["disparity_px"]
    df["Z_m"] = df["Z_raw_m"] * Z_SCALE
    return df

# ---------- (3) Track Manager (ID上限10) ----------
class Track:
    __slots__ = ("tid","last_frame","hits","age_miss","confirmed",
                 "x","y","z","history_len")
    def __init__(self, tid, frame, x, y, z):
        self.tid = tid
        self.last_frame = frame
        self.hits = 1
        self.age_miss = 0
        self.confirmed = (self.hits >= MIN_HITS_TO_CONFIRM)
        self.x, self.y, self.z = x, y, z
        self.history_len = 1

    def cost_to(self, x, y, z):
        if np.isnan(x) or np.isnan(y) or np.isnan(self.x) or np.isnan(self.y):
            return math.inf
        c = abs(self.x - x) + abs(self.y - y)
        if USE_Z_IN_COST and (not np.isnan(self.z)) and (not np.isnan(z)):
            c += W_Z * abs(self.z - z)
        return c

def canonical_xy_z(row):
    """左abs優先→無ければ右abs。zはその行の Z_m（なければNaN）。"""
    lx, ly = row.get("left_x_center_abs", np.nan), row.get("left_y_center_abs", np.nan)
    if not (np.isnan(lx) or np.isnan(ly)):
        x, y = float(lx), float(ly)
    else:
        rx, ry = row.get("right_x_center_abs", np.nan), row.get("right_y_center_abs", np.nan)
        x, y = float(rx), float(ry)
    z = row.get("Z_m", np.nan)
    try:
        z = float(z)
    except:
        z = np.nan
    return x, y, z

def run_track_manager(pairs_with_z: pd.DataFrame) -> pd.DataFrame:
    df = pairs_with_z.sort_values(["frame_id", "left_id"]).reset_index(drop=True).copy()
    frames = df["frame_id"].drop_duplicates().to_list()

    # IDプール（1..10）
    id_pool = deque(range(1, MAX_TRACKS + 1))
    active: dict[int, Track] = {}           # tid -> Track
    dormant: dict[int, int] = {}            # tid -> expire_frame（REID_WINDOW内はIDキープ）

    out_rows = []

    for f in frames:
        idx = df.index[df["frame_id"] == f].to_numpy()
        obs = []
        for i in idx:
            x, y, z = canonical_xy_z(df.loc[i])
            obs.append((i, x, y, z))

        # 1) アクティブTrack と 観測 の割当
        act_ids = list(active.keys())
        if len(act_ids) and len(obs):
            # 成本行列
            C = np.full((len(act_ids), len(obs)), fill_value=TEMPORAL_GATE_PX + 1e6, dtype=float)
            for ai, tid in enumerate(act_ids):
                for oj, (_, x, y, z) in enumerate(obs):
                    c = active[tid].cost_to(x, y, z)
                    if c <= TEMPORAL_GATE_PX:
                        C[ai, oj] = c
            # Hungarian
            ri, cj = linear_sum_assignment(C)
            matched_obs = set()
            matched_act = set()
            for ai, oj in zip(ri, cj):
                if C[ai, oj] <= TEMPORAL_GATE_PX:
                    tid = act_ids[ai]
                    i, x, y, z = obs[oj]
                    # 更新
                    tr = active[tid]
                    tr.x, tr.y, tr.z = x, y, z
                    tr.hits += 1
                    tr.confirmed = (tr.hits >= MIN_HITS_TO_CONFIRM)
                    tr.age_miss = 0
                    tr.last_frame = f
                    tr.history_len += 1
                    matched_obs.add(oj)
                    matched_act.add(ai)
                    out_rows.append((i, tid))
            # 未割当 Track は miss
            for ai, tid in enumerate(act_ids):
                if ai not in matched_act:
                    tr = active[tid]
                    tr.age_miss += 1
                    tr.last_frame = f
            # 未割当 観測 → 新規or再利用
            for oj, (i, x, y, z) in enumerate(obs):
                if oj in matched_obs:
                    continue
                # まず誕生用の厳しめゲート（近傍既存Trackがないか）を軽く見る（任意）
                can_birth = True
                # 上限管理：アクティブ + 休止(未満)でMAX_TRACKS以内なら新規可
                total_ids_in_use = len(active) + len(dormant)
                if total_ids_in_use >= MAX_TRACKS or len(id_pool) == 0:
                    # 上限に達している → この観測はID付与せずスキップ
                    out_rows.append((i, -1))
                    continue
                # 新規発行
                new_tid = id_pool.popleft()
                tr = Track(new_tid, f, x, y, z)
                # 誕生の確定は MIN_HITS_TO_CONFIRM だが、とりあえず保持
                active[new_tid] = tr
                out_rows.append((i, new_tid))
        else:
            # アクティブ or 観測が空
            if len(obs):
                for (i, x, y, z) in obs:
                    total_ids_in_use = len(active) + len(dormant)
                    if total_ids_in_use >= MAX_TRACKS or len(id_pool) == 0:
                        out_rows.append((i, -1))
                        continue
                    new_tid = id_pool.popleft()
                    active[new_tid] = Track(new_tid, f, x, y, z)
                    out_rows.append((i, new_tid))
            else:
                # 観測なし → 全アクティブはmissカウント
                for tid in list(active.keys()):
                    active[tid].age_miss += 1
                    active[tid].last_frame = f

        # 2) 老化・休止・再利用管理
        # missがMAX_AGE_MISSESを超えたら休止へ
        for tid in list(active.keys()):
            if active[tid].age_miss > MAX_AGE_MISSES:
                # 休止へ移動（REID_WINDOWだけIDを保持）
                dormant[tid] = f + REID_WINDOW_FRAMES
                del active[tid]
        # 期限切れの休止IDをプールに返却
        for tid in list(dormant.keys()):
            if f >= dormant[tid]:
                # IDを再利用可能に
                del dormant[tid]
                if tid not in id_pool:
                    id_pool.append(tid)
                    id_pool = deque(sorted(id_pool))  # 小さい順に戻す

    # 出力DataFrameへ global_track_id を付与
    tid_map = dict(out_rows)  # i(index in df) -> tid
    df["global_track_id"] = df.index.map(lambda i: tid_map.get(i, -1))
    return df

# ---------- main ----------
def main():
    left  = pd.read_csv(LEFT_CSV)
    right = pd.read_csv(RIGHT_CSV)
    for c in REQ_COLS:
        if c not in left.columns or c not in right.columns:
            raise ValueError(f"必要列が不足: {REQ_COLS}")

    left  = left[REQ_COLS].copy()
    right = right[REQ_COLS].copy()

    # (1)
    pairs = per_frame_left_right_hungarian(left, right)
    os.makedirs(os.path.dirname(OUT_PAIRS) or ".", exist_ok=True)
    pairs.to_csv(OUT_PAIRS, index=False)
    print(f"[1/3] Stereo pairs -> {OUT_PAIRS}  rows={len(pairs):,}")

    # (2)
    pairs_z = compute_z(pairs)
    pairs_z_paired = pairs_z[pairs_z["right_id"] != -1].copy()
    pairs_z_paired.to_csv(OUT_PAIRS_WITH_Z, index=False)
    print(f"[2/3] Pairs with Z (paired only) -> {OUT_PAIRS_WITH_Z}  rows={len(pairs_z_paired):,}")
    print(f"     f_px={FOCAL_PX:.2f}, B={BASELINE_M}, Z_SCALE={Z_SCALE}")

    # (3) Track Manager（ID上限10）
    temporal_cap10 = run_track_manager(pairs_z)
    temporal_cap10.to_csv(OUT_TEMPORAL_TRACKS, index=False)
    n_tracks = temporal_cap10.loc[temporal_cap10["global_track_id"]>0, "global_track_id"].nunique()
    print(f"[3/3] Temporal (TrackManager, cap10) -> {OUT_TEMPORAL_TRACKS}  rows={len(temporal_cap10):,}")
    print(f"     Unique active IDs (<=10 by design): {n_tracks}")

if __name__ == "__main__":
    main()
