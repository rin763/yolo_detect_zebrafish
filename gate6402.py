# -*- coding: utf-8 -*-
"""
GATE2 (per-ID p95_dxy / p95_dz ベース, 下限なし・上限のみ任意)
- 先に Z のハードゲート: 1.0 <= Z_m <= 1.3 (絶対)
- 速度は XY: dxy[px/frame], Z: dz[m/frame]
- 各 ID の p95 に倍率を掛けた値をゲートに採用（下限は設けない）
- 必要なら各ゲートに上限キャップだけ設定可能（下限は無し）

入力:
  ./box/min_xy_and_temporal_trackmanager_cap10.csv  (global_track_id付き)

出力:
  ./box/gate2_full_with_flags.csv
  ./box/gate2_filtered_kept.csv
  ./box/gate2_id_summary.csv
  ./box/gate2_rejected_neighbors.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ========= 入出力 =========
IN_CSV   = "./box/min_xy_and_temporal_trackmanager_cap10.csv"
OUT_FULL = "./box/gate2_full_with_flags.csv"
OUT_KEPT = "./box/gate2_filtered_kept.csv"
OUT_SUM  = "./box/gate2_id_summary.csv"
OUT_NEI  = "./box/gate2_rejected_neighbors.csv"

# ========= 共通設定 =========
FPS = 20.0
DT  = 1.0 / FPS

# Zハードゲート（絶対）
Z_MIN, Z_MAX = 1.0, 1.3

# per-ID p95_dxy をゲートに使う設定（XY）
P95_MULTIPLIER_XY = 1.10        # p95_dxy に掛ける倍率（例: +10%）
MAX_CAP_DXY_PER_FRAME = 8.0     # 上限キャップ(px/frame)。不要なら None

# per-ID p95_dz をゲートに使う設定（Z）
ENABLE_Z_GATE = True
P95_MULTIPLIER_Z  = 1.10        # p95_dz に掛ける倍率（例: +10%）
MAX_CAP_DZ_PER_FRAME = None     # 上限キャップ(m/frame)。不要なら None

# 近傍出力（排斥行の前後Kフレームを抜粋）
NEIGHBOR_K = 2

# ========= ユーティリティ =========
def ensure_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

def canonical_xy(row: pd.Series) -> Tuple[float, float]:
    """左abs優先→右abs。両方なければ(NaN, NaN)。"""
    lx, ly = row.get("left_x_center_abs", np.nan), row.get("left_y_center_abs", np.nan)
    if not (pd.isna(lx) or pd.isna(ly)):
        return float(lx), float(ly)
    rx, ry = row.get("right_x_center_abs", np.nan), row.get("right_y_center_abs", np.nan)
    return float(rx), float(ry)

def apply_z_hard_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Z範囲 [Z_MIN, Z_MAX] を満たし、かつ global_track_id > 0 の行のみ残す。"""
    keep = (df["global_track_id"] > 0) & (df["Z_m"] >= Z_MIN) & (df["Z_m"] <= Z_MAX)
    return df.loc[keep].copy()

def compute_steps_per_track(g: pd.DataFrame) -> pd.DataFrame:
    """
    同一IDで frame が連続(Δframe=1)している箇所だけ
      - dxy: px/frame
      - dz : m /frame
    を計算（穴を跨がない）。参考用に speed_xy = dxy/DT も付与。
    """
    g = g.sort_values("frame_id").copy()
    g["dx"] = g["x_abs"].diff()
    g["dy"] = g["y_abs"].diff()
    g["dz"] = g["Z_m"].diff()
    g["dframe"] = g["frame_id"].diff()

    cont = (g["dframe"] == 1)
    g.loc[~cont, ["dx", "dy", "dz"]] = np.nan

    g["dxy"] = np.sqrt(g["dx"]**2 + g["dy"]**2)           # px/frame
    g["dz_abs"] = g["dz"].abs()                            # m /frame
    g["speed_xy"] = g["dxy"] / DT                          # px/s（参考）
    return g

def build_neighbors_for_rejected(df_full: pd.DataFrame, k: int) -> pd.DataFrame:
    """reject==1 の行について、同じIDの前後kフレームを抽出。"""
    if df_full.empty:
        return df_full.copy()
    rows = []
    for tid, g in df_full.groupby("global_track_id"):
        if tid <= 0:
            continue
        g = g.sort_values("frame_id").copy()
        idx_rej = g.index[g["reject"] == 1].to_list()
        if not idx_rej:
            continue
        frames = g["frame_id"].to_numpy()
        index_by_frame = {int(frames[i]): int(g.index[i]) for i in range(len(frames))}
        for ridx in idx_rej:
            f0 = int(g.loc[ridx, "frame_id"])
            for ff in range(f0 - k, f0 + k + 1):
                if ff in index_by_frame:
                    rows.append(index_by_frame[ff])
    if not rows:
        return pd.DataFrame(columns=df_full.columns)
    return df_full.loc[sorted(set(rows))].sort_values(["global_track_id", "frame_id"]).copy()

# ========= メイン処理 =========
def main():
    # 読み込み
    df = pd.read_csv(IN_CSV)
    ensure_columns(df, [
        "frame_id", "global_track_id",
        "left_x_center_abs", "left_y_center_abs",
        "right_x_center_abs", "right_y_center_abs",
        "Z_m"
    ])

    # カノニカルXY（左優先→右）
    xy = df.apply(canonical_xy, axis=1, result_type="expand")
    xy.columns = ["x_abs", "y_abs"]
    df = pd.concat([df, xy], axis=1)

    # Zハードゲート（絶対）
    df_z = apply_z_hard_gate(df)
    if df_z.empty:
        Path(os.path.dirname(OUT_FULL) or ".").mkdir(parents=True, exist_ok=True)
        for p in [OUT_FULL, OUT_KEPT, OUT_SUM, OUT_NEI]:
            pd.DataFrame().to_csv(p, index=False)
        print("Zハードゲート後にデータがありません。Z計算や範囲を確認してください。")
        return

    # dxy, dz (per frame) を計算（IDごと、連続区間のみ）
    df_sp = df_z.groupby("global_track_id", group_keys=False).apply(compute_steps_per_track)
    df_sp = df_sp.reset_index(drop=True)

    # === per-ID p95 をベースにゲート算出（下限なし／上限のみ任意） ===
    # XY: dxy [px/frame]
    per_id_p95_dxy = df_sp.groupby("global_track_id")["dxy"].apply(
        lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan
    )
    gate_xy = per_id_p95_dxy * P95_MULTIPLIER_XY
    if MAX_CAP_DXY_PER_FRAME is not None:
        gate_xy = gate_xy.clip(upper=MAX_CAP_DXY_PER_FRAME)
    gate_xy = gate_xy.to_dict()  # {tid: gate_px_per_frame}

    # Z : dz [m/frame]
    if ENABLE_Z_GATE:
        per_id_p95_dz = df_sp.groupby("global_track_id")["dz_abs"].apply(
            lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan
        )
        gate_z = per_id_p95_dz * P95_MULTIPLIER_Z
        if MAX_CAP_DZ_PER_FRAME is not None:
            gate_z = gate_z.clip(upper=MAX_CAP_DZ_PER_FRAME)
        gate_z = gate_z.to_dict()  # {tid: gate_m_per_frame}
    else:
        per_id_p95_dz = pd.Series(index=per_id_p95_dxy.index, dtype=float)
        gate_z = {}

    # === フラグ付け ===
    def flag_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        tid = int(g["global_track_id"].iloc[0])

        thr_xy = gate_xy.get(tid, np.nan)
        thr_z  = gate_z.get(tid, np.nan) if ENABLE_Z_GATE else np.nan

        g["gate_dxy_used_px_per_frame"] = thr_xy
        g["gate_dz_used_m_per_frame"]   = thr_z

        g["flag_speed_xy"] = (g["dxy"]    > thr_xy).astype(int)
        g["flag_speed_z"]  = (g["dz_abs"] > thr_z).astype(int) if ENABLE_Z_GATE else 0

        # どれか立てば reject
        g["reject"] = ((g["flag_speed_xy"]==1) | (g["flag_speed_z"]==1)).astype(int)
        return g

    df_flag = df_sp.groupby("global_track_id", group_keys=False).apply(flag_group).reset_index(drop=True)

    # 出力準備
    Path(os.path.dirname(OUT_FULL) or ".").mkdir(parents=True, exist_ok=True)

    # 全量（フラグ付き）
    df_flag.to_csv(OUT_FULL, index=False)

    # 通過のみ
    df_kept = df_flag[df_flag["reject"] == 0].copy()
    df_kept.to_csv(OUT_KEPT, index=False)

    # ID別サマリ
    def q95(s: pd.Series):
        s = s.dropna()
        return np.nan if s.empty else np.nanpercentile(s, 95)

    id_sum = df_flag.groupby("global_track_id").agg(
        frames=("frame_id", "nunique"),
        count=("frame_id", "size"),
        mean_dxy=("dxy", "mean"),
        p95_dxy=("dxy", q95),
        mean_dz=("dz_abs", "mean"),
        p95_dz=("dz_abs", q95),
        mean_speed_xy=("speed_xy", "mean"),
        p95_speed_xy=("speed_xy", q95),
        rejects=("reject", "sum")
    ).reset_index()

    # 使ったゲートを付与
    id_sum["gate_dxy_used_px_per_frame"] = id_sum["global_track_id"].map(gate_xy)
    id_sum["p95_multiplier_xy_used"] = P95_MULTIPLIER_XY
    id_sum["max_cap_xy_px_per_frame"] = MAX_CAP_DXY_PER_FRAME

    if ENABLE_Z_GATE:
        id_sum["gate_dz_used_m_per_frame"] = id_sum["global_track_id"].map(gate_z)
        id_sum["p95_multiplier_z_used"] = P95_MULTIPLIER_Z
        id_sum["max_cap_z_m_per_frame"] = MAX_CAP_DZ_PER_FRAME
    else:
        id_sum["gate_dz_used_m_per_frame"] = np.nan
        id_sum["p95_multiplier_z_used"] = np.nan
        id_sum["max_cap_z_m_per_frame"] = np.nan

    id_sum.to_csv(OUT_SUM, index=False)

    # 排斥行の前後抜粋
    nei = build_neighbors_for_rejected(df_flag, NEIGHBOR_K)
    nei.to_csv(OUT_NEI, index=False)

    # ログ
    total = len(df_flag)
    kept  = len(df_kept)
    rej   = total - kept
    pct   = (rej / total * 100.0) if total else 0.0

    print("==== GATE2 (per-ID p95_dxy & p95_dz, no lower floor) ====")
    print(f"Input CSV                        : {IN_CSV}")
    print(f"Z hard gate                      : [{Z_MIN}, {Z_MAX}] m")
    print(f"XY p95 multiplier                : {P95_MULTIPLIER_XY}")
    print(f"XY upper cap (px/frame)          : {MAX_CAP_DXY_PER_FRAME}")
    print(f"Z  gate enabled                  : {ENABLE_Z_GATE}")
    print(f"Z  p95 multiplier                : {P95_MULTIPLIER_Z}")
    print(f"Z  upper cap (m/frame)           : {MAX_CAP_DZ_PER_FRAME}")
    print("---------------------------------------------------------")
    print(f"Total rows (post Z hard gate)    : {total:,}")
    print(f"Kept rows                        : {kept:,}")
    print(f"Rejected rows                    : {rej:,} ({pct:.2f}%)")
    print("---------------------------------------------------------")
    print(f"Out (full+flags)                 : {OUT_FULL}")
    print(f"Out (kept only)                  : {OUT_KEPT}")
    print(f"Out (id summary)                 : {OUT_SUM}")
    print(f"Out (rejected neighbors)         : {OUT_NEI}")

if __name__ == "__main__":
    main()
