# -*- coding: utf-8 -*-
"""
Gate3: 観測/補完フラグ付きの補完CSVを出力
- 入力 : ./box/gate2_filtered_kept.csv  （global_track_id / frame_id / x_abs,y_abs,Z_m を含む）
- 出力 :
    ./box/gate3_imputed_long.csv
    ./box/watch/wide_x_by_frame_track.csv
    ./box/watch/wide_y_by_frame_track.csv
    ./box/watch/wide_z_by_frame_track.csv

仕様:
- 全フレーム × 全global_track_id（>0のみ）の完全格子を生成
- 線形内挿で x_abs,y_abs,Z_m を補完（端の外挿はNaNへ戻す）
- 欠損ブロック長が MAX_INTERP_GAP を超える区間は補完結果をNaNへ戻す（too_long_gap=1）
- Gate2の“観測”有無から is_obs_* を作り、補完で埋まった位置に is_imp_* を立てる
- Zは最後に [Z_MIN, Z_MAX] にクリップ
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ====== 入出力 ======
IN_CSV     = "./box/gate2_filtered_kept.csv"
OUT_LONG   = "./box/gate3_imputed_longz.csv"
OUT_WIDE_X = "./box/watch/wide_x_by_frame_trackz.csv"
OUT_WIDE_Y = "./box/watch/wide_y_by_frame_trackz.csv"
OUT_WIDE_Z = "./box/watch/wide_z_by_frame_trackz.csv"

# ====== パラメータ ======
FPS = 20.0
DT  = 1.0 / FPS

Z_MIN, Z_MAX = 1.0, 1.3       # クリップ範囲
MAX_INTERP_GAP = 12           # 内挿する最大ギャップ長（フレーム数）。超えるとNaNへ戻す
ID_MAX = 10                   # 参考（処理には影響なし）

REQ_COLS_MIN = ["frame_id", "global_track_id", "Z_m"]
ABS_CANDIDATES = [
    ("left_x_center_abs", "left_y_center_abs"),
    ("right_x_center_abs", "right_y_center_abs"),
]

def ensure_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

def canonical_xy(row: pd.Series) -> Tuple[float, float]:
    """左abs優先→右abs。両方NaNなら (NaN, NaN)。"""
    lx, ly = row.get("left_x_center_abs", np.nan), row.get("left_y_center_abs", np.nan)
    if not (pd.isna(lx) or pd.isna(ly)):
        return float(lx), float(ly)
    rx, ry = row.get("right_x_center_abs", np.nan), row.get("right_y_center_abs", np.nan)
    return float(rx), float(ry)

def interpolate_with_gap_limit(g: pd.DataFrame,
                               cols=("x_abs","y_abs","Z_m"),
                               max_gap=12,
                               z_clip=(1.0, 1.3)) -> pd.DataFrame:
    """
    同一ID時系列g（frame_id昇順）を線形内挿。
    - 端の外挿はNaNへ戻す
    - 欠損ブロック長 > max_gap の区間は補完結果をNaNに戻す（too_long_gap=1）
    - Z_mを[z_clip]にクリップ
    戻り値: frame_id を含む DataFrame（x_abs,y_abs,Z_m, too_long_gap を列に持つ）
    """
    cols_list = list(cols)
    g = g.sort_values("frame_id").copy()
    g = g.set_index("frame_id")

    # 欠けている列は作成
    for c in cols_list:
        if c not in g.columns:
            g[c] = np.nan

    g_index = g.index
    gi = g.copy()

    # 内挿（pandasは端も埋める→後で端外挿をNaNへ戻す）
    gi[cols_list] = gi[cols_list].interpolate(method="linear", limit_direction="both")

    # 端の外挿を元に戻す（代表: x_abs の観測範囲で制限）
    f_idx = g["x_abs"].first_valid_index()
    l_idx = g["x_abs"].last_valid_index()
    if f_idx is not None:
        gi.loc[g_index < f_idx, cols_list] = np.nan
    if l_idx is not None:
        gi.loc[g_index > l_idx, cols_list] = np.nan

    # 欠損ブロック（代表: x_abs）を検出し、大穴は補完結果をNaNに戻す
    gi["too_long_gap"] = 0
    if (max_gap is not None) and (len(g_index) > 0):
        na = g["x_abs"].isna().to_numpy()
        idx = g_index.to_numpy()
        in_block = False
        start = None
        prev_f = None
        for f, isna in zip(idx, na):
            if isna and not in_block:
                in_block = True; start = f
            if (not isna) and in_block:
                in_block = False
                end = prev_f
                gap_len = int(end - start + 1)
                if gap_len > max_gap:
                    gi.loc[(gi.index >= start) & (gi.index <= end), cols_list] = np.nan
                    gi.loc[(gi.index >= start) & (gi.index <= end), "too_long_gap"] = 1
            prev_f = f
        if in_block:
            end = prev_f
            gap_len = int(end - start + 1)
            if gap_len > max_gap:
                gi.loc[(gi.index >= start) & (gi.index <= end), cols_list] = np.nan
                gi.loc[(gi.index >= start) & (gi.index <= end), "too_long_gap"] = 1

    # Zクリップ
    if "Z_m" in cols_list and z_clip is not None:
        gi["Z_m"] = gi["Z_m"].clip(lower=z_clip[0], upper=z_clip[1])

    gi = gi.reset_index()
    return gi

def pivot_save(df_src: pd.DataFrame, value_col: str, out_path: str):
    wide = df_src.pivot(index="frame_id", columns="global_track_id", values=value_col).sort_index()
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path)

def main():
    df = pd.read_csv(IN_CSV)

    # 最小必須列
    ensure_columns(df, REQ_COLS_MIN)

    # x_abs, y_abs が無ければ（左優先→右）から作成
    if not {"x_abs","y_abs"}.issubset(df.columns):
        has_abs_source = any(all(c in df.columns for c in pair) for pair in ABS_CANDIDATES)
        if not has_abs_source:
            raise ValueError("x_abs,y_abs を作成するための abs 座標（left_*_abs / right_*_abs）が見つかりません。")
        xy = df.apply(canonical_xy, axis=1, result_type="expand")
        xy.columns = ["x_abs", "y_abs"]
        df = pd.concat([df, xy], axis=1)

    # 対象IDとフレーム全集合
    frames_all = np.arange(df["frame_id"].min(), df["frame_id"].max() + 1, dtype=int)
    ids = sorted([i for i in df["global_track_id"].unique() if i > 0])
    if len(ids) > ID_MAX:
        print(f"Note: unique global_track_id = {len(ids)} (> {ID_MAX})")

    # 完全格子
    base_grid = pd.MultiIndex.from_product(
        [frames_all, ids], names=["frame_id","global_track_id"]
    ).to_frame(index=False)

    # Gate2 観測（どこが実測だったか）の元データを控える
    df_obs = df[["frame_id","global_track_id","x_abs","y_abs","Z_m"]].copy()

    # 完全格子にマージ → 現在の観測（欠損含む）
    df_full = base_grid.merge(df_obs, on=["frame_id","global_track_id"], how="left")

    # 観測フラグ（もともと値があったところ）
    obs_flags = base_grid.merge(
        df_obs.assign(
            is_obs_x=lambda d: d["x_abs"].notna().astype(int),
            is_obs_y=lambda d: d["y_abs"].notna().astype(int),
            is_obs_z=lambda d: d["Z_m"].notna().astype(int),
        )[["frame_id","global_track_id","is_obs_x","is_obs_y","is_obs_z"]],
        on=["frame_id","global_track_id"], how="left"
    ).fillna({"is_obs_x":0,"is_obs_y":0,"is_obs_z":0})

    # 各IDごとに補完＋フラグ付け
    def interp_and_flag(group: pd.DataFrame) -> pd.DataFrame:
        raw = group[["frame_id","x_abs","y_abs","Z_m"]].copy().set_index("frame_id")
        gi = interpolate_with_gap_limit(
            group[["frame_id","x_abs","y_abs","Z_m"]],
            cols=("x_abs","y_abs","Z_m"),
            max_gap=MAX_INTERP_GAP, z_clip=(Z_MIN, Z_MAX)
        ).set_index("frame_id")

        is_imp_x = ((raw["x_abs"].isna()) & gi["x_abs"].notna()).astype(int)
        is_imp_y = ((raw["y_abs"].isna()) & gi["y_abs"].notna()).astype(int)
        is_imp_z = ((raw["Z_m"].isna())   & gi["Z_m"].notna()).astype(int)

        rows = gi.copy()
        rows["is_imp_x"] = is_imp_x.reindex(rows.index, fill_value=0)
        rows["is_imp_y"] = is_imp_y.reindex(rows.index, fill_value=0)
        rows["is_imp_z"] = is_imp_z.reindex(rows.index, fill_value=0)

        rows = rows.reset_index()  # ここで frame_id が列に戻る
        # ★ グループIDを列として明示的に付与（これが無いと merge キーが無い）
        rows["global_track_id"] = int(group["global_track_id"].iloc[0])  # または rows["global_track_id"] = group.name
        return rows

    imputed = (
        df_full
        .groupby("global_track_id", group_keys=False)
        .apply(interp_and_flag)
        .reset_index(drop=True)
    )

    # 観測フラグを結合
    imputed = imputed.merge(obs_flags, on=["frame_id","global_track_id"], how="left")
    imputed["is_obs_any"] = ((imputed["is_obs_x"]==1) | (imputed["is_obs_y"]==1) | (imputed["is_obs_z"]==1)).astype(int)
    imputed["is_imp_any"] = ((imputed["is_imp_x"]==1) | (imputed["is_imp_y"]==1) | (imputed["is_imp_z"]==1)).astype(int)

    # 出力（ロング）
    Path(os.path.dirname(OUT_LONG) or ".").mkdir(parents=True, exist_ok=True)
    imputed.to_csv(OUT_LONG, index=False)

    # 出力（ワイド）
    pivot_save(imputed, "x_abs", OUT_WIDE_X)
    pivot_save(imputed, "y_abs", OUT_WIDE_Y)
    pivot_save(imputed, "Z_m",  OUT_WIDE_Z)

    # 統計ログ
    fully_filled = imputed[["x_abs","y_abs","Z_m"]].notna().all(axis=1).sum()
    remain_nan  = (~imputed[["x_abs","y_abs","Z_m"]].notna().all(axis=1)).sum()
    print("==== Gate3: 観測/補完フラグ付き補完 ====")
    print(f"Frames                  : {frames_all.min()} .. {frames_all.max()} (count={len(frames_all):,})")
    print(f"Track IDs               : {ids} (count={len(ids)})")
    print(f"Fully filled rows       : {fully_filled:,}")
    print(f"Remaining NaN rows      : {remain_nan:,}")
    print(f"Out (long)              : {OUT_LONG}")
    print(f"Out (wide X)            : {OUT_WIDE_X}")
    print(f"Out (wide Y)            : {OUT_WIDE_Y}")
    print(f"Out (wide Z)            : {OUT_WIDE_Z}")

if __name__ == "__main__":
    main()
