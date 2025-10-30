# -*- coding: utf-8 -*-
"""
Gate3 → RTS+Hermite(反射境界) → 完全埋め（全フレーム×全ID） → 最終CSV(2種)

入力:
  ./box/gate3_imputed_longz.csv
   必須: frame_id, global_track_id, x_abs, y_abs, Z_m, is_obs_x, is_obs_y, is_obs_z

出力:
  ./box/gate3_rts_smoothedz.csv     # RTS+Hermite 出力（推定系列のみ、観測は未改変）
  ./box/anim/anim_zprefz.csv        # x,y=観測優先, z=観測優先（観測なければ推定）→ 完全埋め
  ./box/anim/anim_zrts_onlyz.csv    # x,y=観測優先, z=推定のみ → 完全埋め

注意:
  - 観測値 (x_abs, y_abs, Z_m) には“反射/減衰/clip”等を一切かけません。
  - 反射境界処理は「推定系列 (x_s,y_s,z_s, vx_s,vy_s,vz_s)」の更新時のみに適用します。
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ===== I/O =====
IN_GATE3 = "./box/gate3_imputed_long.csv"
OUT_RTS  = "./box/gate3_rts_smoothed.csv"
OUT_DIR  = "./box/anim"
OUT_ZPREF = os.path.join(OUT_DIR, "anim_zpref.csv")
OUT_ZRTS  = os.path.join(OUT_DIR, "anim_zrts_only.csv")

# ===== Params =====
FPS = 20.0
DT  = 1.0 / FPS

# Zレンジ（物理）
Z_MIN, Z_MAX = 1.0, 1.3

# “薄い”最終ガード用 ε（ハードclip はしない）
SAFE_EPS = 1e-6

# predict-only 中のプロセスノイズ増幅（滑らか＋暴走抑制のバランス）
GAP_Q_SCALE = 7.0

# 長欠損を橋渡しする最小長（フレーム）
HERMITE_MIN_GAP = 16

# KFノイズ設定（要調整可）
Q_base_pos = 0.5
Q_base_vel = 1.0
R_x = 1.0
R_y = 1.0
R_z = 0.005

# 反射境界パラメータ（張り付き防止）
BOUNCE_KEEP = 0.6   # 反射時に速度を何倍残すか（0.4〜0.8程度）
EPS_IN = 1e-4       # 反射後に境界から内側へ戻す最小距離

# ========== Utils ==========
def ensure_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"必要列が不足: {miss}")

def carry_short_edges(series: pd.Series, limit: int) -> pd.Series:
    if not limit or limit <= 0: return series
    s = series.copy()
    s = s.fillna(method="ffill", limit=limit)
    s = s.fillna(method="bfill", limit=limit)
    return s

def reflect_in_bounds(z, vz, zmin=Z_MIN, zmax=Z_MAX):
    """
    z が [zmin, zmax] を超えたら、値を内側に反射し、速度を反転かつ減衰させる。
    観測値には適用しない。推定系列にのみ適用すること。
    """
    if not np.isfinite(z):
        return z, vz
    if z < zmin:
        over = zmin - z
        z = zmin + max(EPS_IN, over)
        vz = abs(vz) * BOUNCE_KEEP
    elif z > zmax:
        over = z - zmax
        z = zmax - max(EPS_IN, over)
        vz = -abs(vz) * BOUNCE_KEEP
    return z, vz

# ========== KF + RTS（観測未改変／推定のみ反射） ==========
def kf_rts_one_track(g: pd.DataFrame) -> pd.DataFrame:
    """
    1トラックの KF→（簡易RTS相当）：
      - 観測 (x_abs, y_abs, Z_m) は未改変
      - 推定系列 (x_s,y_s,z_s, vx_s,vy_s,vz_s) にのみ反射境界を適用
      - i0以前は定速逆外挿で埋め（そこでも反射適用）
    """
    g = g.sort_values("frame_id").reset_index(drop=True)
    n = len(g)

    obs_x = (g["is_obs_x"]==1).to_numpy()
    obs_y = (g["is_obs_y"]==1).to_numpy()
    obs_z = (g["is_obs_z"]==1).to_numpy()
    obs_any = obs_x | obs_y | obs_z

    # 観測（未改変のまま使用）
    x_obs = g["x_abs"].to_numpy(float)
    y_obs = g["y_abs"].to_numpy(float)
    z_obs = g["Z_m"  ].to_numpy(float)

    F = np.eye(6); F[0,3]=DT; F[1,4]=DT; F[2,5]=DT
    Q_base = np.diag([Q_base_pos, Q_base_pos, Q_base_pos, Q_base_vel, Q_base_vel, Q_base_vel])

    x_f = np.zeros((n,6), dtype=float)*np.nan
    P_f = [None]*n
    x_pred = np.zeros((n,6), dtype=float)*np.nan
    P_pred = [None]*n
    pred_only = np.ones(n, dtype=int)

    # 初期化フレーム
    i0 = int(np.where(obs_any)[0][0]) if obs_any.any() else 0

    x0 = np.zeros(6)
    if obs_x[i0]: x0[0]=x_obs[i0]
    if obs_y[i0]: x0[1]=y_obs[i0]
    if obs_z[i0]: x0[2]=z_obs[i0]
    P0 = np.diag([100,100,100,50,50,50])
    x_f[i0]=x0; P_f[i0]=P0; pred_only[i0]=0 if obs_any[i0] else 1

    # forward
    for k in range(i0+1, n):
        Q = Q_base * (GAP_Q_SCALE if not (obs_any[k-1]) else 1.0)
        xpr = F @ x_f[k-1]
        Ppr = F @ P_f[k-1] @ F.T + Q
        x_pred[k]=xpr; P_pred[k]=Ppr

        H=[]; z=[]; R=[]
        if obs_x[k]: H.append([1,0,0,0,0,0]); z.append(x_obs[k]); R.append(R_x)
        if obs_y[k]: H.append([0,1,0,0,0,0]); z.append(y_obs[k]); R.append(R_y)
        if obs_z[k]: H.append([0,0,1,0,0,0]); z.append(z_obs[k]); R.append(R_z)

        if H:
            H=np.array(H,float); z=np.array(z,float).reshape(-1,1); R=np.diag(R)
            yk = z - (H @ xpr.reshape(-1,1))
            S  = H @ Ppr @ H.T + R
            K  = Ppr @ H.T @ np.linalg.inv(S)
            xup = xpr.reshape(-1,1) + K @ yk
            Pup = (np.eye(6) - K @ H) @ Ppr
            x_f[k]=xup.ravel(); P_f[k]=Pup; pred_only[k]=0
        else:
            x_f[k]=xpr; P_f[k]=Ppr; pred_only[k]=1

        # 反射境界（推定系列のみ）
        x_f[k,2], x_f[k,5] = reflect_in_bounds(x_f[k,2], x_f[k,5])

    # i0以前を定速逆外挿（反射適用）
    for k in range(i0-1, -1, -1):
        x_next = x_f[k+1].copy()
        x_prev = x_next.copy()
        x_prev[0] -= x_next[3]*DT
        x_prev[1] -= x_next[4]*DT
        x_prev[2] -= x_next[5]*DT
        # 反射境界（推定系列のみ）
        x_prev[2], x_prev[5] = reflect_in_bounds(x_prev[2], x_prev[5])
        x_f[k]=x_prev; P_f[k]=P0; pred_only[k]=1

    # 最終（ここではforwardを採用）
    out = pd.DataFrame({
        "frame_id": g["frame_id"],
        "x_s": x_f[:,0], "y_s": x_f[:,1], "z_s": x_f[:,2],
        "vx_s": x_f[:,3], "vy_s": x_f[:,4], "vz_s": x_f[:,5],
        "pred_only": pred_only
    })
    # “薄い”最終ガードのみ（ハードclipは使わない）
    out["z_s"] = out["z_s"].clip(Z_MIN+SAFE_EPS, Z_MAX-SAFE_EPS)
    return out

# ========== Hermite（推定系列のみ反射） ==========
def hermite_bridge_track(df_s: pd.DataFrame,
                         obs_any: np.ndarray,
                         min_gap_len: int, dt: float) -> pd.DataFrame:
    """
    観測が無い連続区間のうち、長さ>=min_gap_len を Hermite で橋渡し。
    生成するのは推定系列のみ。観測は未改変。
    生成点にも反射境界を適用（張り付き防止）。
    """
    df = df_s.sort_values("frame_id").reset_index(drop=True).copy()
    n=len(df); obs=obs_any.astype(bool)
    gaps=[]; in_gap=False; st=None
    for i in range(n):
        if (not obs[i]) and (not in_gap):
            in_gap=True; st=i
        elif obs[i] and in_gap:
            ed=i-1
            if ed-st+1>=min_gap_len: gaps.append((st,ed))
            in_gap=False
    if in_gap:
        ed=n-1
        if ed-st+1>=min_gap_len: gaps.append((st,ed))
    if not gaps: 
        # 薄いガード
        df["z_s"] = df["z_s"].clip(Z_MIN+SAFE_EPS, Z_MAX-SAFE_EPS)
        return df

    for a,b in gaps:
        k0=a-1; k1=b+1
        if k0<0 or k1>=n: continue
        if not (obs[k0] and obs[k1]): continue

        p0=np.array([df.at[k0,"x_s"],df.at[k0,"y_s"],df.at[k0,"z_s"]],float)
        p1=np.array([df.at[k1,"x_s"],df.at[k1,"y_s"],df.at[k1,"z_s"]],float)
        v0=np.array([df.at[k0,"vx_s"],df.at[k0,"vy_s"],df.at[k0,"vz_s"]],float)
        v1=np.array([df.at[k1,"vx_s"],df.at[k1,"vy_s"],df.at[k1,"vz_s"]],float)
        N=(k1-k0); dT=N*dt
        if not np.isfinite(dT) or dT<=0: continue

        for k in range(a,b+1):
            t=(k-k0)/N
            h00= 2*t**3-3*t**2+1
            h10=   t**3-2*t**2+t
            h01=-2*t**3+3*t**2
            h11=   t**3-  t**2
            p=h00*p0 + h10*(dT*v0) + h01*p1 + h11*(dT*v1)
            df.at[k,"x_s"]=p[0]; df.at[k,"y_s"]=p[1]; df.at[k,"z_s"]=p[2]
            dh00=6*t**2-6*t; dh10=3*t**2-4*t+1; dh01=-dh00; dh11=3*t**2-2*t
            dpdn = dh00*p0 + dh10*(dT*v0) + dh01*p1 + dh11*(dT*v1)
            v=dpdn/dT
            df.at[k,"vx_s"]=v[0]; df.at[k,"vy_s"]=v[1]; df.at[k,"vz_s"]=v[2]

            # 生成点にも反射境界（推定系列のみ）
            z_new, vz_new = reflect_in_bounds(df.at[k,"z_s"], df.at[k,"vz_s"])
            df.at[k,"z_s"]  = z_new
            df.at[k,"vz_s"] = vz_new

    df["z_s"] = df["z_s"].clip(Z_MIN+SAFE_EPS, Z_MAX-SAFE_EPS)
    return df

# ========== “完全埋め”の仕上げ（観測優先） ==========
def linear_fill(s: pd.Series) -> pd.Series:
    return s.interpolate("linear", limit_direction="both")

def constant_velocity_extrapolate(df: pd.DataFrame, col_pos: str, col_vel: str) -> pd.Series:
    """
    端で NaN が残っても、近傍2点から速度を見積もって一定速度で外挿して埋め切る。
    観測値はここでは既に選別済み (x_final/y_final/z_pref or z_rts)。
    """
    s = df[col_pos].copy()
    if s.isna().sum()==0: return s
    idx = s.index.to_numpy()
    # 前方側
    first = s.first_valid_index()
    if first is not None and first>idx.min():
        j = s.index[(s.index<=first) & s.notna()].max()
        k = s.index[(s.index> first) & s.notna()].min()
        if pd.notna(j) and pd.notna(k) and k>j:
            v = (s.loc[k]-s.loc[j]) / ((k-j)*DT)
            for t in range(first-1, idx.min()-1, -1):
                s.loc[t] = s.loc[t+1] - v*DT
        else:
            s.loc[:first-1] = s.loc[first]
    # 後方側
    last = s.last_valid_index()
    if last is not None and last<idx.max():
        j = s.index[(s.index< last) & s.notna()].max()
        k = s.index[(s.index>=last) & s.notna()].min()
        if pd.notna(j) and pd.notna(k) and k>j:
            v = (s.loc[k]-s.loc[j]) / ((k-j)*DT)
            for t in range(last+1, idx.max()+1):
                s.loc[t] = s.loc[t-1] + v*DT
        else:
            s.loc[last+1:] = s.loc[last]
    return s

def finalize_full_fill(per_track: pd.DataFrame) -> pd.DataFrame:
    """
    優先度: 観測 → 推定(RTS/Hermite) → 線形補間 → 端の外挿 → 最近傍保持
    x_final, y_final, z_final を“必ず”埋める。
    ここでも観測値は未改変。z のclipは最後に“薄いガード”のみ。
    """
    g = per_track.sort_values("frame_id").copy()

    # 1) 観測優先ベース（観測をそのまま優先採用。観測値に反射やclipは適用しない）
    g["x_final"] = np.where(g["is_obs_x"]==1, g["x_abs"], g["x_s"])
    g["y_final"] = np.where(g["is_obs_y"]==1, g["y_abs"], g["y_s"])
    g["z_pref"]  = np.where(g["is_obs_z"]==1, g["Z_m"],  g["z_s"])  # 観測優先版
    g["z_rts"]   = g["z_s"]                                        # 推定のみ版

    # 2) 線形補間
    for c in ["x_final","y_final","z_pref","z_rts"]:
        g[c] = linear_fill(g[c])

    # 3) 端の定速外挿
    g = g.set_index("frame_id")
    g["x_final"] = constant_velocity_extrapolate(g, "x_final", "vx_s")
    g["y_final"] = constant_velocity_extrapolate(g, "y_final", "vy_s")
    g["z_pref"]  = constant_velocity_extrapolate(g, "z_pref",  "vz_s")
    g["z_rts"]   = constant_velocity_extrapolate(g, "z_rts",   "vz_s")
    g = g.reset_index()

    # 4) 最近傍保持
    for c in ["x_final","y_final","z_pref","z_rts"]:
        g[c] = g[c].fillna(method="ffill").fillna(method="bfill")

    # 5) z の“薄いガード”のみ（観測値も最終出力では安全のため±εに収めるがデータ自体は未改変扱い）
    g["z_pref"] = g["z_pref"].clip(Z_MIN+SAFE_EPS, Z_MAX-SAFE_EPS)
    g["z_rts"]  = g["z_rts" ].clip(Z_MIN+SAFE_EPS, Z_MAX-SAFE_EPS)

    return g

# ========== Main ==========
def main():
    df = pd.read_csv(IN_GATE3)
    ensure_cols(df, ["frame_id","global_track_id","x_abs","y_abs","Z_m","is_obs_x","is_obs_y","is_obs_z"])

    # --- 全フレーム×全ID の完全グリッド ---
    all_frames = pd.Index(range(int(df["frame_id"].min()), int(df["frame_id"].max())+1), name="frame_id")
    all_ids = sorted(df["global_track_id"].unique())
    full_idx = pd.MultiIndex.from_product([all_frames, all_ids], names=["frame_id","global_track_id"])
    df_full = df.set_index(["frame_id","global_track_id"]).reindex(full_idx).reset_index()

    # 観測フラグが欠けたら0に（観測値自体はそのまま）
    for c in ["is_obs_x","is_obs_y","is_obs_z"]:
        if c in df_full:
            df_full[c] = df_full[c].fillna(0).astype(int)
        else:
            df_full[c] = 0

    # --- KF+RTS（推定系列のみ反射境界） ---
    rts_list=[]
    for tid, g in df_full.groupby("global_track_id", sort=True):
        sm = kf_rts_one_track(g[["frame_id","x_abs","y_abs","Z_m","is_obs_x","is_obs_y","is_obs_z"]].copy())
        sm["global_track_id"]=tid
        rts_list.append(sm)
    rts = pd.concat(rts_list, ignore_index=True)

    # Hermite（長欠損にのみ、推定系列へ反射境界付きで橋渡し）
        # Hermite（長欠損にのみ、推定系列へ反射境界付きで橋渡し）
    merged = df_full[["frame_id","global_track_id","is_obs_x","is_obs_y","is_obs_z"]].merge(
        rts, on=["frame_id","global_track_id"], how="left", validate="one_to_one"
    )
    merged["is_obs_any"] = ((merged["is_obs_x"]==1)|(merged["is_obs_y"]==1)|(merged["is_obs_z"]==1)).astype(int)

    bridged=[]
    for tid, g in merged.groupby("global_track_id", sort=True):
        gb = hermite_bridge_track(
            g[["frame_id","x_s","y_s","z_s","vx_s","vy_s","vz_s"]],
            g["is_obs_any"].to_numpy(bool),
            HERMITE_MIN_GAP, DT
        )
        gb["global_track_id"]=tid
        bridged.append(gb)
    rts_b = pd.concat(bridged, ignore_index=True)

    # ✅【追加】pred_only を元の RTS 結果から引き継ぐ
    rts_b = rts_b.merge(
        rts[["frame_id","global_track_id","pred_only"]],
        on=["frame_id","global_track_id"],
        how="left"
    )


    # 検証出力（推定系列のみ）
    Path(os.path.dirname(OUT_RTS) or ".").mkdir(parents=True, exist_ok=True)
    rts_b.merge(df_full[["frame_id","global_track_id"]], on=["frame_id","global_track_id"], how="right")[
        ["frame_id","global_track_id","x_s","y_s","z_s","vx_s","vy_s","vz_s","pred_only"]
    ].to_csv(OUT_RTS, index=False)

    # --- Gate3 と RTS を結合し、完全埋め ---
    m = df_full.merge(rts_b, on=["frame_id","global_track_id"], how="left", validate="one_to_one")

    filled=[]
    for tid, g in m.groupby("global_track_id", sort=True):
        filled.append(finalize_full_fill(g.copy()))
    m2 = pd.concat(filled, ignore_index=True)

    # 出力（Z観測優先 / Z=推定のみ）…観測値は未改変で記録
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    common_cols = ["frame_id","global_track_id",
                   "x_abs","y_abs","Z_m","x_s","y_s","z_s",
                   "is_obs_x","is_obs_y","is_obs_z","pred_only"]

    m2[common_cols + ["x_final","y_final","z_pref"]].rename(columns={"z_pref":"z_final"}).to_csv(OUT_ZPREF, index=False)
    m2[common_cols + ["x_final","y_final","z_rts"] ].rename(columns={"z_rts" :"z_final"}).to_csv(OUT_ZRTS,  index=False)

    # 監査ログ
    total = len(m2)
    nan_pref = m2[["x_final","y_final","z_pref"]].isna().any(axis=1).sum()
    nan_rts  = m2[["x_final","y_final","z_rts" ]].isna().any(axis=1).sum()
    print("==== Done: fully filled (with reflecting boundaries on estimates only) ====")
    print(f"RTS out        : {OUT_RTS}")
    print(f"anim_zpref out : {OUT_ZPREF}  (Z観測優先; 観測は未改変)")
    print(f"anim_zrts  out : {OUT_ZRTS}   (Z推定のみ)")
    print(f"Total rows     : {total:,}")
    print(f"NaN rows (pref): {nan_pref:,}")
    print(f"NaN rows (rts) : {nan_rts:,}")

if __name__ == "__main__":
    main()
