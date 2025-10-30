# -*- coding: utf-8 -*-
"""
animate_tracks_3d_from_animcsv.py

対応CSV:
  - ./box/anim/anim_zpref.csv       （Zは観測優先）
  - ./box/anim/anim_zrts_only.csv   （ZはRTS/Hermiteのみ）

必須列:
  frame_id, global_track_id, x_final, y_final, z_final
（監査用列があっても無視。NaNは自動的にスキップ描画）

機能:
  - 指定IDのみ3Dアニメ表示（x=横px, y=奥行[m or px], z=高さpx）
  - タイトルに実時間 t=MM:SS を表示（frame_id と FPS から算出）
  - Space: 一時停止/再開
  - [ / ] : トレイル長を減らす/増やす
  - PLAY_RATE で再生速度を変更（0.5=半速, 0.25=1/4速, 2.0=2倍速）
  - 保存時は FFMpegWriter を使用（ffmpeg 必要）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ====== 設定 ======
# ここを片方に切り替え
CSV_IN = "./box/anim/anim_zprefz.csv"
#CSV_IN = "./box/anim/anim_zrts_only.csv"

IMG_W_PX = 640
IMG_H_PX = 480

Z_MIN, Z_MAX = 1.0, 1.3      # 奥行き[m]の表示レンジ（z_finalはm想定）
USE_Z_IN_METERS = True       # True: z[m]のまま表示 / False: px換算して表示
Z_TO_PX = 800.0              # USE_Z_IN_METERS=False のときの m→px 換算係数

SAVE_VIDEO = False           # Trueでmp4保存
FPS = 20                     # ★ 実動画のFPS
INTERVAL_MS = 40             # 1フレームの描画間隔(ms)

TRAIL_FRAMES_INIT = 120      # 初期トレイル長（フレーム数）
PLAY_RATE = 0.5             # 1.0=等速, 0.5=半速, 0.25=1/4速, 2.0=倍速

# ====== データ読込 ======
df = pd.read_csv(CSV_IN)

need = {"frame_id","global_track_id","x_final","y_final","z_final"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"必要列がありません: {missing}")

# 整形
df = df.sort_values(["frame_id","global_track_id"]).reset_index(drop=True)

# 利用可能IDを表示して選択
all_ids = sorted(df["global_track_id"].unique())
print("利用可能なID:", all_ids)
try:
    user_input = input("表示したいIDを入力してください（例: 1 / 1,3,7 / all）: ").strip()
except KeyboardInterrupt:
    print("\nキャンセルしました。"); raise SystemExit

if user_input.lower() in ("all", "*"):
    target_ids = all_ids
else:
    try:
        ids_in = [int(x) for x in user_input.split(",") if x.strip()]
        target_ids = [i for i in ids_in if i in all_ids]
        if not target_ids:
            raise ValueError
    except ValueError:
        print("⚠️ 無効な入力です。例: 1 または 1,3,7 または all")
        raise SystemExit

# ====== フレーム系列 ======
frames = sorted(df["frame_id"].unique())
fid0 = frames[0]                   # 時間計算の基準フレーム

# series[tid] = (T, 3) = (x_px, y_px, z_m)
# 欠損は NaN のまま保持（描画時にスキップ）
series = {}
for tid in target_ids:
    g = df[df["global_track_id"]==tid][["frame_id","x_final","y_final","z_final"]]
    g = g.set_index("frame_id").reindex(frames)
    series[tid] = g.values  # (x, y, z[m])

# ====== 再生順（PLAY_RATE を反映） ======
T = len(frames)
if PLAY_RATE <= 0:
    raise SystemExit("PLAY_RATE は正の値にしてください。")
if PLAY_RATE < 1.0:
    # スロー: 各フレームを複数回描く
    repeat = max(1, int(round(1.0 / PLAY_RATE)))   # 0.5→2回, 0.25→4回
    anim_order = [i for i in range(T) for _ in range(repeat)]
else:
    # 早回し: フレームを間引く
    stride = max(1, int(round(PLAY_RATE)))         # 2.0→2ステップ
    anim_order = list(range(0, T, stride))

# ====== Z(奥行き)の変換関数 ======
if USE_Z_IN_METERS:
    def depth_view(zm): return zm
    y_label = "Depth Z [m]"
    y_lim = (Z_MIN, Z_MAX)
else:
    def depth_view(zm): return zm * Z_TO_PX
    y_label = f"Depth Z [px] (≈ {int(Z_TO_PX)} px/m)"
    y_lim = (Z_MIN*Z_TO_PX, Z_MAX*Z_TO_PX)

# ====== 時間フォーマッタ ======
def fmt_time_from_frame(fid: int) -> str:
    t = (fid - fid0) / float(FPS)   # 秒
    if t < 0: t = 0.0
    m = int(t // 60); s = int(t % 60)
    return f"{m:02d}:{s:02d}"

# ====== 描画セットアップ ======
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

lines = {}
for tid in target_ids:
    (line,) = ax.plot([], [], [], linewidth=2, label=f"ID {tid}")
    lines[tid] = line

ax.set_xlim(0, IMG_W_PX)
ax.set_ylim(*y_lim)                 # y軸=奥行き
ax.set_zlim(IMG_H_PX, 0)            # z軸は上=0/下=IMG_H_PX に反転
ax.set_xlabel("X [px] (left→right)")
ax.set_ylabel(y_label)
ax.set_zlabel("Height Y [px] (0=top)")
ax.legend(ncol=2, fontsize=8)

# ====== 操作 ======
paused = False
trail_frames = TRAIL_FRAMES_INIT

def on_key(event):
    global paused, trail_frames
    if event.key == " ":
        paused = not paused
        print("⏸ Pause" if paused else "▶ Resume")
    elif event.key == "[":
        trail_frames = max(1, trail_frames - 10)
        print(f"⟵ trail = {trail_frames} frames")
    elif event.key == "]":
        trail_frames = min(len(frames), trail_frames + 10)
        print(f"⟶ trail = {trail_frames} frames")

fig.canvas.mpl_connect("key_press_event", on_key)

# ====== 更新 ======
def update(k):
    if paused:
        return []
    idx = anim_order[k]             # 実際に描くフレーム番号
    fid = frames[idx]
    start = max(0, idx - trail_frames + 1)

    # 各IDの軌跡を更新（NaNは自動スキップ：描画配列から除外）
    for tid in target_ids:
        traj = series[tid]
        seg = traj[start:idx+1, :]
        # NaN除去（x or y or z が NaN の点は捨てる）
        mask = np.isfinite(seg).all(axis=1)
        if not np.any(mask):
            # このIDのこの区間に有効点なし → 空データをセット
            lines[tid].set_data([], [])
            lines[tid].set_3d_properties([])
            continue

        xs = seg[mask, 0]                      # x(px)
        ys = depth_view(seg[mask, 2])          # z(m)→表示単位
        zs = seg[mask, 1]                      # y(px)（軸で反転する）
        lines[tid].set_data(xs, ys)
        lines[tid].set_3d_properties(zs)

    t_str = fmt_time_from_frame(fid)
    ax.set_title(f"frame: {fid}  t={t_str}  |  IDs: {target_ids}  |  trail={trail_frames}  |  rate={PLAY_RATE}x")
    return list(lines.values())

anim = FuncAnimation(
    fig, update,
    frames=len(anim_order),
    interval=INTERVAL_MS,
    blit=False
)

# ====== 実行/保存 ======
if SAVE_VIDEO:
    out = f"./box/anim/anim_ids{'_'.join(map(str,target_ids))}_trail{TRAIL_FRAMES_INIT}_rate{PLAY_RATE}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=8000)
    anim.save(out, writer=writer)
    print(f"✅ 保存しました: {out}")
else:
    plt.tight_layout()
    plt.show()
