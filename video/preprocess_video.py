import cv2
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 入力動画パスと出力動画パス
# my local path
# input_video_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\video\training\30min\output_right.mp4"
# output_video_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\video\processed_train_video_right.mp4"

input_video_path = '/Users/rin/Documents/畢業專題/YOLO/video/training/30min/output_left.mp4'
output_video_path = './video/processed_train_video_left.mp4'

# シャープ化カーネル
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# 動画の読み込み
cap = cv2.VideoCapture(input_video_path)

# 動画情報取得
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の設定（MP4形式）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ノイズ除去
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # シャープ化
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 3チャンネルに戻して保存用に変換
    sharpened_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    out.write(sharpened_color)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f'Processed {frame_count} frames...')

# 後処理
cap.release()
out.release()
print('動画の前処理が完了しました。')
