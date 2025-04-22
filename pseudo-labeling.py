from ultralytics import YOLO
import os

# load model
model = YOLO("./runs/detect/train8/weights/best.pt")

# 未ラベル画像のディレクトリ
unlabeled_dir = "./images"

# 擬似ラベルの出力先
output_dir = "./pseudo_labels"

# 推論（信頼度が高いもののみラベル付け）
model.predict(
    source=unlabeled_dir,
    conf=0.7,               # 信頼度の閾値（調整可能）
    save_txt=True,
    save_conf=True,
    project=output_dir,
    name="pseudo",
    exist_ok=True
)
