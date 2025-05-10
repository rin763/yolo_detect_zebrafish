from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11s.pt')
    model.train(
        data='/Users/rin/Documents/畢業專題/YOLO/yolo-3/datasets/data.yaml',
        epochs=100,
        imgsz=640,
        device='mps'  # ← ここを追加！
    )
