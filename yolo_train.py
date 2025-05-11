from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11s.pt')
    model.train(
        data='./processed_datasets/data.yaml',
        epochs=100,
        imgsz=640,
        verbose=True,
        batch=16,
        device='mps'  # Use 'mps' for Mac with M1/M2 chip, 'cuda' for NVIDIA GPU, or 'cpu' for CPU (if no GPU is available)
    )
