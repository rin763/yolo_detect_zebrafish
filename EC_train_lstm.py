import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from Rin.EC_object_tracking import LSTMTracker, ObjectTracker
import sys

sys.stdout.reconfigure(encoding='utf-8')

class TrackingDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # データディレクトリから時系列データを読み込む
        for track_file in os.listdir(data_dir):
            if track_file.endswith('.npy'):
                track_data = np.load(os.path.join(data_dir, track_file))
                
                # シーケンスとターゲットを作成
                for i in range(len(track_data) - sequence_length):
                    sequence = track_data[i:i+sequence_length]
                    target = track_data[i+sequence_length]
                    self.sequences.append(sequence)
                    self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def collect_and_split_data(model_path, video_paths, output_base_dir="data"):
    """
    Collect data from videos and split into training and validation sets
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")
    
    # Check if video files exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directories
    train_dir = os.path.join(output_base_dir, "train_data")
    val_dir = os.path.join(output_base_dir, "val_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Initialize tracker
    try:
        tracker = ObjectTracker(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tracker: {str(e)}")
    
    # Create temporary directory for data collection
    temp_dir = os.path.join(output_base_dir, "temp_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Collect data from each video
    total_tracks = 0
    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")
        try:
            tracker.collect_training_data(video_path, temp_dir)
            # Check number of collected tracks
            track_files = [f for f in os.listdir(temp_dir) if f.endswith('.npy')]
            total_tracks += len(track_files)
            print(f"Collected {len(track_files)} tracks from {video_path}")
        except Exception as e:
            print(f"Warning: Error occurred while processing video {video_path}: {str(e)}")
            continue
    
    if total_tracks == 0:
        raise RuntimeError("No data collected from any video.")
    
    # Split collected data into training and validation sets
    track_files = [f for f in os.listdir(temp_dir) if f.endswith('.npy')]
    if not track_files:
        raise RuntimeError("No data collected.")
    
    np.random.shuffle(track_files)  # Random shuffle
    
    # Split with 8:2 ratio
    split_idx = int(len(track_files) * 0.8)
    train_files = track_files[:split_idx]
    val_files = track_files[split_idx:]
    
    # Move files to respective directories
    try:
        for file in train_files:
            src = os.path.join(temp_dir, file)
            dst = os.path.join(train_dir, file)
            os.rename(src, dst)
        
        for file in val_files:
            src = os.path.join(temp_dir, file)
            dst = os.path.join(val_dir, file)
            os.rename(src, dst)
    except Exception as e:
        raise RuntimeError(f"Error occurred while splitting data: {str(e)}")
    
    # Remove temporary directory
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Failed to remove temporary directory: {str(e)}")
    
    print(f"\nData collection and splitting completed:")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    return train_dir, val_dir
#added save_path="best_lstm_model.pth"
def train_model(model, train_data_dir, val_data_dir, batch_size=32, epochs=100, learning_rate=0.001, save_path="best_lstm_model.pth"):
    # Prepare datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Loading training data...")
    try:
        train_dataset = TrackingDataset(train_data_dir)
        print(f"Training samples: {len(train_dataset)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {str(e)}")
    
    print("Loading validation data...")
    try:
        val_dataset = TrackingDataset(val_data_dir)
        print(f"Validation samples: {len(val_dataset)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load validation data: {str(e)}")
    
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for sequences, targets in train_loader:

            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            
            # Show progress every 10 batches
            if train_batches % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {train_batches}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        # Show epoch results
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), save_path)
                #torch.save(model.state_dict(), r"C:\Users\et439\OneDrive\桌面\project\Rin\best_lstm_model.pth")

                print(f"Model saved! New best validation loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Warning: Failed to save model: {str(e)}")
        
        print("-" * 50)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    try:
        # YOLO model path
        model_path = r"C:\Users\et439\OneDrive\桌面\project\Rin\train_results\weights\best.pt"
        
        # Video paths
        video_paths = [
            r"C:\Users\et439\OneDrive\桌面\project\Rin\video\processed_train_video_left.mp4",
            r"C:\Users\et439\OneDrive\桌面\project\Rin\video\processed_train_video_right.mp4"
        ]
        
        # Collect and split data
        train_data_dir, val_data_dir = collect_and_split_data(model_path, video_paths)
        
        # Initialize model
        model = LSTMTracker()
        
        # Start training
        train_model(
            model=model,
            train_data_dir=train_data_dir,
            val_data_dir=val_data_dir,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            #added save_path & "models" file
            save_path=r"C:\Users\et439\OneDrive\桌面\project\Rin\models\best_lstm_model.pth"
        )
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except RuntimeError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")