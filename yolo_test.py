from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# load model
model = YOLO("./train_results/weights/best.pt")

# read video file
cap = cv2.VideoCapture("./video/processed_video.mp4")

if cap is None or not cap.isOpened():
    print("Error: Could not open video.")
    exit()

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_height = 1080
output_width = int((output_height / original_height) * original_width)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./video/yolo_result.mp4', fourcc, fps, (output_width, output_height))
  
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        resized_frame = cv2.resize(frame, (output_width, output_height))
        # Run YOLO inference on the frame
        results = model(resized_frame, conf=0.6, iou=0.3)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()