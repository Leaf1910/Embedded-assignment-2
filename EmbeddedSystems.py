import cv2
import torch  # PyTorch for YOLOv5

# Load YOLOv5 model (YOLOv5s is the small version, you can also try 'yolov5m' or 'yolov5l')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s' for fast inference

# Path to your video file
video_path = "Road.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_path = "OD_Video_Output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render results on the frame
    results.render()  # This will draw the bounding boxes and labels directly on the frame

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_path}")
