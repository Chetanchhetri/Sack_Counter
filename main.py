import cv2
from ultralytics import YOLO, solutions

# 1. Load weights and VERIFY
model_path = "best.pt" 
model = YOLO(model_path) 
if model.names[0] != 'sack':
    print("WARNING: Class 0 is NOT 'sack'. Check your weights file!")

# 2. Open video
video_path = "test_1.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                        cv2.CAP_PROP_FRAME_HEIGHT, 
                                        cv2.CAP_PROP_FPS))

# 3. Define a ZONE instead of a line for higher accuracy
# This creates a 'box' in the middle of the doorway
counting_region = [(50, 100), (950, 100), (950, 270), (50, 270)]

video_writer = cv2.VideoWriter("sack_counting_final.avi",
                       cv2.VideoWriter_fourcc(*"MJPG"),
                       fps, (w, h))

# 4. Initialize Object Counter with Sensitivity Tweaks
counter = solutions.ObjectCounter(
    show=True,
    region=counting_region,
    model=model_path,
    classes=[0],
    line_width=3,
    conf=0.1,              # Increased slightly to avoid false 'human' detections
    tracker="bytetrack.yaml" # Better for objects partially hidden by people
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Process frame
    results = counter(frame)
    processed_frame = results.plot_im 

    # Extract numerical counts from the results object
    total_sacks = results.in_count + results.out_count
    print(f"Sacks Detected: {total_sacks}", end="\r")

    video_writer.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
