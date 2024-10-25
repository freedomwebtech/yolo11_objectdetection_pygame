import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Initialize YOLO model
model = YOLO("yolo11s.pt")
names = model.model.names



# OpenCV VideoCapture (Use a video file or webcam)
cap = cv2.VideoCapture('vidp.mp4')






count = 0


while True:
    

    # OpenCV: Read frame from video
    ret, frame = cap.read()
    if not ret:
        # Reset to the start of the video if the video ends
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue  # Process every third frame

    frame = cv2.resize(frame, (1020, 600))

    # YOLO: Run tracking on the frame
    results = model.track(frame, persist=True, classes=0)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get YOLO detections (bounding boxes, class IDs, track IDs)
        boxes = results[0].boxes.xyxy.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Process each detected object
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            c = names[class_id]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) 
           

    # Show the original frame in OpenCV window with annotated bounding boxes and track ID
    cv2.imshow("RGB", frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

