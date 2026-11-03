import cv2
from ultralytics import YOLO
import time

CAMERA_URL = "http://10.174.97.26:8080/video"
CONFIDENCE = 0.35
EMPTY_SECONDS = 5

import os

MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolo11n.pt"
print(f"Loading AI model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_URL)

if not camera.isOpened():
    print("Camera connection failed. Check your phone camera server and Wi-Fi.")
    raise SystemExit(1)

print("AI shelf monitoring started. Press Q to quit.")

empty_since = None
alert_active = False

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame. Check the phone camera connection.")
        break

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    detections = results[0].boxes
    annotated_frame = results[0].plot()

    if detections is None or len(detections) == 0:
        if empty_since is None:
            empty_since = time.time()

        if time.time() - empty_since >= EMPTY_SECONDS:
            cv2.putText(
                annotated_frame,
                "ALERT: NO OBJECTS DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            if not alert_active:
                print("ALERT: No objects detected in the camera view.")
                alert_active = True
    else:
        empty_since = None
        alert_active = False

    cv2.imshow("Smart Shelf - Monitoring", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
