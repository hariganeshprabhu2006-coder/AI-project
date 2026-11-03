import cv2
from ultralytics import YOLO

CAMERA_URL = "http://10.174.97.26:8080/video"

print("Loading AI model...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(CAMERA_URL)

if not camera.isOpened():
    print("Camera connection failed. Check your phone camera server and Wi-Fi.")
    raise SystemExit(1)

print("AI camera started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    results = model.predict(frame, conf=0.35, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Smart Shelf - AI Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
