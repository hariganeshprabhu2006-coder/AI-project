import cv2
from pathlib import Path

url = "http://10.174.97.26:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("ERROR: Camera connection failed")
    raise SystemExit(1)

ok, frame = cap.read()
cap.release()

if not ok:
    print("ERROR: Could not capture a frame")
else:
    path = Path("dataset/images/pepsodent/test.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(path), frame)
    print("Saved successfully:", saved)
    print("File exists:", path.exists())
    print("Image path:", path.resolve())
