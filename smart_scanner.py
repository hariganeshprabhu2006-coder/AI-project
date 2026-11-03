import cv2
import sqlite3
from pyzbar.pyzbar import decode

CAMERA_URL = "http://10.174.97.26:8080/video"

camera = cv2.VideoCapture(CAMERA_URL)

if not camera.isOpened():
    print("Camera connection failed.")
    raise SystemExit(1)

print("Smart Shelf Scanner started. Press Q to quit.")

last_barcode = None

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    for barcode in decode(frame):
        code = barcode.data.decode("utf-8")
        x, y, w, h = barcode.rect

        conn = sqlite3.connect("products.db")
        product = conn.execute(
            "SELECT name, price, expiry_date FROM products WHERE barcode = ?",
            (code,)
        ).fetchone()
        conn.close()

        if product:
            name, price, expiry = product
            label = f"{name} Rs.{price:.2f}"
            print(f"Product: {name} | Price: Rs.{price:.2f} | Expiry: {expiry or 'Not entered'}")
        else:
            label = f"Unknown product: {code}"
            print(f"Unregistered barcode: {code}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2
        )

    cv2.imshow("Smart Shelf Product Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
