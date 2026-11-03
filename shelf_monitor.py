
import cv2

CAMERA_URL = "http://10.174.97.26:8080/video"


def main():
    print("Connecting to phone camera...")

    camera = cv2.VideoCapture(CAMERA_URL)

    if not camera.isOpened():
        print("ERROR: Cannot connect to phone camera.")
        print("Check that IP Webcam is running on your phone.")
        return

    print("Camera connected successfully.")
    print("Showing live camera without boxes.")
    print("Press Q to close the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        cv2.imshow("Smart Shelf - Live Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()
