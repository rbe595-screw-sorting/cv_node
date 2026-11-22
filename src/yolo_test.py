import cv2
from ultralytics import YOLO

camera = "/dev/video4" # realsense color 
cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)

print("Exists:", __import__("os").path.exists(camera))
print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError(f"Could not open {camera}")

model = YOLO("weights/best.pt")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame")
            break

        result = model(frame, device="cuda:0", verbose=False)[0]
        cv2.imshow("YOLO RealSense color", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
