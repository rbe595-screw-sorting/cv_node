import cv2
from ultralytics import YOLO
import os
import sys
import subprocess


os.environ["HIP_VISIBLE_DEVICES"] = "0"      # this process and its children only

subprocess.run([sys.executable, "-c", 
                "import os; print(os.environ['HIP_VISIBLE_DEVICES'])"],
                capture_output=True, text=True,
)

image_path = "imgs/screw_006.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

model = YOLO("weights/best.pt")
result = model(image, device="cuda:0", verbose=False)[0]

# bounding box locations
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    print(
        f"{class_name}: "
        f"x1={x1:.1f}, y1={y1:.1f}, "
        f"x2={x2:.1f}, y2={y2:.1f}, "
        f"confidence={confidence:.2f}"
    )


# --> for location determination
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    roi = image[y1:y2, x1:x2]



# --> for classifying what type (DBSCAN)
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    roi = image[y1:y2, x1:x2]


display = cv2.resize(result.plot(), (960, 540))
cv2.imshow("YOLO RealSense color", display)

cv2.waitKey(0)
cv2.destroyAllWindows()
