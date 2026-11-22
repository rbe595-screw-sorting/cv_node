# cv_node
ROS 2 Node for detecting objects using blob detection and transforming the coordinates into robot frame and possibly classifying them as well.

## Setup
This project uses uv, use this command to install on linux:

```bash
ros2 run cv_node detection.py
```


## For AMD GPUs
make sure the flag `HIP_VISIBLE_DEVICES=0` is set before running anything.

### Notes
using this model for now:
```latex
@misc{ screws-pctto_dataset,
  title = { screws Dataset },
  type = { Open Source Dataset },
  author = { vinti },
  howpublished = { \url{ https://universe.roboflow.com/vinti/screws-pctto } },
  url = { https://universe.roboflow.com/vinti/screws-pctto },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { jul },
  note = { visited on 2026-11-16 },
}
```

(kylie): previous main loop from before:

```python

def main():
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

    ratios = []
    brightnesses = []

    if gray is None or color_img is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    all_detections = []

    for polarity in ("dark", "white"):
        mask = create_mask(gray, polarity)
        detections = detect_blobs(mask, color_img, polarity)
        all_detections.extend(detections)

    detections = remove_duplicates(all_detections)

    output = np.zeros_like(color_img)

    number = 1

    for detection in detections:
        metrics = calculate_metrics(detection, gray, color_img,)

        if metrics is None:
            continue

        ratios.append(metrics['length_width_ratio'])
        brightnesses.append(metrics['brightness'])

        contour = detection["contour"]
        centroid = metrics["centroid"]

        # logging or publisher
        print(
            f"screw #{number}: "
            f"center={centroid}, "
            f"L/W={metrics['length_width_ratio']:.2f}, "
            f"brightness={metrics['brightness']:.2f}"
        )

        cv2.drawContours(output, [contour], -1, (255, 255, 255), 1)

        cv2.drawMarker(output, centroid, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        number += 1

    print(ratios)
    print(brightnesses)

    cv2.imwrite("watershed_result.png", output)

```