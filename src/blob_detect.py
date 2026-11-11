import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSPresetProfiles
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField

class BlobDetect(Node):
    def __init__(self):
        super().__init__('cv_blob')

        qos = QoSPresetProfiles.SENSOR_DATA

        # subscriber here
        self.create_subscription(Image, '/img_raw', self.img_callback, qos_profile=qos)

        # publish list of centroids of blobs
        self.locations = self.create_publisher(Float64MultiArray, '/centroids', 10)
        self.annotated_img = self.create_publisher(Image, '/annotated_img', 10)

        self.image = None
        self.cv_msg = None

        self.bridge = CvBridge()

        # OpenCV params (later make into config file)
        params = cv2.SimpleBlobDetector.Params()

        params.minThreshold = 50
        params.maxThreshold = 200
        params.thresholdStep = 10

        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 10000

        params.filterByInertia = True
        params.minInertiaRatio = 0.02
        params.maxInertiaRatio = 0.35

        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByColor = False

        self.detector = cv2.SimpleBlobDetector.create(params)

        self.threshold = 120
        self.min_length = 30


    def img_callback(self, msg):
        self.image = msg
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        annotated, centroids = self.process_image(cv_image)

        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        image_msg.header = msg.header
        self.annotated_img.publish(image_msg)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        points = [(float(x), float(y), 0.0) for x, y in centroids]
        cloud = point_cloud2.create_cloud(msg.header, fields, points)
        self.locations.publish(cloud)

    def process_image(self, image):

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        gray = img.copy()

        keypoints = self.detector.detect(gray)
        result = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        number = 1
        centroids = []

        for contour in contours:
            points = contour.reshape(-1, 2).astype(np.float32)

            centroid = tuple(np.mean(points, axis=0).astype(int))

            
            mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
            center = mean[0]
            direction = eigenvectors[0]

            projections = (points - center) @ direction
            p1 = tuple(points[np.argmin(projections)].astype(int))
            p2 = tuple(points[np.argmax(projections)].astype(int))

            # filter by length
            length_pixels = np.linalg.norm(np.array(p2, dtype=float) - np.array(p1, dtype=float))
            if length_pixels < self.min_length:
                continue

            # append to centroid list post filtering
            centroids.append(centroid)

            # i think you can write to images and then publish them?
            cv2.putText(output, f"{length_pixels:.1f}px", tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.drawContours(output, [contour], -1, (255, 255, 255), 1)

            cv2.putText(output, f"screw #{number}", tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"screw #{number}: {centroid}")

            cv2.drawMarker(output, centroid, (0, 255, 0),  markerType=cv2.MARKER_CROSS, markerSize=12,thickness=2)
            cv2.line(output, p1, p2, (0, 0, 255), 2)  # red in OpenCV BGR

            number += 1

        return output, centroids

    def pixel_to_camera(self):
        '''coming soon!'''
        pass


def main(args=None):
    rclpy.init(args=args)
    node = BlobDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()