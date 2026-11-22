import os
import cv2
import numpy as np

import rclpy
from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class Detection(Node):
    def __init__(self):
        super().__init__('cv_detections')

        # TF buffer setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # TEMPORARY 
        self.screw_world_pts = self.create_publisher(PointStamped, '/screw_location', 10)
        self.create_timer(1.0, self.process_image)

        # openCV thresholding function params
        self.declare_parameter('threshold', 80)
        self.declare_parameter('min_length', 20)
        self.declare_parameter("max_length", 1000)
        self.declare_parameter('min_area', 500)
        self.declare_parameter('duplicate_iou', 0.50)
        self.declare_parameter('duplicate_distance', 20)

        # YOLO model parameters
        self.declare_parameter('model_path', 'weights/best.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('inference_dev', 'amd') # can be amd, nvidia, cpu

        # subscribe to camera image feed (color from realsense)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.get_latest_frame, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.store_camera_info, 10)
        self.br = CvBridge()

        # member variables for global access
        self.color_img = None
        self.gray_img = None
        self.latest_img_msg = None
        self.cam_info = None
        self.depth = 0.1142

        # read variables
        self.weights = self.get_parameter("weights").value
        self.threshold = self.get_parameter("threshold").value
        self.min_length = self.get_parameter("min_length").value
        self.max_length = self.get_parameter("max_length").value
        self.min_area = self.get_parameter("min_area").value
        self.duplicate_iou = self.get_parameter("duplicate_iou").value
        self.duplicate_distance = self.get_parameter("duplicate_distance").value
        self.conf_thres = self.get_parameter('confidence').value
    
        # YOLO model setup
        self.device = self.get_parameter('inference_dev').value
        if self.device == 'amd':
            # set amdgpu variable
            os.environ["HIP_VISIBLE_DEVICES"] = "0" 
            self.device = "cuda:0"
        elif self.device == "nvidia":
            self.device = "cuda"
        elif self.device == "cpu":
            self.device = "cpu"
        else:
            self.get_logger().error(f"Device {self.device} does not exist. Use ")
        
        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Loading YOLOv8 model: {model_path} (device: {self.device})')

        self.model = YOLO(model_path)
        self.model.to(self.device)



    def get_latest_frame(self, msg:Image):
        '''get the latest frame from the image message publisher and save as openCV member variable'''
        self.latest_img_msg = msg
        self.color_img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.gray_img = self.br.imgmsg_to_cv2(msg, desired_encoding='gray') # potentially wrong encoding code

    def get_latest_info(self, msg:CameraInfo):
        ''''''
        self.cam_info = msg

    # image processing helper functions
    def intersection_over_union(box_a, box_b):
        '''calculates intersection over union of two bboxes'''
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection

        return intersection / union if union else 0.0


    def bbox_center(self, bbox):
        '''returns the center of a bounding box'''
        x, y, width, height = bbox

        return np.array(
            [x + width / 2, y + height / 2],
            dtype=float
    )


    def is_duplicate(self, candidate, existing):
        '''determines if there are duplicate detections'''

        iou = self.intersection_over_union(candidate["bbox"], existing["bbox"])
        center_distance = np.linalg.norm(self.bbox_center(candidate["bbox"]) - self.bbox_center(existing["bbox"]))

        return (iou > self.duplicate_iou or center_distance < self.duplicate_distance)


    def create_mask(self, gray, polarity):
        '''creates a mask with changable polarity'''
        threshold_type = (cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY)

        _, mask = cv2.threshold(gray, self.threshold, 255, threshold_type )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    def create_watershed_markers(self, mask, polarity):
        seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        seed_mask = cv2.erode(mask, seed_kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE (5, 5)))

        number_of_seeds, seed_markers = cv2.connectedComponents(seed_mask, connectivity=8)

        markers = seed_markers.astype(np.int32) + 1
        markers[mask == 0] = 1

        seed_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for label in range(2, markers.max() + 1):
            pixels = np.argwhere(markers == label)

            if len(pixels) == 0:
                continue

            y, x = pixels.mean(axis=0).astype(int)

            cv2.putText(seed_overlay, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # make into ROS logging
        print(f"{polarity}: " f"{number_of_seeds - 1} watershed seeds")

        return markers

    def detect_blobs(self, mask, color_img, polarity):
        '''detect blobs in the image using watershed method from OpenCV'''

        markers = self.create_watershed_markers(mask, polarity)
        watershed_markers = markers.copy()
        cv2.watershed(color_img, watershed_markers)
        watershed_overlay = color_img.copy()
        watershed_overlay[watershed_markers == -1] = (0, 0, 255)

        detections = []

        for label in range(2, watershed_markers.max() + 1):
            blob_mask = np.uint8(watershed_markers == label) * 255

            if cv2.countNonZero(blob_mask) == 0:
                continue

            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            detections.append({
                "contour": contour,
                "bbox": (x, y, width, height),
                "mask": blob_mask
            })

        return detections

    def remove_duplicates(self, detections):
        unique_detections = []

        for candidate in detections:
            if not any(self.is_duplicate(candidate, existing) for existing in unique_detections):
                unique_detections.append(candidate)

        return unique_detections

    
    def verify_with_yolo(self, detections):
        '''verify blobs are screws by referencing YOLO bounding box location'''

        # run yolo model --> bounding boxes
        result = self.model(self.latest_image, conf=self.conf_thres, device=self.device, verbose=False)[0]

        # publish debug image
        yolo_debug_img = result.plot() 
        yolo_debug_msg = self.bridge.cv2_to_imgmsg(yolo_debug_img, encoding='bgr8')
        #yolo_debug_msg.header = img_msg.header

    
        # option 1: box overlap
        # for detection in detection
            # use intersection in union to see if boxes overlap
            # if overlap, then true

        # option 2: centroid of box and screw are close-ish
        # for detection in detection
                    # use intersection in union to see if boxes overlap
                    # if within distance, then true
        
        pass
    

    def calculate_metrics(self, detection, gray):
        contour = detection["contour"]
        blob_mask = detection["mask"]

        rotated_rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rotated_rect

        blob_length = max(width, height)
        blob_width = min(width, height)

        if blob_width == 0:
            return None

        if not self.min_length <= blob_length <= self.max_length:
            return None

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centroid = (cx, cy)

        brightness = cv2.mean(gray, mask=blob_mask)[0]

        return {
            "centroid": centroid,
            "length_width_ratio": blob_length / blob_width,
            "brightness": brightness
        }


    def pixel_to_world(self, u, v, depth_m, camera_frame="camera_color_optical_frame", world_frame="world"):

        # if this does not work, try self.cam_info.k
        K = np.asarray(self.cam_info.r, dtype=float)

        # back-project pixel (u, v) into the camera optical frame
        xyz_camera = depth_m * np.linalg.inv(K) @ np.array([u, v, 1.0])

        point = PointStamped()
        point.header.frame_id = camera_frame
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = float(xyz_camera[0])
        point.point.y = float(xyz_camera[1])
        point.point.z = float(xyz_camera[2])

        try:
            point_world = self.tf_buffer.transform(point, world_frame, timeout=Duration(seconds=1.0))
            # log pose?
            # return np.array([point_world.point.x, point_world.point.y, point_world.point.z])
            return point_world
        except Exception as exc:
            self.get_logger().error(f"TF transform failed: {exc}")
            return None
        

    def process_image(self):
        """main image processing functions service"""

        # get image and convert into gray and color images
        color_img = self.color_img
        gray = self.gray_img

        # arrays for ratios, brightnesses, and all detections
        ratios = []
        brightnesses = []
        all_detections = []

        # mask stuff, polarity 
        for polarity in ("dark", "white"):
            mask = self.create_mask(gray, polarity)
            detections = self.detect_blobs(mask, color_img, polarity)
            all_detections.extend(detections)

        detections = self.remove_duplicates(all_detections)
        output = np.zeros_like(color_img)
        
        number = 1

        for detection in detections:
            metrics = self.calculate_metrics(detection, gray, color_img)
    
            if metrics is None:
                continue
    
            ratios.append(metrics['length_width_ratio'])
            brightnesses.append(metrics['brightness'])
    
            contour = detection["contour"]
            centroid = metrics["centroid"]

            world_point = self.pixel_to_world(centroid[0], centroid[1],  self.depth, camera_frame="camera_color_optical_frame", world_frame="base_link")

            self.screw_world_pts.publish(world_point)
            self.get_logger().info("published screw pose")

            # logging or publisher...
            # print(
            #     f"screw #{number}: "
            #     f"center={centroid}, "
            #     f"L/W={metrics['length_width_ratio']:.2f}, "
            #     f"brightness={metrics['brightness']:.2f}"
            # )
    
            # cv2.drawContours(output, [contour], -1, (255, 255, 255), 1)
    
            # cv2.drawMarker(output, centroid, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            #number += 1
    

        # print(ratios)
        # print(brightnesses)

        
       # cv2.imwrite("watershed_result.png", output)

        
def main(args=None):
    rclpy.init(args=args)
    node = Detection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


