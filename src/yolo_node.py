import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    BoundingBox2D,
    ObjectHypothesisWithPose,
    ObjectHypothesis
)

from geometry_msgs.msg import PoseWithCovariance
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO

class YoloV8Node(Node):

    def __init__(self):
        super().__init__('yolov8_node')

        self.declare_parameter('model_path', '/home/kymadogg/')
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_image_topic', '/yolo/detected_image')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('confidence', 0.5)

        model_path = self.get_parameter('model_path').value
        input_topic = self.get_parameter('input_topic').value
        output_image_topic = self.get_parameter('output_image_topic').value
        detections_topic = self.get_parameter('detections_topic').value
        self.conf_thres = self.get_parameter('confidence').value

        # if amdgpu set env variable that makes it work to 0

        try:
            torch.zeros(1).cuda()
            self.device = 'cuda'
        except Exception:
            self.device = 'cpu'
        self.get_logger().info(f'Loading YOLOv8 model: {model_path} (device: {self.device})')
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.bridge = CvBridge()

        qos = QoSProfile( # need to change later...
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.img_pub = self.create_publisher(Image, output_image_topic, qos)
        self.det_pub = self.create_publisher(Detection2DArray, detections_topic, qos)

        self.sub = self.create_subscription(Image, input_topic, self.image_callback, qos)

        self.get_logger().info('YOLOv8 ROS 2 node started and subscribed to 'f'{input_topic}')

    def image_callback(self, img_msg: Image):
        """Main callback: run inference and publish results."""
        try:
            # Convert ROS Image → OpenCV (BGR8 is the standard for cameras)
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return
        
        results = self.model.track(
            cv_image,
            conf=self.conf_thres,
            persist=True,
            device=self.device
        )
        result = results[0]

        annotated_cv = result.plot()          
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_cv, encoding='bgr8')
        annotated_msg.header = img_msg.header
        self.img_pub.publish(annotated_msg)

        det_array = Detection2DArray()
        det_array.header = img_msg.header
        
        # tracking ids (if enabled)
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().tolist()
        else:
            track_ids = [None] * len(result.boxes)

        img_h, img_w = cv_image.shape[:2]

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            confidence = float(box.conf[0].item())
            class_idx = int(box.cls[0].item())
            class_name = result.names[class_idx]
            track_id = track_ids[i] if track_ids[i] is not None else -1
            # BoundingBox2D (normalized to 0-1)
            bbox = BoundingBox2D()
            bbox.center.position.x = (x1 + x2) / 2.0 / img_w
            bbox.center.position.y = (y1 + y2) / 2.0 / img_h
            bbox.center.theta = 0.0
            bbox.size_x = (x2 - x1) / img_w
            bbox.size_y = (y2 - y1) / img_h

            hyp = ObjectHypothesis()
            hyp.class_id = f"{class_name}_ID{int(track_id):04d}" # include track ID, not sure if this is correct...
            hyp.score = confidence

            hyp_pose = ObjectHypothesisWithPose()
            hyp_pose.hypothesis = hyp
            hyp_pose.pose = PoseWithCovariance()

            # Detection2D
            detection = Detection2D()
            detection.header = img_msg.header
            detection.bbox = bbox
            detection.results = [hyp_pose]

            det_array.detections.append(detection)

        self.det_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down YOLOv8 node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()