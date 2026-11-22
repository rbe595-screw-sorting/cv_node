# node for classifying screws based on attributes like L/W ratio and average brightness.

import rclpy
from rclpy.node import Node

class DBSCAN(Node):
    def __init__(self):
        super().__init__('dbscan')

        # screws message type
        self.create_subscription()

def main(args=None):
    rclpy.init(args=args)
    node = DBSCAN()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()