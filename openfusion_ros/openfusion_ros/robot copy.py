import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tf2_ros import Buffer, TransformListener

from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET, RED
from openfusion_ros.utils.conversions import transform_to_matrix

class Camera:
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.current_image = None
        self.current_depth = None
        self.rgb_stamp = None
        self.depth_stamp = None
        self.rgb_sub = None
        self.depth_sub = None

    def setup_subscribers(self):
        try:
            rgb_topic = self.node.get_parameter("rgb_topic").get_parameter_value().string_value
            depth_topic = self.node.get_parameter("depth_topic").get_parameter_value().string_value
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

            self.rgb_sub = self.node.create_subscription(Image, rgb_topic, self.rgb_callback, qos)
            self.depth_sub = self.node.create_subscription(Image, depth_topic, self.depth_callback, qos)

            self.node.get_logger().info(f"{BOLD}Camera initialized with topics: {rgb_topic}, {depth_topic}{RESET}")
            self.node.get_logger().info(f"{BOLD}Camera QoS: depth={qos.depth}, reliability={qos.reliability}{RESET}")
            self.node.get_logger().info(f"{BLUE}Camera subscribers initialized.{RESET}")
        except Exception as e:
            self.node.get_logger().error(f"Failed to set up camera subscribers: {e}")

    def teardown_subscribers(self):
        self.rgb_sub = None
        self.depth_sub = None
        self.node.get_logger().info(f"{YELLOW}Camera subscribers torn down.{RESET}")

    def rgb_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.rgb_stamp = msg.header.stamp
            self.node.get_logger().info(f"{BLUE}RGB image received at {self.rgb_stamp}.{RESET}")
        except Exception as e:
            self.node.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_img.dtype == np.float32:
                depth_img = (depth_img * 1000.0).astype(np.uint16)
            elif depth_img.dtype != np.uint16:
                self.node.get_logger().warn(f"Unexpected depth dtype: {depth_img.dtype}")
                return
            self.current_depth = np.ascontiguousarray(depth_img)
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.node.get_logger().error(f"Failed to convert depth image: {e}")

class Robot:
    def __init__(self, node):
        self.node = node
        self.current_pose = None
        self.camera = Camera(node)

        node.declare_parameter("parent_frame", "map")
        node.declare_parameter("child_frame", "camera")

        self.parent_frame = node.get_parameter("parent_frame").get_parameter_value().string_value
        self.child_frame = node.get_parameter("child_frame").get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def activate(self):
        self.camera.setup_subscribers()

    def deactivate(self):
        self.camera.teardown_subscribers()

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, now)
            return transform_to_matrix(transform)
        except Exception as e:
            self.node.get_logger().warn(f"Transform not available: {e}")
            return None
