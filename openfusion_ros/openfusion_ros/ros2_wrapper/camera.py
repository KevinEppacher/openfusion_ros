from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf_transformations import translation_from_matrix, quaternion_from_matrix
from sensor_msgs.msg import CameraInfo
import numpy as np
from collections import deque
from rclpy.time import Duration, Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET, RED
from openfusion_ros.utils.conversions import transform_to_matrix
from openfusion_ros.utils.opencv import show_ros_image, show_ros_depth_image

class CamInfo:
    def __init__(self, cam_info_msg: CameraInfo = None):
        self.cam_info_msg = cam_info_msg

    def get_intrinsics(self):
        if not self.cam_info_msg:
            return None
        return np.array(self.cam_info_msg.k, dtype=np.float64).reshape(3, 3)

    def get_size(self):
        if not self.cam_info_msg:
            return (0, 0)
        return self.cam_info_msg.width, self.cam_info_msg.height
    
    def get_horizontal_fov_deg(self):
        if not self.cam_info_msg:
            return 70.0  # fallback
        fx = self.cam_info_msg.k[0]
        width = self.cam_info_msg.width
        fov_rad = 2 * np.arctan2(width, 2 * fx)
        return np.degrees(fov_rad)

class Camera:
    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.rgb_topic = None
        self.depth_topic = None
        self.debug_images = False

        self.rgb_msg = None
        self.depth_msg = None

    def on_configure(self):
        self.node.get_logger().debug(f"{BLUE}{BOLD}Configuring Camera...{RESET}")

        # Parameters
        self.node.declare_parameter("robot.camera.rgb_topic", "/rgb")
        self.node.declare_parameter("robot.camera.depth_topic", "/depth")
        self.node.declare_parameter("robot.camera.debug_images", False)

        self.node.declare_parameter("robot.camera.qos_reliability", "best_effort")
        self.node.declare_parameter("robot.camera.qos_history", "keep_last")
        self.node.declare_parameter("robot.camera.qos_depth", 10)

        # Get values
        self.rgb_topic = self.node.get_parameter("robot.camera.rgb_topic").value
        self.depth_topic = self.node.get_parameter("robot.camera.depth_topic").value
        self.debug_images = self.node.get_parameter("robot.camera.debug_images").value

        qos_reliability = self.node.get_parameter("robot.camera.qos_reliability").value.lower()
        qos_history = self.node.get_parameter("robot.camera.qos_history").value.lower()
        qos_depth = self.node.get_parameter("robot.camera.qos_depth").value

        reliability = (
            ReliabilityPolicy.RELIABLE if qos_reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        history = (
            HistoryPolicy.KEEP_ALL if qos_history == "keep_all"
            else HistoryPolicy.KEEP_LAST
        )

        self.qos_profile = QoSProfile(
            reliability=reliability,
            history=history,
            depth=qos_depth,
        )

        self.node.get_logger().info(
            f"Camera configured with QoS(reliability={qos_reliability}, history={qos_history}, depth={qos_depth})"
        )

    def on_activate(self):
        self.node.get_logger().debug(f"{YELLOW}{BOLD}Activating Camera...{RESET}")

        self.rgb_sub = self.node.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, self.qos_profile
        )
        self.depth_sub = self.node.create_subscription(
            Image, self.depth_topic, self.depth_callback, self.qos_profile
        )

        self.node.get_logger().info(
            f"Subscribed to {self.rgb_topic} and {self.depth_topic}"
        )

    def on_deactivate(self):
        self.node.get_logger().debug(f"{YELLOW}Deactivating Camera...{RESET}")
        self.rgb_sub = None
        self.depth_sub = None

    def on_cleanup(self):
        self.node.get_logger().debug(f"{BLUE}Cleaning up Camera...{RESET}")
        self.rgb_msg = None
        self.depth_msg = None

    def on_shutdown(self):
        self.node.get_logger().debug(f"{RED}{BOLD}Shutting down Camera...{RESET}")
        self.on_cleanup()

    def rgb_callback(self, msg: Image):
        self.rgb_msg = msg
        if self.debug_images:
            show_ros_image(msg, "RGB")

    def depth_callback(self, msg: Image):
        self.depth_msg = msg
        if self.debug_images:
            show_ros_depth_image(msg, "Depth")

    def get_rgb(self):
        return self.rgb_msg

    def get_depth(self):
        return self.depth_msg