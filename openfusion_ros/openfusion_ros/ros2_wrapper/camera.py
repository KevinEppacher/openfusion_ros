from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from openfusion_ros.utils import BLUE, YELLOW, BOLD, RESET
from openfusion_ros.utils.opencv import show_ros_image, show_ros_depth_image
import rclpy

class CamInfo:
    def __init__(self, msg: CameraInfo = None):
        self.msg = msg

    def get_intrinsics(self):
        if not self.msg:
            return None
        return np.array(self.msg.k, dtype=np.float64).reshape(3, 3)

    def get_size(self):
        if not self.msg:
            return (0, 0)
        return self.msg.width, self.msg.height

    def scale_to(self, new_width, new_height):
        """Return a new CameraInfo with scaled intrinsics."""
        if self.msg is None:
            return None

        scaled = CameraInfo()
        scaled.header = self.msg.header
        scaled.distortion_model = self.msg.distortion_model
        scaled.d = list(self.msg.d)

        scale_x = new_width / self.msg.width
        scale_y = new_height / self.msg.height

        K = np.array(self.msg.k, dtype=np.float64).reshape(3, 3)
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y
        scaled.k = K.flatten().tolist()

        P = np.array(self.msg.p, dtype=np.float64).reshape(3, 4)
        P[0, 0] *= scale_x
        P[1, 1] *= scale_y
        P[0, 2] *= scale_x
        P[1, 2] *= scale_y
        scaled.p = P.flatten().tolist()

        scaled.width = int(new_width)
        scaled.height = int(new_height)
        return scaled


class Camera:
    """Camera interface with integrated CameraInfo subscription and resizing."""

    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()

        # Declare parameters
        self.rgb_topic = self.declare_param("robot.camera.rgb_topic", "/rgb")
        self.depth_topic = self.declare_param("robot.camera.depth_topic", "/depth")
        self.debug_images = self.declare_param("robot.camera.debug_images", False)
        self.resize_width = self.declare_param("robot.camera.resize_width", 640)
        self.resize_height = self.declare_param("robot.camera.resize_height", 480)
        qos_reliability = self.declare_param("robot.camera.qos_reliability", "best_effort").lower()
        qos_history = self.declare_param("robot.camera.qos_history", "keep_last").lower()
        qos_depth = self.declare_param("robot.camera.qos_depth", 10)

        self.print_parameters()

        reliability = ReliabilityPolicy.RELIABLE if qos_reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        history = HistoryPolicy.KEEP_ALL if qos_history == "keep_all" else HistoryPolicy.KEEP_LAST
        self.qos_profile = QoSProfile(reliability=reliability, history=history, depth=qos_depth)

        # Subscribers
        self.rgb_msg = None
        self.depth_msg = None
        self.camera_info = CamInfo()
        self.camera_info_received = False

        self.rgb_sub = node.create_subscription(
            Image, self.rgb_topic, self.cb_rgb, self.qos_profile
        )
        self.depth_sub = node.create_subscription(
            Image, self.depth_topic, self.cb_depth, self.qos_profile
        )
        self.caminfo_sub = node.create_subscription(
            CameraInfo, "/camera_info", self.cb_cam_info, self.qos_profile
        )

        self.node.get_logger().info(f"Subscribed to {self.rgb_topic}, {self.depth_topic}, and /camera_info")

    # -----------------------------------------------------------------------
    def declare_param(self, name, default):
        if not self.node.has_parameter(name):
            self.node.declare_parameter(name, default)
        return self.node.get_parameter(name).value

    def print_parameters(self):
        self.node.get_logger().info(
            f"{BLUE}{BOLD}Camera parameters:{RESET}\n"
            f"  rgb_topic: {YELLOW}{self.rgb_topic}{RESET}\n"
            f"  depth_topic: {YELLOW}{self.depth_topic}{RESET}\n"
            f"  resize: {YELLOW}{self.resize_width}x{self.resize_height}{RESET}\n"
            f"  debug_images: {YELLOW}{self.debug_images}{RESET}\n"
        )

    # -----------------------------------------------------------------------
    def cb_cam_info(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.camera_info = CamInfo(msg)
            self.camera_info_received = True
            self.node.get_logger().info(f"Received CameraInfo {msg.width}x{msg.height}")

    def wait_for_camera_info(self, timeout=5.0):
        """Wait until a CameraInfo message is received."""
        import time
        start = time.time()
        while not self.camera_info_received and time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.camera_info_received

    # -----------------------------------------------------------------------
    def cb_rgb(self, msg: Image):
        self.rgb_msg = msg
        if self.debug_images:
            show_ros_image(msg, "RGB")

    def cb_depth(self, msg: Image):
        self.depth_msg = msg
        if self.debug_images:
            show_ros_depth_image(msg, "Depth")

    def get_rgb(self):
        return self.rgb_msg

    def get_depth(self):
        return self.depth_msg

    def get_resized_camera_info(self):
        if not self.camera_info.msg:
            return None
        return self.camera_info.scale_to(self.resize_width, self.resize_height)

    def get_intrinsics(self):
        info = self.get_resized_camera_info()
        if not info:
            return None
        return np.array(info.k).reshape(3, 3)

    def get_size(self):
        return self.resize_height, self.resize_width
