from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf_transformations import translation_from_matrix, quaternion_from_matrix
from sensor_msgs.msg import CameraInfo
import numpy as np
from collections import deque
from rclpy.time import Duration, Time

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


class Camera:
    def __init__(self, node):
        self.class_name = self.__class__.__name__
        self.node = node
        self.bridge = CvBridge()
        self.rgb_sub = None
        self.depth_sub = None
        self.rgb_topic = None
        self.depth_topic = None
        self.debug_images = False
        self.camera_infos = None
        self.rgb_buffer = None
        self.depth_buffer = None
        self.max_age = None

    def on_configure(self):
        self.node.get_logger().debug(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        # Declare parameters for frame names
        if not self.node.has_parameter("max_buffer_size"):
            self.node.declare_parameter("max_buffer_size", 50)
        if not self.node.has_parameter("max_buffer_age_sec"):
            self.node.declare_parameter("max_buffer_age_sec", 2.0)
        if not self.node.has_parameter("rgb_topic"):
            self.node.declare_parameter("rgb_topic", "rgb")
        if not self.node.has_parameter("depth_topic"):
            self.node.declare_parameter("depth_topic", "depth")
        if not self.node.has_parameter("debug_images"):
            self.node.declare_parameter("debug_images", False)

        # Get parameter values
        self.rgb_topic = self.node.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.node.get_parameter("depth_topic").get_parameter_value().string_value
        self.debug_images = self.node.get_parameter("debug_images").get_parameter_value().bool_value
        self.max_buffer_size = self.node.get_parameter("max_buffer_size").get_parameter_value().integer_value
        max_buffer_age_sec = self.node.get_parameter("max_buffer_age_sec").get_parameter_value().double_value

        self.rgb_buffer = deque(maxlen=self.max_buffer_size)
        self.depth_buffer = deque(maxlen=self.max_buffer_size)
        self.max_age = Duration(seconds=max_buffer_age_sec)

        self.node.get_logger().debug(f"{BLUE}{BOLD}Finished configuring {self.class_name}{RESET}")


    def on_activate(self):
        self.node.get_logger().debug(f"{YELLOW}{BOLD}Activating {self.class_name}...{RESET}")

        # Subscribers
        self.rgb_sub = self.node.create_subscription(Image, '/rgb', self.rgb_callback, 10)
        self.depth_sub = self.node.create_subscription(Image, '/depth', self.depth_callback, 10)

        self.node.get_logger().debug(f"{BLUE}{self.class_name} activated.{RESET}")

    def on_deactivate(self):
        self.node.get_logger().debug(f"{YELLOW}Deactivating {self.class_name}...{RESET}")
        self.rgb_sub = None
        self.depth_sub = None
        self.node.get_logger().debug(f"{YELLOW}{self.class_name} subscriptions deactivated.{RESET}")

    def on_cleanup(self):
        self.node.get_logger().debug(f"{BLUE}Cleaning up {self.class_name}...{RESET}")
        self.rgb_sub = None
        self.depth_sub = None
        self.rgb_topic = None
        self.depth_topic = None
        self.debug_images = False
        self.camera_infos = None
        self.rgb_buffer = None
        self.depth_buffer = None
        self.max_age = None

    def on_shutdown(self):
        self.node.get_logger().debug(f"{RED}{BOLD}Shutting down {self.class_name}...{RESET}")
        self.on_cleanup() 

    def rgb_callback(self, msg: Image):
        self._prune_old(self.rgb_buffer)
        self.rgb_buffer.append(msg)

        if self.debug_images:
            image = self.rgb_buffer[0] if len(self.rgb_buffer) > 0 else msg
            show_ros_image(image, "RGB Image")
        
    def depth_callback(self, msg: Image):
        self._prune_old(self.depth_buffer)
        self.depth_buffer.append(msg)

        if self.debug_images:
            image = self.depth_buffer[0] if len(self.depth_buffer) > 0 else msg
            show_ros_depth_image(image, "Depth Image")

    def _prune_old(self, buffer):
        now = self.node.get_clock().now()
        while buffer and (now - Time.from_msg(buffer[0].header.stamp)) > self.max_age:
            buffer.popleft()

    def get_rgb(self, which='latest'):
        if which == 'latest':
            return self.rgb_buffer[-1] if self.rgb_buffer else None
        elif which == 'oldest':
            return self.rgb_buffer[0] if self.rgb_buffer else None
    
    def get_depth(self, which='latest'):
        if which == 'latest':
            return self.depth_buffer[-1] if self.depth_buffer else None
        elif which == 'oldest':
            return self.depth_buffer[0] if self.depth_buffer else None
    
    def get_synced_pair(self, tolerance_sec=0.02):
        for rgb in reversed(self.rgb_buffer):
            rgb_time = Time.from_msg(rgb.header.stamp).nanoseconds * 1e-9
            for depth in reversed(self.depth_buffer):
                depth_time = Time.from_msg(depth.header.stamp).nanoseconds * 1e-9
                if abs(rgb_time - depth_time) < tolerance_sec:
                    return rgb, depth
        return None, None