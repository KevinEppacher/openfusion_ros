import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tf2_ros import Buffer, TransformListener

from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET, RED
from openfusion_ros.utils.conversions import transform_to_matrix
from openfusion_ros.utils.opencv import show_ros_image, show_ros_depth_image

class Camera:
    def __init__(self, node):
        self.class_name = self.__class__.__name__
        self.node = node
        self.bridge = CvBridge()
        self.current_image = None
        self.current_depth = None
        self.rgb_sub = None
        self.depth_sub = None
        self.rgb_topic = None
        self.depth_topic = None

    def on_configure(self):
        self.node.get_logger().info(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        # Declare parameters for frame names
        self.node.declare_parameter("rgb_topic", "rgb")
        self.node.declare_parameter("depth_topic", "depth")

        # Get parameter values
        self.rgb_topic = self.node.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.node.get_parameter("depth_topic").get_parameter_value().string_value

        self.node.get_logger().info(f"{BLUE}{BOLD}Finished configuring {self.class_name}{RESET}")


    def on_activate(self):
        self.node.get_logger().info(f"{YELLOW}{BOLD}Activating {self.class_name}...{RESET}")

        # Subscribers
        self.rgb_sub = self.node.create_subscription(Image, '/rgb', self.rgb_callback, 10)
        self.depth_sub = self.node.create_subscription(Image, '/depth', self.depth_callback, 10)

        self.node.get_logger().info(f"{BLUE}{self.class_name} activated.{RESET}")

    def on_deactivate(self):
        self.node.get_logger().info(f"{YELLOW}Deactivating {self.class_name}...{RESET}")

    def on_cleanup(self):
        self.node.get_logger().info(f"{BLUE}Cleaning up {self.class_name}...{RESET}")

    def on_shutdown(self):
        self.node.get_logger().info(f"{RED}{BOLD}Shutting down {self.class_name}...{RESET}")

    def rgb_callback(self, msg: Image):
        show_ros_image(msg, "RGB Image")

    def depth_callback(self, msg: Image):
        show_ros_depth_image(msg, "Depth Image")

class Robot:
    def __init__(self, node):
        self.node = node
        self.class_name = self.__class__.__name__
        self.current_pose = None
        self.camera = Camera(node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def on_configure(self):
        self.node.get_logger().info(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        # Declare parameters for frame names
        self.node.declare_parameter("parent_frame", "map")
        self.node.declare_parameter("child_frame", "camera")

        # Get parameter values
        self.parent_frame = self.node.get_parameter("parent_frame").get_parameter_value().string_value
        self.child_frame = self.node.get_parameter("child_frame").get_parameter_value().string_value

        self.node.get_logger().info(f"{BLUE}{BOLD}Finished configuring {self.class_name}{RESET}")
        self.camera.on_configure()
        

    def on_activate(self):
        self.node.get_logger().info(f"{YELLOW}{BOLD}Activating {self.class_name}...{RESET}")
        self.camera.on_activate()
        self.node.get_logger().info(f"{BLUE}{self.class_name} activated.{RESET}")

    def on_deactivate(self):
        self.node.get_logger().info(f"{YELLOW}Deactivating {self.class_name}...{RESET}")
        pass

    def on_cleanup(self):
        self.node.get_logger().info(f"{BLUE}Cleaning up {self.class_name}...{RESET}")
        pass

    def on_shutdown(self):
        self.node.get_logger().info(f"{RED}{BOLD}Shutting down {self.class_name}...{RESET}")
        pass

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, now)
            return transform_to_matrix(transform)
        except Exception as e:
            self.node.get_logger().warn(f"Transform not available: {e}")
            return None
