from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf_transformations import translation_from_matrix, quaternion_from_matrix

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
        self.debug_images = False

    def on_configure(self):
        self.node.get_logger().debug(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        # Declare parameters for frame names
        self.node.declare_parameter("rgb_topic", "rgb")
        self.node.declare_parameter("depth_topic", "depth")
        self.node.declare_parameter("debug_images", False)

        # Get parameter values
        self.rgb_topic = self.node.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.node.get_parameter("depth_topic").get_parameter_value().string_value
        self.debug_images = self.node.get_parameter("debug_images").get_parameter_value().bool_value

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
        self.current_image = None
        self.current_depth = None
        self.rgb_topic = None
        self.depth_topic = None

    def on_shutdown(self):
        self.node.get_logger().debug(f"{RED}{BOLD}Shutting down {self.class_name}...{RESET}")
        self.on_cleanup() 

    def rgb_callback(self, msg: Image):
        if self.debug_images:
            show_ros_image(msg, "RGB Image")

    def depth_callback(self, msg: Image):
        if self.debug_images:
            show_ros_depth_image(msg, "Depth Image")