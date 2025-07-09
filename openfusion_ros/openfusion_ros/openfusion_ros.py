import rclpy
from vlm_base.vlm_base import VLMBaseLifecycleNode
from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET
from openfusion_ros.robot import Robot
from openfusion_ros.utils.opencv import show_image

class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')

    def load_model(self):
        self.get_logger().debug(f"{YELLOW}{BOLD}Loading model... {RESET}")
        self.get_logger().debug(f"{BLUE}{BOLD}Model loaded successfully.{RESET}")

    def load_robot(self):
        self.robot = Robot(self)

    def pcl_timer_callback(self):
        pass

    def append_pose_timer_callback(self):
        pass
