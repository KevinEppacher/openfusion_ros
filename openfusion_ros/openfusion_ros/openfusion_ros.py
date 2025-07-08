import rclpy
from vlm_base.vlm_base import VLMBaseLifecycleNode
from openfusion_ros.utils import BLUE, BOLD, RESET
from openfusion_ros.robot import Robot

class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')
        self.robot = Robot(self)
        self.timer = self.create_timer(0.1, self.timer_callback)


    def load_model(self):
        self.get_logger().info(f"{BLUE}{BOLD}Loading model...{RESET}")

    def timer_callback(self):
        self.robot.get_pose()