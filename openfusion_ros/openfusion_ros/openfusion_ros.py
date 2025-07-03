import rclpy
from vlm_base.vlm_base import VLMBaseLifecycleNode

class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')

    def load_model(self):
        self.get_logger().info("Loading OpenFusion model...")

    