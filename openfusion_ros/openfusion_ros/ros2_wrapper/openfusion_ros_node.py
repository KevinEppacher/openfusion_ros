import rclpy
from openfusion_ros.ros2_wrapper.openfusion_ros import OpenFusionNode

def main(args=None):
    rclpy.init(args=args)
    node = OpenFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
