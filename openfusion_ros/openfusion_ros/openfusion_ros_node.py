import rclpy
from openfusion_ros.openfusion_ros import OpenFusionNode

def main(args=None):
    rclpy.init(args=args)
    node = OpenFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

