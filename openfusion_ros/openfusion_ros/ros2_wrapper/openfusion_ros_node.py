import rclpy
from openfusion_ros.ros2_wrapper.openfusion_ros import OpenFusionNode

def main(args=None):
    rclpy.init(args=args)
    node = OpenFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()