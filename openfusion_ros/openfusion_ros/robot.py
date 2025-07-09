import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from tf_transformations import translation_from_matrix, quaternion_from_matrix

from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET, RED
from openfusion_ros.utils.conversions import transform_to_matrix
from openfusion_ros.camera import Camera

class Robot:
    def __init__(self, node):
        self.node = node
        self.class_name = self.__class__.__name__
        self.current_pose = None
        self.camera = Camera(node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.pose_pub = None
        self.pose_timer = None
        self.publish_interval = 0.2  # seconds

    def on_configure(self):
        self.node.get_logger().debug(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        self.node.declare_parameter("parent_frame", "map")
        self.node.declare_parameter("child_frame", "camera")
        self.node.declare_parameter("pose_topic", "robot_pose")

        self.parent_frame = self.node.get_parameter("parent_frame").get_parameter_value().string_value
        self.child_frame = self.node.get_parameter("child_frame").get_parameter_value().string_value
        self.pose_topic = self.node.get_parameter("pose_topic").get_parameter_value().string_value

        self.node.get_logger().debug(f"{BLUE}{BOLD}Finished configuring {self.class_name}{RESET}")
        self.camera.on_configure()

    def on_activate(self):
        self.node.get_logger().debug(f"{YELLOW}{BOLD}Activating {self.class_name}...{RESET}")
        self.camera.on_activate()

        self.pose_pub = self.node.create_publisher(PoseStamped, self.pose_topic, 10)
        self.pose_timer = self.node.create_timer(self.publish_interval, self.publish_pose)

        self.node.get_logger().debug(f"{BLUE}{self.class_name} activated.{RESET}")

    def on_deactivate(self):
        self.node.get_logger().debug(f"{YELLOW}Deactivating {self.class_name}...{RESET}")
        self.camera.on_deactivate()

        if self.pose_timer:
            self.pose_timer.cancel()
            self.pose_timer = None

    def on_cleanup(self):
        self.node.get_logger().debug(f"{BLUE}Cleaning up {self.class_name}...{RESET}")
        self.camera.on_cleanup()
        self.current_pose = None
        self.parent_frame = None
        self.child_frame = None
        self.pose_pub = None
        self.pose_timer = None

    def on_shutdown(self):
        self.node.get_logger().debug(f"{RED}{BOLD}Shutting down {self.class_name}...{RESET}")
        self.camera.on_shutdown()

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, now)
            return transform_to_matrix(transform)
        except Exception as e:
            self.node.get_logger().warn(f"Transform not available: {e}")
            return None

    def publish_pose(self):
        pose_matrix = self.get_pose()
        if pose_matrix is None:
            return

        translation = translation_from_matrix(pose_matrix)
        quaternion = quaternion_from_matrix(pose_matrix)

        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.pose.position.x = translation[0]
        msg.pose.position.y = translation[1]
        msg.pose.position.z = translation[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(msg)