import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf_transformations import translation_from_matrix, quaternion_from_matrix
from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET
from openfusion_ros.utils.conversions import transform_to_matrix, convert_stamp_to_sec
from openfusion_ros.utils.opencv import ros_image_2_opencv
from openfusion_ros.ros2_wrapper.camera import Camera

class Robot:
    """Robot interface that publishes poses and fetches TF transforms without lifecycle hooks."""

    def __init__(self, node):
        self.node = node
        self.class_name = self.__class__.__name__

        # --- Parameters ----------------------------------------------------
        self.parent_frame = self.declare_param("robot.parent_frame", "map")
        self.child_frame = self.declare_param("robot.child_frame", "camera_link")
        self.pose_topic = self.declare_param("robot.pose_topic", "robot_pose")
        self.max_delta_time = self.declare_param("robot.max_delta_time", 0.05)
        self.pose_source = self.declare_param("robot.pose_source", "tf")
        self.print_parameters()

        # --- Camera and TF setup -------------------------------------------
        self.camera = Camera(node)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=True)

        # --- Pose I/O ------------------------------------------------------
        self.pose_pub = self.node.create_publisher(PoseStamped, self.pose_topic + "/debug", 10)
        self.pose_sub = None
        self.pose_timer = self.node.create_timer(0.05, self.publish_pose)
        self.current_pose_msg = None
        self.transform = None

        # --- Optional subscription if external pose topic is used ----------
        if self.pose_source == "topic":
            self.pose_sub = self.node.create_subscription(
                TransformStamped, self.pose_topic, self.pose_callback, 10
            )
            self.node.get_logger().info(f"Subscribed to pose topic: {self.pose_topic}")
        else:
            self.node.get_logger().info(f"Using TF lookup: {self.parent_frame} → {self.child_frame}")

        self.node.get_logger().info(f"{BLUE}Robot initialized ({self.pose_source} mode).{RESET}")

    # -----------------------------------------------------------------------
    def declare_param(self, name, default):
        if not self.node.has_parameter(name):
            self.node.declare_parameter(name, default)
        return self.node.get_parameter(name).value

    # -----------------------------------------------------------------------
    def update_pose(self):
        """Updates self.transform either from topic or TF buffer."""
        if self.pose_source == "topic":
            if self.current_pose_msg is None:
                self.node.get_logger().debug("No pose received yet from topic.")
                return None
            self.transform = self.current_pose_msg
            return

        try:
            rgb = self.camera.get_rgb()
            time_target = rclpy.time.Time.from_msg(rgb.header.stamp) if rgb else rclpy.time.Time()
            self.transform = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, time_target)
        except Exception as e:
            self.node.get_logger().warn(f"Transform not available: {e}")
            self.transform = None

    # -----------------------------------------------------------------------
    def get_pose(self, datatype="matrix"):
        """Returns pose as transform matrix or raw TF message."""
        self.update_pose()
        if self.transform is None:
            self.node.get_logger().debug("Transform is None. Cannot get pose.")
            return None

        if datatype == "matrix":
            return transform_to_matrix(self.transform)
        elif datatype == "tf2":
            return self.transform
        else:
            self.node.get_logger().warn(f"Unknown datatype '{datatype}' requested.")
            return None

    # -----------------------------------------------------------------------
    def get_openfusion_input(self):
        """Fetch synchronized (pose, rgb, depth) for fusion."""
        pose = self.get_pose(datatype="tf2")
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()

        if pose is None or rgb is None or depth is None:
            reason = "pose" if pose is None else "RGB" if rgb is None else "depth"
            self.node.get_logger().warn(f"Skipping frame: {reason} is None.")
            return None

        rgb_time = convert_stamp_to_sec(rgb.header.stamp)
        depth_time = convert_stamp_to_sec(depth.header.stamp)
        pose_time = convert_stamp_to_sec(pose.header.stamp)
        max_diff = max(abs(rgb_time - depth_time),
                       abs(rgb_time - pose_time),
                       abs(depth_time - pose_time))
        if max_diff > self.max_delta_time:
            self.node.get_logger().warn(f"Timestamps not synchronized (Δ={max_diff:.3f}s).")
            return None

        converted_pose = transform_to_matrix(pose)
        converted_rgb = ros_image_2_opencv(rgb, "bgr8", self.node.get_logger())
        converted_depth = ros_image_2_opencv(depth, "passthrough", self.node.get_logger())
        return converted_pose, converted_rgb, converted_depth

    # -----------------------------------------------------------------------
    def publish_pose(self):
        """Publishes current TF pose as PoseStamped for debug."""
        pose_matrix = self.get_pose("matrix")
        if pose_matrix is None:
            return

        translation = translation_from_matrix(pose_matrix)
        quaternion = quaternion_from_matrix(pose_matrix)
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = translation
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quaternion
        self.pose_pub.publish(msg)

    # -----------------------------------------------------------------------
    def pose_callback(self, msg: PoseStamped):
        self.current_pose_msg = msg

    def print_parameters(self):
        self.node.get_logger().info(
            f"{BLUE}{BOLD}{self.class_name} parameters:{RESET}\n"
            f"  parent_frame: {YELLOW}{self.parent_frame}{RESET}\n"
            f"  child_frame: {YELLOW}{self.child_frame}{RESET}\n"
            f"  pose_topic: {YELLOW}{self.pose_topic}{RESET}\n"
            f"  max_delta_time: {YELLOW}{self.max_delta_time}{RESET}\n"
            f"  pose_source: {YELLOW}{self.pose_source}{RESET}"
        )