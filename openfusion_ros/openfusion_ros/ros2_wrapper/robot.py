import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf_transformations import translation_from_matrix, quaternion_from_matrix
from rclpy.duration import Duration

from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET, RED
from openfusion_ros.utils.conversions import transform_to_matrix, convert_stamp_to_sec
from openfusion_ros.ros2_wrapper.camera import Camera
from openfusion_ros.utils.opencv import ros_image_2_opencv

class Robot:
    def __init__(self, node):
        self.node = node
        self.class_name = self.__class__.__name__
        self.current_pose = None
        self.camera = Camera(node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=True)

        self.pose_pub = None
        self.pose_timer = None
        self.publish_interval = 0.01  # seconds

        self.pose_source = "tf"  # default: use TF buffer
        self.pose_sub = None
        self.current_pose_msg = None

        self.prev_pose = None
        self.transform = None

    def on_configure(self):
        self.node.get_logger().debug(f"{BLUE}{BOLD}Configuring {self.class_name}...{RESET}")

        if not self.node.has_parameter("robot.parent_frame"):
            self.node.declare_parameter("robot.parent_frame", "map")
        if not self.node.has_parameter("robot.child_frame"):
            self.node.declare_parameter("robot.child_frame", "camera")
        if not self.node.has_parameter("robot.pose_topic"):
            self.node.declare_parameter("robot.pose_topic", "robot_pose")
        if not self.node.has_parameter("robot.max_delta_time"):
            self.node.declare_parameter("robot.max_delta_time", 0.01)
        if not self.node.has_parameter("robot.pose_source"):
            self.node.declare_parameter("robot.pose_source", "tf")      # options: "tf" or "topic"

        self.parent_frame = self.node.get_parameter("robot.parent_frame").get_parameter_value().string_value
        self.child_frame = self.node.get_parameter("robot.child_frame").get_parameter_value().string_value
        self.pose_topic = self.node.get_parameter("robot.pose_topic").get_parameter_value().string_value
        self.max_delta_time = self.node.get_parameter("robot.max_delta_time").get_parameter_value().double_value
        self.pose_source = self.node.get_parameter("robot.pose_source").get_parameter_value().string_value

        self.node.get_logger().debug(f"{BLUE}{BOLD}Finished configuring {self.class_name}{RESET}")
        self.camera.on_configure()

    def on_activate(self):
        self.node.get_logger().debug(f"{YELLOW}{BOLD}Activating {self.class_name}...{RESET}")
        self.camera.on_activate()

        self.pose_pub = self.node.create_publisher(PoseStamped, self.pose_topic+"/debug", 10)
        self.pose_timer = self.node.create_timer(self.publish_interval, self.publish_pose)

        if self.pose_source == "topic":
            self.pose_sub = self.node.create_subscription(
                TransformStamped,
                self.pose_topic,
                self.pose_callback,
                10
            )
            self.node.get_logger().info(f"Subscribed to pose topic: {self.pose_topic}")
        else:
            self.node.get_logger().info(f"Using TF buffer for pose lookup: {self.parent_frame} → {self.child_frame}")

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

    def update_pose(self, _=None):
        if self.pose_source == "topic":
            if self.current_pose_msg is None:
                self.node.get_logger().info("No pose received yet from topic.")
                return None
            self.transform = self.current_pose_msg
            return

        try:
            rgb = self.camera.get_rgb(which="latest")
            if rgb is not None:
                time_target = rclpy.time.Time.from_msg(rgb.header.stamp)
            else:
                time_target = rclpy.time.Time()
            self.transform = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, time_target)
        except Exception as e:
            self.node.get_logger().warn(f"Transform not available: {e}")
            self.transform = None

    def get_pose(self, datatype='matrix'):
        self.update_pose(self.current_pose)
        transform = self.transform
        if transform is None:
            self.node.get_logger().warn("Transform is None. Cannot get pose.")
            return None
                
        match datatype:
            case 'matrix':
                return transform_to_matrix(transform)
            case 'tf2':
                return transform
            case _:
                self.node.get_logger().warn(f"Unknown datatype '{datatype}' requested.")
                return None
        
    def get_openfusion_input(self):
        pose = self.get_pose(datatype='tf2')
        rgb = self.camera.get_rgb(which='latest')
        depth = self.camera.get_depth(which='latest')

        # Check if any of the images or pose are None
        if pose is None or rgb is None or depth is None:
            reason = (
                "pose" if pose is None else
                "RGB" if rgb is None else
                "depth"
            )
            self.node.get_logger().warn(f"Skipping frame: {reason} is None.")
            return None

        # Extract timestamps
        rgb_time = convert_stamp_to_sec(rgb.header.stamp)
        depth_time = convert_stamp_to_sec(depth.header.stamp)
        pose_time =convert_stamp_to_sec(pose.header.stamp)

        # Check if the timestamps are synchronized
        max_diff = max(
            abs(rgb_time - depth_time),
            abs(rgb_time - pose_time),
            abs(depth_time - pose_time)
        )

        if max_diff > self.max_delta_time:
            self.node.get_logger().warn(f"Timestamps not synchronized (Δ={max_diff:.3f}s). Skipping frame.")
            return None

        converted_pose = transform_to_matrix(pose)
        converted_rgb = ros_image_2_opencv(rgb, 'bgr8', self.node.get_logger())
        converted_depth = ros_image_2_opencv(depth, 'passthrough', self.node.get_logger())
        return converted_pose, converted_rgb, converted_depth

    def publish_pose(self):
        now = self.node.get_clock().now()

        pose_matrix = self.get_pose(datatype='matrix')
        if pose_matrix is None:
            return

        translation = translation_from_matrix(pose_matrix)
        quaternion = quaternion_from_matrix(pose_matrix)

        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.parent_frame
        msg.pose.position.x = translation[0]
        msg.pose.position.y = translation[1]
        msg.pose.position.z = translation[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(msg)
        self.last_publish_time = now

    def pose_callback(self, msg: PoseStamped):
        self.current_pose_msg = msg