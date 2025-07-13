import rclpy
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import State
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseArray, Pose
import tf_transformations


from vlm_base.vlm_base import VLMBaseLifecycleNode
from openfusion_ros.utils import BLUE, RED, YELLOW, GREEN, BOLD, RESET
from openfusion_ros.ros2_wrapper.robot import Robot
from openfusion_ros.ros2_wrapper.camera import CamInfo
from openfusion_ros.slam import build_slam, BaseSLAM
from openfusion_ros.utils.utils import prepare_openfusion_input
from openfusion_ros.utils.conversions import convert_stamp_to_sec


class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')
        self.camera_info = CamInfo()
        
        # Publishers
        self.pose_pub = None  # Publisher for PoseArray
        self.pc_pub = None  # LifecyclePublisher for PointCloud2
        self.semantic_pc_pub = None  # Publisher for semantic pointcloud

        # Subcribers
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        self.clock_sub = None

        # Class member variables
        self.latest_clock = None
        self.pose_array = None

    def on_configure(self, state: State):
        result = super().on_configure(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        # Declare parameters
        if not self.has_parameter("parent_frame"):  
            self.declare_parameter("parent_frame", "map")

        # Get parameter values
        self.parent_frame = self.get_parameter("parent_frame").get_parameter_value().string_value

        # Create Publishers
        self.pc_pub = self.create_publisher(PointCloud2, "pointcloud", 10)
        self.semantic_pc_pub = self.create_publisher(PointCloud2,'semantic_pointcloud',10)
        self.pose_pub = self.create_publisher(PoseArray, 'pose_array', 10)

        # Create Subscribers
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        
        self.get_logger().info(f"{GREEN}[{self.get_name()}] PointCloud LifecyclePublisher created.{RESET}")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        if self.pc_pub:
            self.pc_pub.on_activate()
            self.get_logger().info(f"{GREEN}[{self.get_name()}] PointCloud publisher activated.{RESET}")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        if self.pc_pub:
            self.pc_pub.on_deactivate()
            self.get_logger().info(f"{YELLOW}[{self.get_name()}] PointCloud publisher deactivated.{RESET}")

        return super().on_deactivate(state)

    def load_robot(self):
        self.robot = Robot(self)
        return self.robot
    
    def load_model(self):
        if not self.robot:
            self.get_logger().error(f"{RED}Robot is not initialized. Cannot load model.{RESET}")
            return False

        if self.camera_info is None or self.camera_info.cam_info_msg is None:
            self.get_logger().error("CameraInfo not set.")
            return False

        camera_instrinsics = self.camera_info.get_intrinsics()
        if camera_instrinsics is None:
            self.get_logger().warn(f"{RED}Camera intrinsics not set. Cannot load model.{RESET}")
            return False

        params, args = prepare_openfusion_input(self.camera_info,
                                                depth_max=10.0,
                                                algorithm="vlfusion",
                                                voxel_size=0.01953125,
                                                block_resolution=8,
                                                block_count=20000)

        self.get_logger().debug(f"{YELLOW}{BOLD}Loading model...{RESET}")
        self.model = build_slam(args, camera_instrinsics, params)
        self.get_logger().debug(f"{BLUE}{BOLD}Model loaded successfully.{RESET}")
        return True

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = CamInfo(msg)

    def publish_pointcloud(self, points, colors):
        if points is None or len(points) == 0:
            self.get_logger().warn("No points to publish")
            return

        colors = np.clip(colors, 0, 1)
        colors_uint8 = (colors * 255).astype(np.uint8)
        rgb_uint32 = (colors_uint8[:, 0].astype(np.uint32) << 16 |
                      colors_uint8[:, 1].astype(np.uint32) << 8 |
                      colors_uint8[:, 2].astype(np.uint32))
        cloud = [(x, y, z, rgb) for (x, y, z), rgb in zip(points, rgb_uint32)]
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        header = Header()
        header.stamp = self.get_timestamp()
        header.frame_id = "map"
        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.pc_pub.publish(pc2_msg)

    def pcl_timer_callback(self):
        self.model.vo()
        self.model.compute_state(encode_image=True)

        points, colors = self.model.point_state.get_pc()
        self.publish_pointcloud(points, colors)
        self.publish_pose_array()

        self.process_semantic_query()

    def append_pose_timer_callback(self):
        result = self.robot.get_openfusion_input()
        if result is None:
            return

        pose, rgb, depth = result
        T_camera_map = np.linalg.inv(pose)

        self.model.io.update(rgb, depth, T_camera_map)
        self.model.vo()
        self.model.compute_state(encode_image=True)

    def process_semantic_query(self):
        """Handles semantic query and publishing of the filtered pointcloud."""
        try:
            if isinstance(self.model, BaseSLAM) and hasattr(self.model, "query"):
                query_points, scores = self.model.query(
                    "chair", topk=10, only_poi=True
                )

                if query_points is not None and len(query_points) > 0:
                    query_colors = self.map_scores_to_colors(query_points, scores)
                    self.publish_semantic_pointcloud(query_points, query_colors)
                else:
                    self.get_logger().warn(f"Semantic query '{self.semantic_input.text_query}' returned no points.")
        except Exception as e:
            self.get_logger().error(f"Semantic query failed: {e}")

    def map_scores_to_colors(self, query_points, scores):
        """Converts semantic scores to red-scale RGB colors for visualization."""
        # Default minimum score (for points without explicit score)
        default_score = 0.2
        full_scores = np.full(query_points.shape[0], default_score, dtype=np.float32)

        # Fill known scores into full_scores array
        if scores is not None and len(scores) <= len(full_scores):
            full_scores[:len(scores)] = scores

        # Normalize scores and clamp to [0, 1]
        full_scores = np.nan_to_num(full_scores, nan=0.0, posinf=1.0, neginf=0.0)
        full_scores = np.clip(full_scores, 0.0, 1.0)
        full_scores = (full_scores - full_scores.min()) / (full_scores.max() - full_scores.min() + 1e-8)

        # Map to red gradient: dark red (low score) to bright red (high score)
        min_red = 0.4
        red_channel = min_red + full_scores * (1.0 - min_red)
        green_channel = np.zeros_like(red_channel)
        blue_channel = np.zeros_like(red_channel)

        return np.stack([red_channel, green_channel, blue_channel], axis=1)
    
    def publish_semantic_pointcloud(self, points, colors):
        if points is None or len(points) == 0:
            self.get_logger().warn("No semantic points to publish")
            return

        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        rgb_uint32 = (colors_uint8[:, 0].astype(np.uint32) << 16 |
                    colors_uint8[:, 1].astype(np.uint32) << 8 |
                    colors_uint8[:, 2].astype(np.uint32))

        cloud = [(x, y, z, rgb) for (x, y, z), rgb in zip(points, rgb_uint32)]

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        header = Header()
        header.stamp = self.get_timestamp()
        header.frame_id = "map"

        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.semantic_pc_pub.publish(pc2_msg)

    def get_timestamp(self):
        return self.latest_clock if self.latest_clock is not None else self.get_clock().now().to_msg()

    def clock_callback(self, msg: Clock):
        current_time = msg.clock
        if self.latest_clock:
            prev = convert_stamp_to_sec(self.latest_clock)
            curr = convert_stamp_to_sec(current_time)
            if curr < prev:
                self.get_logger().warn("Detected rosbag loop! Clearing TF buffer.")
                self.tf_buffer.clear()

        self.latest_clock = current_time

    def publish_pose_array(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_timestamp()
        pose_array.header.frame_id = self.parent_frame

        for matrix in self.model.point_state.poses:
            inverted_matrix = np.linalg.inv(matrix)
            pose = Pose()
            pose.position.x = inverted_matrix[0, 3]
            pose.position.y = inverted_matrix[1, 3]
            pose.position.z = inverted_matrix[2, 3]
            q = tf_transformations.quaternion_from_matrix(inverted_matrix)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)

        self.pose_pub.publish(pose_array)