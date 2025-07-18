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
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import SetParametersResult

from vlm_base.vlm_base import VLMBaseLifecycleNode
from openfusion_ros.utils import BLUE, RED, YELLOW, GREEN, BOLD, RESET
from openfusion_ros.ros2_wrapper.robot import Robot
from openfusion_ros.ros2_wrapper.camera import CamInfo
from openfusion_ros.slam import build_slam, BaseSLAM
from openfusion_ros.utils.utils import prepare_openfusion_input
from openfusion_ros.utils.conversions import convert_stamp_to_sec
from multimodal_query_msgs.msg import SemanticPrompt

class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')
        self.camera_info = CamInfo()

        # Timers
        self._pcl_timer = None
        self._append_pose_timer = None

        # Publishers
        self.pose_pub = None  # Publisher for PoseArray
        self.pc_pub = None  # LifecyclePublisher for PointCloud2
        self.semantic_pc_pub = None  # Publisher for semantic pointcloud

        # Subcribers
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        self.clock_sub = None

        # Class member variables
        self.pose_array = None
        self.semantic_input = None

    def on_configure(self, state: State):
        # Declare parameters
        if not self.has_parameter("robot.parent_frame"):  
            self.declare_parameter("robot.parent_frame", "map")
        if not self.has_parameter("pose_min_translation"):
            self.declare_parameter("pose_min_translation", 0.05)
        if not self.has_parameter("pose_min_rotation"):
            self.declare_parameter("pose_min_rotation", 5.0)
        if not self.has_parameter("topk"):
            self.declare_parameter("topk", 10)
        if not self.has_parameter("skip_loading_model"):
            self.declare_parameter("skip_loading_model", False)

        # Get parameter values
        self.parent_frame = self.get_parameter("robot.parent_frame").get_parameter_value().string_value
        self.pose_min_translation = self.get_parameter("pose_min_translation").get_parameter_value().double_value
        self.pose_min_rotation = self.get_parameter("pose_min_rotation").get_parameter_value().double_value
        self.topk = self.get_parameter("topk").get_parameter_value().integer_value
        self.skip_loading_model = self.get_parameter("skip_loading_model").get_parameter_value().bool_value

        # Call base class configure
        result = super().on_configure(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        # Add dynamic reconfigure
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Create Publishers
        self.pc_pub = self.create_publisher(PointCloud2, "pointcloud", 10)
        self.semantic_pc_pub = self.create_publisher(PointCloud2,'semantic_pointcloud',10)
        self.pose_pub = self.create_publisher(PoseArray, 'pose_array', 10)

        # Create Subscribers
        self.prompt_sub = self.create_subscription(SemanticPrompt, '/user_prompt', self.semantic_prompt_callback, 10)

        # self.print_all_parameters()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        self._pcl_timer = self.create_timer(0.1, self.pcl_timer_callback)
        self._append_pose_timer = self.create_timer(1.0, self.append_pose_timer_callback)
        self.get_logger().info(f"{GREEN}[{self.get_name()}] Timers started.{RESET}")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        if self._pcl_timer:
            self._pcl_timer.cancel()
            self._pcl_timer = None
        if self._append_pose_timer:
            self._append_pose_timer.cancel()
            self._append_pose_timer = None

        return super().on_deactivate(state)
    
    def on_cleanup(self, state: State):
        self.get_logger().info(f"{BLUE}[{self.get_name()}] Cleaning up...{RESET}")
        # Timers
        self._pcl_timer = None
        self._append_pose_timer = None

        # Publishers
        self.pose_pub = None  # Publisher for PoseArray
        self.pc_pub = None  # LifecyclePublisher for PointCloud2
        self.semantic_pc_pub = None  # Publisher for semantic pointcloud

        # Subscribers
        self.prompt_sub = None

        # Class member variables
        self.pose_array = None
        self.semantic_input = None
        self.skip_loading_model = False
        return super().on_cleanup(state)

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

        if self.skip_loading_model == True:
            self.get_logger().info(f"{YELLOW}Skipping model loading as per configuration.{RESET}")
            return True
        
        self.get_logger().debug(f"{YELLOW}{BOLD}Loading model...{RESET}")
        self.model = build_slam(args, camera_instrinsics, params)
        self.get_logger().debug(f"{BLUE}{BOLD}Model loaded successfully.{RESET}")
        return True

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = CamInfo(msg)

    def publish_pointcloud(self, points, colors):
        if points is None or len(points) == 0:
            self.get_logger().warn("Not enough points to publish for pointcloud. SLAM model needs to collect more data.")
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
        header.frame_id = self.parent_frame
        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.pc_pub.publish(pc2_msg)

    def pcl_timer_callback(self):
        if not self.model:
            self.get_logger().warn(f"{YELLOW}Model is not loaded. Cannot publish pointcloud.{RESET}")
            return
        
        self.model.vo()
        self.model.compute_state(encode_image=True)

        points, colors = self.model.point_state.get_pc()
        self.publish_pointcloud(points, colors)
        self.publish_pose_array()

        if len(self.model.point_state.poses) <= 10:
            # self.get_logger().info(f"{YELLOW} Not enough poses to publish semantic pointcloud.{RESET}")
            return

        self.process_semantic_query()

    def append_pose_timer_callback(self):
        result = self.robot.get_openfusion_input()
        if result is None:
            return

        pose, rgb, depth = result
        T_camera_map = np.linalg.inv(pose)

        if self.model is None:
            self.get_logger().warn(f"{YELLOW}Model is not loaded. Cannot append pose.{RESET}")
            return

        # Check if the new pose is significantly different from all existing
        if not self.is_pose_unique(T_camera_map, self.model.point_state.poses,
                                                        trans_diff_threshold=self.pose_min_translation,
                                                        fov_deg=self.pose_min_rotation):
            self.get_logger().debug(f"{YELLOW}[{self.get_name()}] Pose not significantly different. Skipping update.{RESET}")
            return

        self.model.io.update(rgb, depth, T_camera_map)
        self.model.vo()
        self.model.compute_state(encode_image=True)

    def process_semantic_query(self):
        """Handles semantic query and publishing of the filtered pointcloud."""
        try:
            if isinstance(self.model, BaseSLAM) and hasattr(self.model, "query"):
                query_points, scores = self.model.query(
                    self.semantic_input.text_query, topk=self.topk, only_poi=True
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
        header.frame_id = self.parent_frame

        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.semantic_pc_pub.publish(pc2_msg)

    def get_timestamp(self):
        return rclpy.time.Time().to_msg()

    def publish_pose_array(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_timestamp()
        pose_array.header.frame_id = self.parent_frame

        if self.model is None or not hasattr(self.model, 'point_state'):
            return

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

    def is_pose_unique(self, new_pose, poses, trans_diff_threshold=0.05, fov_deg=70.0):
        """
        Check if new_pose is significantly different from all poses in the list.
        Rotation is compared against half of the FOV (i.e., cone angle).
        """
        if not poses or len(poses) == 0:
            return True

        half_fov_deg = fov_deg / 2.0

        for existing_pose in poses:
            trans_diff = np.linalg.norm(new_pose[:3, 3] - existing_pose[:3, 3])

            r1 = R.from_matrix(existing_pose[:3, :3])
            r2 = R.from_matrix(new_pose[:3, :3])
            delta_r = r1.inv() * r2
            angle_deg = np.degrees(np.abs(delta_r.magnitude()))

            if trans_diff < trans_diff_threshold and angle_deg < half_fov_deg:
                return False

        return True

    def semantic_prompt_callback(self, msg: SemanticPrompt):
        self.get_logger().info(f"{BLUE}{BOLD} Received text prompt: {msg.text_query} {RESET}")
        self.semantic_input = msg

        # Process the semantic query
        self.process_semantic_query()

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == "topk" and isinstance(param.value, int):
                self.topk = param.value
                self.get_logger().info(f"Dynamically updated topk to {self.topk}")
        return SetParametersResult(successful=True)
    
    def print_all_parameters(self):
        self.get_logger().info("OpenFusionNode parameters:")
        for name in [
            "append_poses_frequency",
            "pointcloud_frequency",
            "pose_min_translation",
            "pose_min_rotation",
            "parent_frame",
            "child_frame",
            "topk",
            "depth_max",
            "logging.enabled",
            "logging.log_file"
        ]:
            value = self.get_parameter(name).value
            self.get_logger().info(f"  {name}: {value}")