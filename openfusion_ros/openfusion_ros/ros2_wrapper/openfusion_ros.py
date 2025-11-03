import time
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf_transformations

from openfusion_ros.utils import BLUE, BOLD, RESET
from openfusion_ros.ros2_wrapper.camera import CamInfo
from openfusion_ros.ros2_wrapper.robot import Robot
from openfusion_ros.utils.utils import prepare_openfusion_input
from openfusion_ros.ros2_wrapper.utils import is_pose_unique, map_scores_to_colors
from openfusion_ros.slam import build_slam, BaseSLAM
from multimodal_query_msgs.msg import SemanticPrompt


# --------------------------------------------------------------------------- #
# Camera Manager
# --------------------------------------------------------------------------- #
class CameraManager:
    def __init__(self, node):
        self.node = node
        self.cam_info = CamInfo()
        self._received_once = False
        self.subscription = None

    def init_subscription(self, on_first_message=None):
        topic = self.node.get_parameter_or("camera_info.topic", "/camera_info")
        self.node.get_logger().info(f"Subscribing to CameraInfo topic: {topic}")
        self.subscription = self.node.create_subscription(
            CameraInfo, topic, lambda msg: self._callback(msg, on_first_message),
            qos_profile_sensor_data
        )

    def _callback(self, msg: CameraInfo, on_first_message):
        if not self._received_once:
            self.cam_info = CamInfo(msg)
            self._received_once = True
            self.node.get_logger().info(f"Received CameraInfo {msg.width}x{msg.height}")
            if on_first_message:
                on_first_message()

    def is_ready(self):
        return self._received_once

    def get_intrinsics(self):
        return self.cam_info.get_intrinsics()


# --------------------------------------------------------------------------- #
# Publisher Manager
# --------------------------------------------------------------------------- #
class PublisherManager:
    def __init__(self, node):
        self.node = node

        # --- Publishers ---
        self.pose_pub = node.create_publisher(PoseArray, "pose_array", 10)
        self.pc_pub = node.create_publisher(PointCloud2, "pointcloud", 10)
        self.semantic_pc_pub = node.create_publisher(PointCloud2, "semantic_pc", 10)
        self.semantic_pc_pub_xyzi = node.create_publisher(PointCloud2, "semantic_pointcloud_xyzi", 10)

    # -----------------------------------------------------------------------
    def publish_pointcloud(self, frame, points, colors):
        if points is None or len(points) == 0:
            return

        colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        rgb_uint32 = (
            (colors_uint8[:, 0].astype(np.uint32) << 16)
            | (colors_uint8[:, 1].astype(np.uint32) << 8)
            | colors_uint8[:, 2].astype(np.uint32)
        )
        cloud = [(x, y, z, rgb) for (x, y, z), rgb in zip(points, rgb_uint32)]

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame
        msg = pc2.create_cloud(header, fields, cloud)
        self.pc_pub.publish(msg)

    # -----------------------------------------------------------------------
    def publish_pose_array(self, frame, poses):
        msg = PoseArray()
        msg.header.frame_id = frame
        msg.header.stamp = self.node.get_clock().now().to_msg()

        for T in poses:
            inv = np.linalg.inv(T)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = inv[:3, 3]
            q = tf_transformations.quaternion_from_matrix(inv)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
            msg.poses.append(pose)

        self.pose_pub.publish(msg)

    # -----------------------------------------------------------------------
    def publish_semantic_pointcloud_xyzi(self, frame, points, scores):
        """Publish semantic point cloud with intensity (score)."""
        if points is None or len(points) == 0:
            self.node.get_logger().warn("No semantic points to publish (XYZI)")
            return

        # Create (x, y, z, intensity) tuples
        cloud = [(x, y, z, float(i)) for (x, y, z), i in zip(points, scores)]

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame

        msg = pc2.create_cloud(header, fields, cloud)
        self.semantic_pc_pub_xyzi.publish(msg)

# --------------------------------------------------------------------------- #
# Model Manager
# --------------------------------------------------------------------------- #
class FusionModelManager:
    def __init__(self, node, robot):
        self.node = node
        self.robot = robot
        self.model = None
        self.iteration = 0

    def load(self):
        """Wait for camera info and RGB frame, then build OpenFusion SLAM."""
        start = time.time()
        timeout = 5.0  # seconds

        # Wait for camera info and image
        img_h, img_w = self.robot.camera.get_size()
        intrinsics = self.robot.camera.get_intrinsics()

        while (img_h == 0 or img_w == 0 or intrinsics is None) and time.time() - start < timeout:
            self.node.get_logger().warn("Waiting for resized CameraInfo and first RGB frame...")
            rclpy.spin_once(self.node, timeout_sec=0.1)
            img_h, img_w = self.robot.camera.get_size()
            intrinsics = self.robot.camera.get_intrinsics()

        if intrinsics is None or img_h == 0 or img_w == 0:
            self.node.get_logger().error("Timeout waiting for valid CameraInfo / intrinsics.")
            return False

        self.node.get_logger().info(f"Using resized intrinsics for {img_w}x{img_h}")

        # Prepare OpenFusion params
        params, args = prepare_openfusion_input(
            self.robot.camera.get_resized_camera_info(),
            depth_max=10.0,
            algorithm="vlfusion",
            voxel_size=0.01953125,
            block_resolution=8,
            block_count=20000,
            img_size=(img_h, img_w),
            input_size=(img_h, img_w)
        )

        # Build model
        self.model = build_slam(args, intrinsics, params)
        self.node.get_logger().info("SLAM model loaded successfully.")
        return True

    def append_pose(self, pose_data):
        pose, rgb, depth = pose_data
        T_camera_map = np.linalg.inv(pose)
        if not is_pose_unique(
            T_camera_map, self.model.point_state.poses,
            trans_diff_threshold=0.05, fov_deg=5.0
        ): return
        self.iteration += 1
        self.model.io.update(rgb, depth, T_camera_map)
        self.model.vo()
        self.model.compute_state(encode_image=self.iteration % 10 == 0)


# --------------------------------------------------------------------------- #
# Main Node
# --------------------------------------------------------------------------- #
class OpenFusionNode(Node):
    def __init__(self):
        super().__init__("openfusion_node")

        # Managers
        self.pub_mgr = PublisherManager(self)
        self.robot = Robot(self)
        self.model_mgr = FusionModelManager(self, self.robot)
        self.model_loaded = False

        # Subscribe to prompts
        self.prompt_sub = self.create_subscription(
            SemanticPrompt, "/user_prompt", self.handle_prompt, 10
        )

        # --- Wait for camera info and then start model loading ---
        if not self.robot.camera.wait_for_camera_info(timeout=5.0):
            self.get_logger().error("Timeout waiting for CameraInfo.")
        else:
            self.get_logger().info("CameraInfo received, launching model load thread...")
            threading.Thread(target=self._load_model_background, daemon=True).start()

        # Timers
        self.timer_pcl = self.create_timer(0.2, self.publish_pcl)
        self.timer_pose = self.create_timer(1.0, self.update_pose)

    # -----------------------------------------------------------------------
    def _on_first_camera_info(self):
        """Called once when first CameraInfo arrives."""
        self.get_logger().info("Loading model after receiving CameraInfo...")
        threading.Thread(target=self._load_model_background, daemon=True).start()

    def _load_model_background(self):
        if self.model_mgr.load():
            self.model_loaded = True
            self.get_logger().info("Model successfully loaded.")
        else:
            self.get_logger().error("Model failed to load.")

    # -----------------------------------------------------------------------
    def handle_prompt(self, msg):
        self.get_logger().info(f"{BLUE}{BOLD}Prompt received: {msg.text_query}{RESET}")

    def publish_pcl(self):
        if not self.model_loaded:
            return
        model = self.model_mgr.model
        if not model:
            return
        points, colors = model.point_state.get_pc()
        self.pub_mgr.publish_pointcloud("map", points, colors)
        self.pub_mgr.publish_pose_array("map", model.point_state.poses)

    def update_pose(self):
        if not self.model_loaded:
            return
        data = self.robot.get_openfusion_input()
        if data:
            self.model_mgr.append_pose(data)
