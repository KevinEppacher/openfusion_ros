#!/usr/bin/env python3
import time
import threading
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import tf_transformations
import os
import json
import open3d as o3d
from nav_msgs.msg import OccupancyGrid
import re
import subprocess

from openfusion_ros.utils import BLUE, YELLOW, RED, GREEN, BOLD, RESET
from openfusion_ros.ros2_wrapper.robot import Robot
from openfusion_ros.utils.utils import prepare_openfusion_input
from openfusion_ros.ros2_wrapper.utils import is_pose_unique, map_scores_to_colors
from openfusion_ros.slam import build_slam, BaseSLAM
from multimodal_query_msgs.msg import SemanticPrompt
from std_srvs.srv import Trigger

# --------------------------------------------------------------------------- #
# Semantic Map Saver (Auto-Versioning + SLAM Snapshot)
# --------------------------------------------------------------------------- #
class SemanticMapSaver:
    """Handles saving semantic point clouds and SLAM maps into dataset structure with versioning."""

    def __init__(self, node: Node):
        self.node = node

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        node.declare_parameter("dataset.root_dir", "/app/src/sage_evaluator/datasets/matterport_isaac")
        node.declare_parameter("dataset.scene_name", "00809-Qpor2mEya8F")
        node.declare_parameter("slam.map_topic", "/map")

        self.root_dir = node.get_parameter("dataset.root_dir").value
        self.scene_name = node.get_parameter("dataset.scene_name").value
        self.map_topic = node.get_parameter("slam.map_topic").value

        # Derived paths
        self.scene_dir = os.path.join(self.root_dir, self.scene_name)
        self.annotations_dir = os.path.join(self.scene_dir, "annotations")
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Auto version detection
        self.annotation_version = self._get_next_version()
        self.annotation_dir = os.path.join(self.annotations_dir, self.annotation_version)
        os.makedirs(self.annotation_dir, exist_ok=True)

        node.get_logger().info(
            f"{BLUE}SemanticMapSaver initialized:{RESET}\n"
            f"  Scene: {self.scene_name}\n"
            f"  New Annotation version: {self.annotation_version}\n"
            f"  Output directory: {self.annotation_dir}"
        )

    # ------------------------------------------------------------------ #
    # Detect latest version and increment it (e.g., v1.0 → v1.1)
    # ------------------------------------------------------------------ #
    def _get_next_version(self):
        existing = [
            d for d in os.listdir(self.annotations_dir)
            if re.match(r"^v\d+\.\d+$", d) and os.path.isdir(os.path.join(self.annotations_dir, d))
        ]
        if not existing:
            return "v1.0"

        # Sort by numeric version (major.minor)
        def parse_version(v):
            major, minor = v[1:].split(".")
            return int(major), int(minor)

        existing.sort(key=parse_version)
        last_major, last_minor = parse_version(existing[-1])
        new_version = f"v{last_major}.{last_minor + 1}"
        return new_version

    # ------------------------------------------------------------------ #
    # Save Semantic Map + SLAM map
    # ------------------------------------------------------------------ #
    def save(self, points, colors, class_ids, class_list, filename_prefix="semantic_map"):
        """Save semantic pointcloud, class mapping, and SLAM map using Nav2 CLI."""

        if points is None or len(points) == 0:
            self.node.get_logger().warn("No points to save — skipping.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ply_path = os.path.join(self.annotation_dir, f"{filename_prefix}_{timestamp}.ply")
        json_path = os.path.join(self.annotation_dir, f"{filename_prefix}_{timestamp}.json")
        map_base = os.path.join(self.annotation_dir, f"slam_map_{timestamp}")

        # ------------------------------------------------------------------ #
        # Save default robot_start_pose.json
        # ------------------------------------------------------------------ #
        try:
            start_pose_path = os.path.join(self.annotation_dir, "robot_start_pose.json")
            if not os.path.exists(start_pose_path):
                start_pose = {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0
                }
                with open(start_pose_path, "w") as f:
                    json.dump(start_pose, f, indent=2)
                self.node.get_logger().info(f"{GREEN}Created zeroed robot_start_pose.json → {start_pose_path}{RESET}")
            else:
                self.node.get_logger().info(f"{YELLOW}robot_start_pose.json already exists → {start_pose_path}{RESET}")
        except Exception as e:
            self.node.get_logger().error(f"{RED}Failed to save robot_start_pose.json: {e}{RESET}")

        # ------------------------------------------------------------------ #
        # Save Semantic Pointcloud
        # ------------------------------------------------------------------ #
        try:
            if class_ids.ndim == 1:
                class_ids = class_ids.reshape(-1, 1)
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
            pcd.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
            pcd.point["class_id"] = o3d.core.Tensor(class_ids, dtype=o3d.core.Dtype.Int32)
            o3d.t.io.write_point_cloud(ply_path, pcd)
            self.node.get_logger().info(f"{GREEN}Saved semantic point cloud → {ply_path}{RESET}")
        except Exception as e:
            self.node.get_logger().error(f"{RED}Failed to save PLY: {e}{RESET}")
            return

        # ------------------------------------------------------------------ #
        # Save Class Mapping
        # ------------------------------------------------------------------ #
        try:
            class_mapping = {int(i): str(name) for i, name in enumerate(class_list)}
            with open(json_path, "w") as f:
                json.dump(class_mapping, f, indent=2)
            self.node.get_logger().info(f"{GREEN}Saved class mapping → {json_path}{RESET}")
        except Exception as e:
            self.node.get_logger().error(f"{RED}Failed to save JSON mapping: {e}{RESET}")

        # ------------------------------------------------------------------ #
        # Save SLAM Map via CLI (identical to Nav2 behavior)
        # ------------------------------------------------------------------ #
        try:
            cmd = [
                "ros2", "run", "nav2_map_server", "map_saver_cli",
                "-f", map_base,
                "--ros-args", "-p", f"topic:={self.map_topic}"
            ]
            self.node.get_logger().info(f"{BLUE}Running map_saver_cli for map export...{RESET}")
            subprocess.run(cmd, check=True)
            self.node.get_logger().info(f"{GREEN}Saved SLAM map using map_saver_cli → {map_base}.pgm{RESET}")
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"{RED}map_saver_cli failed: {e}{RESET}")
        except FileNotFoundError:
            self.node.get_logger().error(f"{RED}map_saver_cli not found — is nav2_map_server installed?{RESET}")

        # ------------------------------------------------------------------ #
        # Update 'current' symlink
        # ------------------------------------------------------------------ #
        current_link = os.path.join(self.annotations_dir, "current")
        try:
            if os.path.islink(current_link) or os.path.exists(current_link):
                os.remove(current_link)
            os.symlink(self.annotation_dir, current_link)
            self.node.get_logger().info(f"{BLUE}Updated 'current' annotation link → {current_link}{RESET}")
        except Exception as e:
            self.node.get_logger().warn(f"{YELLOW}Could not update 'current' link: {e}{RESET}")



# --------------------------------------------------------------------------- #
# Publisher Manager
# --------------------------------------------------------------------------- #
class PublisherManager:
    def __init__(self, node):
        self.node = node

        # Declare topic parameters for flexibility
        node.declare_parameter("topic_names.slam_pointcloud", "slam_pointcloud")
        node.declare_parameter("topic_names.query_pointcloud", "query_pointcloud_xyzi")
        node.declare_parameter("topic_names.semantic_pointcloud", "semantic_pointcloud_rgb")
        node.declare_parameter("topic_names.panoptic_pointcloud", "panoptic_pointcloud_rgb")
        node.declare_parameter("topic_names.pose_array", "pose_array")

        # Retrieve topic names
        self.slam_topic = node.get_parameter("topic_names.slam_pointcloud").value
        self.query_topic = node.get_parameter("topic_names.query_pointcloud").value
        self.semantic_topic = node.get_parameter("topic_names.semantic_pointcloud").value
        self.panoptic_topic = node.get_parameter("topic_names.panoptic_pointcloud").value
        self.pose_array_topic = node.get_parameter("topic_names.pose_array").value

        # Publishers
        self.pose_pub = node.create_publisher(PoseArray, self.pose_array_topic, 10)
        self.slam_pub = node.create_publisher(PointCloud2, self.slam_topic, 10)
        self.query_pub = node.create_publisher(PointCloud2, self.query_topic, 10)
        self.semantic_pub = node.create_publisher(PointCloud2, self.semantic_topic, 10)
        self.panoptic_pub = node.create_publisher(PointCloud2, self.panoptic_topic, 10)

    def _make_colored_cloud(self, frame, points, colors):
        if points is None or len(points) == 0:
            return None
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
        return pc2.create_cloud(header, fields, cloud)

    def publish_slam(self, frame, points, colors):
        msg = self._make_colored_cloud(frame, points, colors)
        if msg:
            self.slam_pub.publish(msg)

    def publish_query(self, frame, points, scores):
        """Query cloud as XYZI (score intensity)."""
        if points is None or len(points) == 0:
            return
        cloud = [(x, y, z, float(i)) for (x, y, z), i in zip(points, scores)]
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame
        msg = pc2.create_cloud(header, fields, cloud)
        self.query_pub.publish(msg)

    def publish_semantic(self, frame, points, colors):
        msg = self._make_colored_cloud(frame, points, colors)
        if msg:
            self.semantic_pub.publish(msg)

    def publish_panoptic(self, frame, points, colors):
        msg = self._make_colored_cloud(frame, points, colors)
        if msg:
            self.panoptic_pub.publish(msg)

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

# --------------------------------------------------------------------------- #
# Model Manager
# --------------------------------------------------------------------------- #
class FusionModelManager:
    def __init__(self, node, robot):
        self.node = node
        self.robot = robot
        self.model = None
        self.iteration = 0

        node.declare_parameter("append_pose.min_translation", 0.05)
        node.declare_parameter("append_pose.max_rotation_deg", 5.0)
        node.declare_parameter("encode_image_every_n_frames", 10)

        self.min_trans = node.get_parameter("append_pose.min_translation").value
        self.max_rot_deg = node.get_parameter("append_pose.max_rotation_deg").value
        self.encode_image_every_n_frames = node.get_parameter("encode_image_every_n_frames").value

    def load(self):
        """Wait for camera info and RGB frame, then build OpenFusion SLAM."""
        start = time.time()
        timeout = 30.0  # seconds

        # Wait for camera info and image
        img_w, img_h = self.robot.camera.get_size()
        intrinsics = self.robot.camera.get_intrinsics()

        while (img_w == 0 or img_h == 0 or intrinsics is None) and time.time() - start < timeout:
            self.node.get_logger().warn("Waiting for resized CameraInfo and first RGB frame...")
            img_w, img_h = self.robot.camera.get_size()
            intrinsics = self.robot.camera.get_intrinsics()
            time.sleep(0.1)

        if intrinsics is None or img_w == 0 or img_h == 0:
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
            block_count=60000,
            img_size=(img_w, img_h),
            input_size=(img_w, img_h)
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
            trans_diff_threshold=self.min_trans, fov_deg=self.max_rot_deg
        ): return
        self.iteration += 1
        self.model.io.update(rgb, depth, T_camera_map)
        self.model.vo()
        self.model.compute_state(encode_image=(self.iteration % self.encode_image_every_n_frames == 0))

class SemanticProcessor:
    def __init__(self, node, model_mgr, pub_mgr):
        self.node = node
        self.model_mgr = model_mgr
        self.pub_mgr = pub_mgr
        self.saver = SemanticMapSaver(node)

        # --- Declare configurable parameters ------------------------------
        self.mode = self.declare_param("semantic.mode", "query")
        self.topk = self.declare_param("semantic.topk", 3)
        self.min_score = self.declare_param("semantic.min_score", 0.1)
        self.max_score = self.declare_param("semantic.max_score", 1.0)

        # Class list configuration
        class_list_path = self.declare_param("semantic.class_list_path", "")

        default_class_list = [
            "vase", "table", "tv shelf", "curtain", "wall", "floor", "ceiling",
            "door", "tv", "room plant", "light", "sofa", "cushion", "wall paint", "chair"
        ]

        # --- Load class list dynamically ---
        self.class_list = self.load_class_list(
            class_list_path=class_list_path,
            default_list=default_class_list
        )

        # Store latest prompt input
        self.semantic_input = None

        # --- Log summary ---
        self.node.get_logger().info(
            f"{BLUE}{BOLD}SemanticProcessor initialized:{RESET}\n"
            f"  mode: {YELLOW}{self.mode}{RESET}\n"
            f"  topk: {YELLOW}{self.topk}{RESET}\n"
            f"  score range: {YELLOW}{self.min_score} – {self.max_score}{RESET}\n"
            f"  class source: {YELLOW}{'JSON file' if class_list_path else 'ROS param'}{RESET}\n"
            f"  classes: {YELLOW}{', '.join(self.class_list[:10])}... ({len(self.class_list)} total){RESET}"
        )


    def declare_param(self, name, default):
        if not self.node.has_parameter(name):
            self.node.declare_parameter(name, default)
        return self.node.get_parameter(name).value

    def handle_prompt(self, msg):
        """Callback for incoming SemanticPrompt messages."""
        self.semantic_input = msg
        self.node.get_logger().info(
            f"{BLUE}{BOLD}Received semantic prompt: {msg.text_query}{RESET}"
        )
        self.process_query()

    def process_query(self):
        """Run semantic query and publish result point clouds."""
        model = self.model_mgr.model
        if model is None or not isinstance(model, BaseSLAM):
            self.node.get_logger().warn("Model not initialized or not BaseSLAM instance.")
            return

        if not self.semantic_input:
            self.node.get_logger().warn("No text query provided.")
            return

        try:
            text_query = self.semantic_input.text_query
            self.node.get_logger().info(f"{YELLOW}Running semantic query: '{text_query}'{RESET}")

            # Perform the query
            query_points, scores = model.query(
                text_query, topk=self.topk, only_poi=True
            )

            # Validate and publish
            if query_points is None or len(query_points) == 0:
                self.node.get_logger().warn(f"Query '{text_query}' returned no points.")
                return

            # colors = map_scores_to_colors(
            #     query_points, scores,
            #     vmin=self.min_score, vmax=self.max_score
            # )

            # self.pub_mgr.publish_pointcloud("map", query_points, colors)
            self.pub_mgr.publish_query("map", query_points, scores)

            self.node.get_logger().info(
                f"{BOLD}Query '{text_query}' finished: {len(query_points)} points published.{RESET}"
            )

        except Exception as e:
            self.node.get_logger().error(f"{RED}Semantic query failed: {e}{RESET}")

    def process_auto(self):
        model = self.model_mgr.model
        if model is None or not isinstance(model, BaseSLAM):
            self.node.get_logger().warn("Model not initialized or not BaseSLAM instance.")
            return

        try:
            mode = self.mode.lower()
            if mode == "panoptic" and hasattr(model, "panoptic_query"):
                result = model.panoptic_query(self.class_list)
            elif mode == "semantic" and hasattr(model, "semantic_query"):
                result = model.semantic_query(self.class_list)
            elif mode == "query" and self.semantic_input:
                result = model.query(self.semantic_input.text_query, topk=self.topk, only_poi=True)
            else:
                return

            if not isinstance(result, tuple) or len(result) not in (2, 3, 4):
                self.node.get_logger().warn("Invalid query result type.")
                return

            if len(result) == 2:
                points, colors = result
                scores = np.zeros(len(points), dtype=np.float32)
            elif len(result) == 3:
                points, colors, class_ids = result
                self.saver.save(points, colors, class_ids, self.class_list, filename_prefix=mode)
            else:
                self.node.get_logger().warn("Unexpected number of return values from query.")

            if points is None or len(points) == 0:
                self.node.get_logger().warn(f"{mode.capitalize()} query returned no points.")
                return

            if mode == "query":
                self.pub_mgr.publish_query("map", points, scores)
            elif mode == "semantic":
                self.pub_mgr.publish_semantic("map", points, colors)
            elif mode == "panoptic":
                self.pub_mgr.publish_panoptic("map", points, colors)

            self.node.get_logger().info(f"{mode.capitalize()} map published ({len(points)} points).")

        except Exception as e:
            self.node.get_logger().error(f"{RED}Auto query failed: {e}{RESET}")

    def load_class_list(self, class_list_path: str, default_list: list):
        """Load class list from JSON file if available, otherwise use parameter or default."""
        # 1. If JSON path is given and valid, try to load
        if class_list_path and os.path.exists(class_list_path):
            try:
                with open(class_list_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.node.get_logger().info(
                        f"{BOLD}Loaded class list from JSON file:{RESET} {class_list_path}"
                    )
                    return loaded
                else:
                    self.node.get_logger().warn(
                        f"Invalid format in {class_list_path}, expected a list. Using default class list."
                    )
                    return default_list
            except Exception as e:
                self.node.get_logger().error(f"Failed to load class list from {class_list_path}: {e}")
                return default_list

        # 2. If JSON path not provided or invalid → use declared param or fallback
        return self.declare_param("semantic.class_list", default_list)
    
# --------------------------------------------------------------------------- #
# Main Node
# --------------------------------------------------------------------------- #
class OpenFusionNode(Node):
    def __init__(self):
        super().__init__("openfusion_node")

        self.pub_mgr = PublisherManager(self)
        self.robot = Robot(self)
        self.model_mgr = FusionModelManager(self, self.robot)
        self.semantic_proc = SemanticProcessor(self, self.model_mgr, self.pub_mgr)
        self.model_loaded = False

        # --- Declare configurable parameters ------------------------------
        self.declare_parameter("timer_period.update_pose", 1.0)
        self.declare_parameter("timer_period.publish_pcl", 3.0)
        self.declare_parameter("user_prompt_topic", "/user_prompt")

        self.update_pose_period = self.get_parameter("timer_period.update_pose").get_parameter_value().double_value
        self.pcl_period = self.get_parameter("timer_period.publish_pcl").get_parameter_value().double_value
        user_prompt_topic = self.get_parameter("user_prompt_topic").get_parameter_value().string_value

        # Subscribe for semantic prompts
        self.prompt_sub = self.create_subscription(
            SemanticPrompt, user_prompt_topic, self.semantic_proc.handle_prompt, 10
        )

        # Wait for camera info, then load model
        if not self.robot.camera.wait_for_camera_info(timeout=30.0):
            self.get_logger().error("Timeout waiting for CameraInfo.")
        else:
            threading.Thread(target=self._load_model_background, daemon=True).start()

        # Timers
        self.timer_pcl = self.create_timer(self.pcl_period, self.publish_pcl)
        self.timer_pose = self.create_timer(self.update_pose_period, self.update_pose)

        # Services for semantic & panoptic triggers
        self.semantic_srv = self.create_service(
            Trigger, "run_semantic_map", self.run_semantic_cb)
        self.panoptic_srv = self.create_service(
            Trigger, "run_panoptic_map", self.run_panoptic_cb)
        self.get_logger().info("Semantic & panoptic trigger services ready.")

        # Internal timing state
        self.last_pcl_start = 0.0
        self.last_pose_start = 0.0

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

    def handle_prompt(self, msg):
        self.get_logger().info(f"{BLUE}{BOLD}Prompt received: {msg.text_query}{RESET}")

    def publish_pcl(self):
        if not self.model_loaded:
            return
        model = self.model_mgr.model
        if not model:
            return
        points, colors = model.point_state.get_pc()
        self.pub_mgr.publish_slam("map", points, colors)

    def update_pose(self):
        # Measure start time for overrun detection
        start = time.time()

        if not self.model_loaded:
            return
        data = self.robot.get_openfusion_input()
        if data:
            self.model_mgr.append_pose(data)
        self.pub_mgr.publish_pose_array("map", self.model_mgr.model.point_state.poses)

        # Check for timer overruns
        elapsed = time.time() - start
        if elapsed > self.update_pose_period:
            self.get_logger().warn(
                f"[update_pose] exceeded timer period: "
                f"{elapsed:.3f}s > {self.update_pose_period:.3f}s"
            )

    def run_semantic_cb(self, request, response):
        try:
            self.semantic_proc.mode = "semantic"
            self.semantic_proc.process_auto()
            response.success = True
            response.message = "Semantic map generation completed."
        except Exception as e:
            response.success = False
            response.message = f"Semantic map failed: {e}"
        return response

    def run_panoptic_cb(self, request, response):
        try:
            self.semantic_proc.mode = "panoptic"
            self.semantic_proc.process_auto()
            response.success = True
            response.message = "Panoptic map generation completed."
        except Exception as e:
            response.success = False
            response.message = f"Panoptic map failed: {e}"
        return response