#!/usr/bin/env python3
import os
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import tf_transformations


class ScanNetCollector(Node):
    def __init__(self):
        super().__init__('scannet_collector')

        # ---------------- Parameters ----------------
        self.declare_parameter('output_dir', '/app/src/OpenFusion/sample/scannet/00809-Qpor2mEya8F')
        self.declare_parameter('max_frames', 1000)
        self.declare_parameter('rgb_topic', '/rgb/throttled')
        self.declare_parameter('depth_topic', '/depth/throttled')
        self.declare_parameter('pose_topic', '/openfusion_ros/camera_pose')
        self.declare_parameter('camera_info_topic', '/camera_info')

        self.output_dir = self.get_parameter('output_dir').value
        self.max_frames = self.get_parameter('max_frames').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        # ---------------- Folder setup ----------------
        os.makedirs(os.path.join(self.output_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'pose'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'intrinsic'), exist_ok=True)

        # ---------------- QoS setup ----------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------- Subscribers ----------------
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos)
        self.pose_sub = Subscriber(self, TransformStamped, self.pose_topic, qos_profile=qos)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos)

        # ---------------- Sync setup ----------------
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pose_sub],
            queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)

        # ---------------- Misc ----------------
        self.bridge = CvBridge()
        self.camera_info = None
        self.frame_idx = 0

        self.get_logger().info(f"Collecting up to {self.max_frames} frames into {self.output_dir}")

    # ----------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo):
        """Store intrinsics once and save intrinsics/extrinsics to disk."""
        if self.camera_info is not None:
            return

        self.camera_info = msg
        K = np.array(msg.k).reshape(3, 3)
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = K

        base_path = os.path.join(self.output_dir, 'intrinsic')
        np.savetxt(os.path.join(base_path, 'intrinsic_color.txt'), intrinsic_4x4, fmt='%.6f')
        np.savetxt(os.path.join(base_path, 'intrinsic_depth.txt'), intrinsic_4x4, fmt='%.6f')

        identity = np.eye(4)
        np.savetxt(os.path.join(base_path, 'extrinsic_color.txt'), identity, fmt='%.6f')
        np.savetxt(os.path.join(base_path, 'extrinsic_depth.txt'), identity, fmt='%.6f')

        self.get_logger().info(f"Saved intrinsics and extrinsics to {base_path}")

    # ----------------------------------------------------------------------
    def synced_callback(self, rgb_msg: Image, depth_msg: Image, pose_msg: TransformStamped):
        """Triggered when RGB, Depth and Pose are approximately synchronized."""
        if self.camera_info is None:
            self.get_logger().warn("CameraInfo not received yet. Skipping frame.")
            return
        if self.frame_idx >= self.max_frames:
            self.get_logger().info("Reached maximum frame count — stopping collection.")
            rclpy.shutdown()
            return

        # --- Convert and save RGB ---
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        rgb_path = os.path.join(self.output_dir, 'rgb', f'{self.frame_idx}.png')
        cv2.imwrite(rgb_path, rgb)

        # --- Convert and scale depth ---
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        # Isaac Sim often publishes normalized depth [0, 1]; convert to meters if needed
        if np.max(depth) <= 1.0:
            depth *= 10.0  # assume 10 m far-plane; adjust if needed

        depth_path = os.path.join(self.output_dir, 'depth', f'{self.frame_idx}.png')
        cv2.imwrite(depth_path, depth)

        # --- Handle transform (map → camera) → invert → (camera → map) ---
        t = pose_msg.transform.translation
        q = pose_msg.transform.rotation
        T_map_camera = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T_map_camera[0, 3], T_map_camera[1, 3], T_map_camera[2, 3] = t.x, t.y, t.z

        T_camera_map = np.linalg.inv(T_map_camera)
        pose_path = os.path.join(self.output_dir, 'pose', f'{self.frame_idx}.txt')
        np.savetxt(pose_path, T_camera_map, fmt='%.6f')

        self.get_logger().info(f"Saved frame {self.frame_idx}: RGB+Depth+Pose")
        self.frame_idx += 1


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ScanNetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info("Shutting down ScanNetCollector.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
