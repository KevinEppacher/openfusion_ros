import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
from tf_transformations import quaternion_from_matrix
from openfusion_ros.slam import build_slam, BaseSLAM
from openfusion_ros.datasets import Dataset
from tqdm import tqdm
from argparse import Namespace
from openfusion_ros.configs.build import get_config
import argparse
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from openfusion_ros.utils.utils import (
    show_pc, save_pc, get_cmap_legend
)
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseArray, Pose
import tf_transformations

class TestOpenFusion(Node):
    def __init__(self):
        super().__init__('test_pose_array_publisher')
        self.publisher = self.create_publisher(PoseArray, '/test_pose_array', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/openfusion/pointcloud', 10)
        self.semantic_pc_pub = self.create_publisher(PointCloud2, '/openfusion/semantic_pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/openfusion/pose_array', 10)
        self.pose_matrices = [] 
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("PoseArray publisher started.")
        self.latest_clock = None

        parser = argparse.ArgumentParser()
        parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
        parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
        parser.add_argument('--data', type=str, default="scannet", help='Path to dir of dataset.')
        parser.add_argument('--scene', type=str, default="scene0010_01", help='Name of the scene in the dataset.')
        parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
        parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
        parser.add_argument('--live', action='store_true')
        parser.add_argument('--stream', action='store_true')
        parser.add_argument('--save', action='store_true')
        parser.add_argument('--load', action='store_true')
        parser.add_argument('--host_ip', type=str, default="YOUR IP") # for stream
        args = parser.parse_args()

        params = get_config(args.data, args.scene)
        
        dataset:Dataset = params["dataset"](params["path"], args.frames, args.stream)

        intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])

        print(f"Intrinsic: {intrinsic}")

        self.slam = build_slam(args, intrinsic, params)

        self.dataset_loop(args, self.slam, dataset)

    def dataset_loop(self, args, slam:BaseSLAM, dataset:Dataset):
        i = 0
        for rgb_path, depth_path, extrinsics in tqdm(dataset):
            # if i >= 20:
            #     break
            rgb, depth = self.slam.io.from_file(rgb_path, depth_path)
            self.slam.io.update(rgb, depth, extrinsics)
            self.slam.vo()
            self.slam.compute_state(encode_image=True)
            self.pose_matrices.append(np.linalg.inv(extrinsics))
            i += 1
        points, colors = self.slam.point_state.get_pc()
        show_pc(points, colors, slam.point_state.poses)

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
        header.frame_id = 'map'
        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.pc_pub.publish(pc2_msg)
    
    def publish_semantic_pointcloud(self, points, colors):
        if points is None or len(points) == 0:
            self.get_logger().warn("Semantic query returned no points")
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
        header.frame_id = 'map'
        pc2_msg = pc2.create_cloud(header, fields, cloud)
        self.semantic_pc_pub.publish(pc2_msg)
    
    def get_timestamp(self):
        return self.latest_clock if self.latest_clock is not None else self.get_clock().now().to_msg()
    
    def publish_pose_array(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_timestamp()
        pose_array.header.frame_id = 'map'

        for matrix in self.slam.point_state.poses:
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
        self.get_logger().info(f"Published PoseArray with {len(pose_array.poses)} poses from slam.point_state.")

    def timer_callback(self):
        if not self.pose_matrices:
            return  # noch nichts gesammelt

        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        i = 0
        for matrix in self.pose_matrices:            
            pose_msg = Pose()
            pose_msg.position.x = matrix[0, 3]
            pose_msg.position.y = matrix[1, 3]
            pose_msg.position.z = matrix[2, 3]

            quat = quaternion_from_matrix(matrix)
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]

            pose_array.poses.append(pose_msg)
            i += 1

        self.publisher.publish(pose_array)
        self.get_logger().info(f"Published PoseArray with {len(pose_array.poses)} poses.")

        points, colors = self.slam.point_state.get_pc()

        self.publish_pointcloud(points, colors)
        self.publish_pose_array()

        # Publish semantic query result as PointCloud
        try:
            query = "monitor"
            points, colors = self.slam.semantic_query([
                "vase", "table", "tv shelf", "curtain", "wall", "floor", "ceiling", "door", "tv",
                "room plant", "light", "sofa", "cushion", "wall paint", "chair"
            ])

            if points is not None and len(points) > 0:
                self.publish_semantic_pointcloud(points, colors)
            else:
                self.get_logger().warn(f"Query '{query}' returned no points.")
        except Exception as e:
            self.get_logger().error(f"Semantic query failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TestOpenFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()