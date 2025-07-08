import rclpy
import rclpy.node
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix

class Camera:
    def __init__(self):
        self.current_image = None
        self.current_depth = None


class Robot:
    def __init__(self, node: rclpy.node.Node):
        self.current_pose = None
        self.camera = Camera()
        self.parent_frame = 'map'
        self.child_frame = 'camera'
        self.node = node

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.none_count = 0

    def get_pose(self):
        try:
            # Current time
            now = rclpy.time.Time()
            # Try to get the transform
            trans = self.tf_buffer.lookup_transform(self.parent_frame, self.child_frame, now)
            
            # Convert to matrix
            t = trans.transform.translation
            q = trans.transform.rotation
            matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
            matrix[0, 3] = t.x
            matrix[1, 3] = t.y
            matrix[2, 3] = t.z

            # Print the matrix nicely
            self.node.get_logger().info(f"\nTransform at time {trans.header.stamp.sec}.{trans.header.stamp.nanosec:09d}:\n{matrix}")
        except Exception as e:
            self.none_count += 1
            self.node.get_logger().warn(f"Transform not available (count: {self.none_count}): {e}")