from tf_transformations import quaternion_matrix
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

def transform_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
    matrix[0, 3] = t.x
    matrix[1, 3] = t.y
    matrix[2, 3] = t.z
    return matrix

def pose_msg_to_matrix(pose_msg: Pose):
    q = [pose_msg.orientation.x, pose_msg.orientation.y,
         pose_msg.orientation.z, pose_msg.orientation.w]
    T = quaternion_matrix(q)
    T[0, 3] = pose_msg.position.x
    T[1, 3] = pose_msg.position.y
    T[2, 3] = pose_msg.position.z
    return T

def convert_stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9