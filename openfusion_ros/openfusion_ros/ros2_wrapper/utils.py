from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray, Pose
from tf_transformations import quaternion_from_matrix, euler_from_quaternion

import numpy as np
from geometry_msgs.msg import Pose
from tf_transformations import quaternion_from_matrix, euler_from_quaternion
from scipy.spatial.transform import Rotation as R

def is_pose_unique(new_pose_mat: np.ndarray, prev_pose_mat: np.ndarray,
                   trans_diff_threshold=0.3, rot_diff_threshold=45.0):
    if prev_pose_mat is None or new_pose_mat is None:
        return True

    trans_diff = np.linalg.norm(new_pose_mat[:3, 3] - prev_pose_mat[:3, 3])
    r1 = R.from_matrix(prev_pose_mat[:3, :3])
    r2 = R.from_matrix(new_pose_mat[:3, :3])
    angle_deg = np.degrees((r1.inv() * r2).magnitude())

    if (trans_diff < trans_diff_threshold and angle_deg < rot_diff_threshold):
        print(f"Pose not unique: trans_diff {trans_diff:.3f} < {trans_diff_threshold} and angle_deg {angle_deg:.2f} < {rot_diff_threshold}")
        return False
    else:
        print(f"Pose unique: trans_diff {trans_diff:.3f} >= {trans_diff_threshold} or angle_deg {angle_deg:.2f} >= {rot_diff_threshold}")
        return True


# def should_integrate(new_pose, last_pose, trans_threshold=0.15, rot_threshold=5.0):
#     if last_pose is None:
#         return True

#     trans = np.linalg.norm(new_pose[:3,3] - last_pose[:3,3])

#     r1 = R.from_matrix(last_pose[:3,:3])
#     r2 = R.from_matrix(new_pose[:3,:3])
#     rot = np.degrees((r1.inv() * r2).magnitude())

#     # If robot didn't move → don't integrate
#     if trans < 0.03:      
#         return False
    

#     # Accept if robot moved OR rotated significantly
#     return (trans > trans_threshold) or (rot > rot_threshold)

def map_scores_to_colors(query_points, scores, vmin=0.0, vmax=1.0):
    """Converts semantic scores to RGB colors using the inferno colormap with customizable normalization."""
    default_score = vmin
    full_scores = np.full(query_points.shape[0], default_score, dtype=np.float32)

    # Fill known scores
    if scores is not None and len(scores) <= len(full_scores):
        full_scores[:len(scores)] = scores

    # Replace NaNs/Infs and clamp values to [vmin, vmax]
    full_scores = np.nan_to_num(full_scores, nan=vmin, posinf=vmax, neginf=vmin)
    full_scores = np.clip(full_scores, vmin, vmax)

    # Normalize to [0, 1] range for colormap
    norm_scores = (full_scores - vmin) / (vmax - vmin + 1e-8)

    # Apply inferno colormap
    inferno_cmap = plt.get_cmap('inferno')
    rgba = inferno_cmap(norm_scores)  # shape (N, 4), values in [0, 1]
    rgb = rgba[:, :3]  # Drop alpha channel

    return rgb  # shape: (N, 3), values in [0.0, 1.0]
