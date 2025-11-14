from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

def is_pose_unique(new_pose, poses, trans_diff_threshold=0.05, fov_deg=70.0):
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
