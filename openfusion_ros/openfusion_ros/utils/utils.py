from openfusion_ros.ros2_wrapper.camera import CamInfo
from argparse import Namespace
import numpy as np
import open3d as o3d
import os
import glob
from scipy.spatial.transform import Rotation as Rot

def prepare_openfusion_input(cam_info : CamInfo, 
                             depth_max : float = 10.0, 
                             algorithm : str = "vlfusion",
                             voxel_size : float = 0.01953125,
                             block_resolution : int = 8,
                             block_count : int = 20000,
                             img_size=None,
                             input_size=None
                             ):
    
    # Get camera resolution from cam_info if not given
    if img_size is None:
        img_size = (cam_info.height, cam_info.width) if cam_info else (480, 640)
    if input_size is None:
        input_size = img_size

    params = {
    'path': '/app/src/OpenFusion/sample/scannet/scene0010_01',
    'depth_scale': 1000.0,
    'depth_max': depth_max,
    'voxel_size': voxel_size,
    'block_resolution': block_resolution,
    'block_count': block_count,
    'img_size': img_size,
    'input_size': input_size  # Keep same if no resizing
    }

    args = Namespace(
        algo=algorithm,
        vl='seem',
        data='scannet',
        scene='scene0010_01',
        frames=-1,
        device='cuda:0',
        live=False,
        stream=False,
        save=False,
        load=False,
        host_ip='YOUR IP'
    )

    return params, args

def custom_intrinsic(intrinsic, old_w, old_h, new_w, new_h):
    new_intrinsic = intrinsic.copy()
    new_intrinsic[0] *= (new_w / old_w)
    new_intrinsic[1] *= (new_h / old_h)
    return new_intrinsic


def get_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_pc(points, colors, poses=None):
    pcd = get_pcd(points, colors)
    if poses is not None:
        cameras_list = []
        for pose in poses:
            camera_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            cameras_list.append(camera_cf.transform(np.linalg.inv(pose)))
        o3d.visualization.draw_geometries([pcd, *cameras_list])
    else:
        o3d.visualization.draw_geometries([pcd])


def save_pc(points, colors, save_path):
    pcd = get_pcd(points, colors)
    o3d.io.write_point_cloud(save_path, pcd)


def kobuki_pose2rgbd():
    # for our Kobuki camera setup
    pose_rgbd_calb = {
        "dX": 32.24/1000,
        "dY": 32.0/1000,
        "dZ": 0.0,
        'dRoll': 180.0,
        'dPitch': 0.0,
        'dYaw': 180.0#
        # roll:x pitch:y yaw:z
    }
    pose2rgbd_matrix = np.eye(4)
    rotation = Rot.from_euler("xyz", [pose_rgbd_calb["dRoll"], pose_rgbd_calb["dPitch"], pose_rgbd_calb["dYaw"]], degrees=True)
    rotation_matrix = rotation.as_matrix()
    pose2rgbd_matrix[:3, :3] = rotation_matrix
    pose2rgbd_matrix[:3, 3] = np.array([pose_rgbd_calb["dX"], pose_rgbd_calb["dY"], pose_rgbd_calb["dZ"]])
    return pose2rgbd_matrix


def preprocess_extrinsics(extrinsics):
    extrinsics = np.stack(extrinsics)
    return relative_transform(
        np.stack([extrinsics[0] for _ in range(len(extrinsics))]),
        extrinsics,
    )


def relative_transform(trans_01, trans_02):
    assert trans_01.shape[-2] == 4 and trans_02.shape[-2] == 4
    trans_10 = np.linalg.inv(trans_01)
    trans_12 = trans_10 @ trans_02
    return trans_12


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False):
    """
    Credit: https://github.com/delestro/rand_cmap/tree/master
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    Args:
        nlabels: Number of labels (size of colormap)
        type: 'bright' for strong colors, 'soft' for pastel colors
        first_color_black: Option to use first color as black, True or False
        last_color_black: Option to use last color as black, True or False
    Usage:
        cmap = rand_cmap(155)
        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([cmap(i) for i in keys.cpu().numpy()])[:,:3]
        )
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [
            (np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.2, high=1),
             np.random.uniform(low=0.9, high=1)
        ) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (np.random.uniform(low=low, high=high),
             np.random.uniform(low=low, high=high),
             np.random.uniform(low=low, high=high)
        ) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap


def get_cmap_legend(cmap, labels, row_length=10, savefile=None):
    """ generate legend for colormap

    Args:
        cmap (matplotlib.colors.Colormap): colormap
        labels (list): list of class names
        row_length (int, optional): number of classes in one row. Defaults to 10.
    """
    import matplotlib.patheffects as pe
    from matplotlib import pyplot as plt

    colors = [cmap(i) for i in range(len(labels))]
    max_text_length = max(len(label) for label in labels)
    # NOTE: adjust the figure size based on the maximum text length
    plt.figure(figsize=(row_length * max_text_length * 0.2, (len(labels) + row_length - 1) // row_length))

    for i, (color, class_name) in enumerate(zip(colors, labels)):
        row = i // row_length
        col = i % row_length
        plt.fill_between([col, col + 1], -row, -row + 1, color=color)
        plt.text(
            col + 0.5, -row + 0.5, class_name, ha='center', va='center',
            rotation=0, color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')]
        )

    plt.axis('off')
    plt.ylim(-(len(labels) + row_length - 1) // row_length, 1)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()