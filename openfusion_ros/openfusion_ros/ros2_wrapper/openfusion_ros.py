import rclpy
from sensor_msgs.msg import CameraInfo

from vlm_base.vlm_base import VLMBaseLifecycleNode
from openfusion_ros.utils import BLUE, RED, YELLOW, BOLD, RESET
from openfusion_ros.ros2_wrapper.robot import Robot
from openfusion_ros.utils.opencv import show_image
from openfusion_ros.ros2_wrapper.camera import CamInfo
from openfusion_ros.slam import build_slam, BaseSLAM
from openfusion_ros.utils.utils import prepare_openfusion_input

class OpenFusionNode(VLMBaseLifecycleNode):
    def __init__(self):
        super().__init__('openfusion_node')
        self.camera_info_sub = self.create_subscription(CameraInfo,'/camera_info',self.camera_info_callback,10)
        self.camera_info = CamInfo()

    def load_model(self):
        if not self.robot:
            self.get_logger().error(f"{RED}Robot is not initialized. Cannot load model.{RESET}")
            return False
        
        camera_infos = self.camera_info

        if camera_infos is None or camera_infos.cam_info_msg is None:
            self.get_logger().error("CameraInfo not set.")
            return False

        camera_instrinsics = camera_infos.get_intrinsics()
        width, height = camera_infos.get_size()

        if camera_instrinsics is None:
            self.get_logger().warn(f"{RED}Camera intrinsics not set. Cannot load model.{RESET}")
            return False

        params, args = prepare_openfusion_input(camera_infos, 
                                                depth_max=10.0, algorithm="vlfusion", 
                                                voxel_size=0.01953125, 
                                                block_resolution=8, 
                                                block_count=20000)

        self.get_logger().debug(f"{YELLOW}{BOLD}Loading model...{RESET}")

        # self.model = build_slam(args, camera_instrinsics, params)

        self.get_logger().debug(f"{BLUE}{BOLD}Model loaded successfully.{RESET}")
        return True

    def load_robot(self):
        self.robot = Robot(self)
        return self.robot

    def pcl_timer_callback(self):
        pass

    def append_pose_timer_callback(self):
        result = self.robot.get_openfusion_input()
        if result is None:
            return  # Skip this iteration if input is not ready

        pose, rgb, depth = result



    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = CamInfo(msg)