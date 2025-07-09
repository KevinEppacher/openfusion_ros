from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from vlm_base import BLUE, GREEN, YELLOW, RED, BOLD, RESET

class VLMBaseLifecycleNode(LifecycleNode):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"{BLUE}{BOLD}Initializing {node_name} lifecycle node.{RESET}")
        self._pcl_timer = None
        self._append_pose_timer = None
        self.robot = None

    def on_configure(self, state: State):
        self.get_logger().info(f"{YELLOW}{BOLD}Configuring the lifecycle node...{RESET}")
        try:
            self.load_model()
            self.load_robot()
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE
        
        if self.robot:
            self.robot.on_configure()
            self.get_logger().info(f"{BLUE}{BOLD}Robot configured successfully.{RESET}")
        else:
            self.get_logger().info(f"{YELLOW}No robot instance available for configuration.{RESET}")

        self.get_logger().info(f"{GREEN}{BOLD}Configured lifecycle node{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.get_logger().info(f"{GREEN}{BOLD}Activating the lifecycle node...{RESET}")
        try:
            if self.robot:
                self.robot.on_activate()
            self._pcl_timer = self.create_timer(1.0, self.pcl_timer_callback)
            self._append_pose_timer = self.create_timer(0.1, self.append_pose_timer_callback)
        except Exception as e:
            self.get_logger().error(f"Activation failed: {e}")
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        self.get_logger().info(f"{YELLOW}Deactivating the lifecycle node...{RESET}")
        if self._pcl_timer:
            self._pcl_timer.cancel()
            self._pcl_timer = None
        if self._append_pose_timer:
            self._append_pose_timer.cancel()
            self._append_pose_timer = None
        if self.robot:
            self.robot.deactivate()
        self.get_logger().info(f"{GREEN}{BOLD}Lifecycle node deactivated.{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        self.get_logger().info(f"{BLUE}Cleaning up the lifecycle node...{RESET}")
        self.robot = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        self.get_logger().info(f"{RED}{BOLD}Shutting down the lifecycle node...{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def load_model(self):
        raise NotImplementedError

    def load_robot(self):
        raise NotImplementedError

    def pcl_timer_callback(self):
        raise NotImplementedError

    def append_pose_timer_callback(self):
        raise NotImplementedError
