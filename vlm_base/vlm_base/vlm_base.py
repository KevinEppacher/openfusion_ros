from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from vlm_base import BLUE, GREEN, YELLOW, RED, BOLD, RESET

class VLMBaseLifecycleNode(LifecycleNode):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"{BLUE}{BOLD}[{node_name}] Initializing lifecycle node...{RESET}")
        self.robot = None
        self.model = None

    def on_configure(self, state: State):
        self.get_logger().info(f"{YELLOW}{BOLD}[{self.get_name()}] Configuring...{RESET}")
        try:
            self.robot = self.load_robot()
            if not self.robot:
                self.get_logger().error(f"{RED}[{self.get_name()}] Robot could not be created.{RESET}")
                return TransitionCallbackReturn.FAILURE

            self.robot.on_configure()
            self.get_logger().info(f"{GREEN}[{self.get_name()}] Robot configured.{RESET}")

            if not self.load_model():
                self.get_logger().error(f"{RED}[{self.get_name()}] Model could not be loaded.{RESET}")
                return TransitionCallbackReturn.FAILURE

        except Exception as e:
            self.get_logger().error(f"{RED}[{self.get_name()}] Configuration failed: {e}{RESET}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(f"{GREEN}{BOLD}[{self.get_name()}] Configuration complete.{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.get_logger().info(f"{YELLOW}{BOLD}[{self.get_name()}] Activating...{RESET}")
        try:
            if self.robot:
                self.robot.on_activate()
        except Exception as e:
            self.get_logger().error(f"{RED}[{self.get_name()}] Activation failed: {e}{RESET}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(f"{GREEN}[{self.get_name()}] Activated.{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        self.get_logger().info(f"{YELLOW}[{self.get_name()}] Deactivating...{RESET}")
        if self.robot:
            self.robot.on_deactivate()
        self.get_logger().info(f"{BLUE}[{self.get_name()}] Deactivated.{RESET}")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        self.get_logger().info(f"{BLUE}[{self.get_name()}] Cleaning up...{RESET}")
        self.robot = None
        self.model = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        self.get_logger().info(f"{RED}{BOLD}[{self.get_name()}] Shutting down...{RESET}")
        
        return TransitionCallbackReturn.SUCCESS

    def load_model(self):
        raise NotImplementedError

    def load_robot(self):
        raise NotImplementedError

    def pcl_timer_callback(self):
        raise NotImplementedError

    def append_pose_timer_callback(self):
        raise NotImplementedError
