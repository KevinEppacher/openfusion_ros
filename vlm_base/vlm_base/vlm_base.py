from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

class VLMBaseLifecycleNode(LifecycleNode):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"Initializing {node_name} lifecycle node.")

    def on_configure(self, state: State):
        self.get_logger().info("Configuring the lifecycle node.")
        self.load_model()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.get_logger().info("Activating the lifecycle node.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        self.get_logger().info("Deactivating the lifecycle node.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        self.get_logger().info("Cleaning up the lifecycle node.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        self.get_logger().info("Shutting down the lifecycle node.")
        return TransitionCallbackReturn.SUCCESS

    # Diese Methoden implementieren Kindklassen:
    def load_model(self):
        raise NotImplementedError