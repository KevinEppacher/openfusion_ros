import os
import requests
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from ament_index_python.packages import get_package_share_directory

MODEL_URL = "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"
PACKAGE_NAME = 'openfusion_ros'
PKG_SHARE = get_package_share_directory(PACKAGE_NAME)
DEFAULT_MODEL_PATH = os.path.join(PKG_SHARE, 'openfusion/zoo/xdecoder_seem/checkpoints', 'seem_focall_v1.pt')

class ModelLoader(LifecycleNode):
    def __init__(self):
        super().__init__('load_model')
        self.model_path = DEFAULT_MODEL_PATH
        self.get_logger().info("ModelLoader LifecycleNode initialized.")

    def load_model(self):
        """Checks and loads model if not available."""
        if os.path.exists(self.model_path):
            self.get_logger().info(f"Model already exists at {self.model_path}")
            return True

        try:
            self.get_logger().info(f"Model not found. Downloading from {MODEL_URL}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(self.model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            self.get_logger().info(f"Model successfully downloaded to {self.model_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to download model: {e}")
            return False

    def on_configure(self, state: State):
        self.get_logger().info("Configuring...")
        if self.load_model():
            return TransitionCallbackReturn.SUCCESS
        else:
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State):
        self.get_logger().info("Activating...")
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model not found at {self.model_path}. Cannot activate.")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Model verified. Node activated.")
        self.create_timer(5.0, self.shutdown_node, oneshot=True)
        return TransitionCallbackReturn.SUCCESS

    def shutdown_node(self):
        self.get_logger().info("Timer expired. Shutting down node...")
        rclpy.shutdown()

    def on_deactivate(self, state: State):
        self.get_logger().info("Deactivating...")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        self.get_logger().info("Cleaning up...")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State):
        self.get_logger().info("Shutting down...")
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    model_node = ModelLoader()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(model_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        model_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
