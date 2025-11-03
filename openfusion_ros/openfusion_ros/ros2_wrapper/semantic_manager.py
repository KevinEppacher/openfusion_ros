import threading
import time
import rclpy

class SemanticManager:
    """Runs heavy semantic inference asynchronously, triggered by prompt callbacks."""
    def __init__(self, node, slam):
        self.node = node
        self.slam = slam
        self.model_busy = threading.Event()
        self.semantic_query_active = False
        self.last_prompt = None
        threading.Thread(target=self.loop, daemon=True).start()

    def trigger(self, text):
        """Trigger semantic encoding when a new prompt arrives."""
        self.last_prompt = text
        self.semantic_query_active = True

    def loop(self):
        while rclpy.ok():
            # Triggered only when a new prompt arrives and model not busy
            if self.semantic_query_active and not self.model_busy.is_set():
                self.model_busy.set()
                try:
                    self.node.get_logger().info(
                        f"SemanticManager: starting semantic encoding for '{self.last_prompt}'"
                    )
                    # Encode only once; let the main executor continue collecting data
                    self.slam.compute_state(bs=1, encode_image=True)
                    self.node.get_logger().info("SemanticManager: semantic encoding complete")
                except Exception as e:
                    self.node.get_logger().error(f"SemanticManager failed: {e}")
                finally:
                    self.model_busy.clear()
                    self.semantic_query_active = False
            time.sleep(0.1)
