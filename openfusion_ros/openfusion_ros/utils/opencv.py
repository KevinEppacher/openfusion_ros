import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

bridge = CvBridge()

def show_image(image, window_name="Image"):
    if image is None or not isinstance(image, np.ndarray):
        print(f"[WARN] Cannot display image: invalid input of type {type(image)}")
        return

    cv2.imshow(window_name, image)
    cv2.waitKey(1)

def show_ros_image(image_msg: Image, window_name="Image"):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        cv2.imshow(window_name, cv_image)
        cv2.waitKey(1)
    except Exception as e:
        print(f"[ERROR] Failed to display ROS image: {e}")

def show_ros_depth_image(image_msg: Image, window_name="Depth Image"):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Handle NaNs
        cv_image = np.nan_to_num(cv_image, nan=0.0)

        # Clip and normalize depth for visualization (adjust range as needed)
        depth_min = 0.5   # meters
        depth_max = 5.0   # meters
        cv_image = np.clip(cv_image, depth_min, depth_max)
        norm_image = ((cv_image - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        # Apply rainbow colormap
        colored_depth = cv2.applyColorMap(norm_image, cv2.COLORMAP_RAINBOW)

        cv2.imshow(window_name, colored_depth)
        cv2.waitKey(1)
    except Exception as e:
        print(f"[ERROR] Failed to display Depth image: {e}")

def ros_image_2_opencv(msg: Image, encoding: str, logger=None):
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

        if encoding == 'passthrough':
            # Optional depth postprocessing
            if cv_img.dtype == np.float32:
                cv_img = (cv_img * 1000.0).astype(np.uint16)
            elif cv_img.dtype != np.uint16:
                if logger:
                    logger.warn(f"Unexpected depth dtype: {cv_img.dtype}")
                return None

        return cv_img

    except Exception as e:
        if logger:
            logger.error(f"Failed to convert image: {e}")
        return None