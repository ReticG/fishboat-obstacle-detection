import cv2
import os

def load_kitti_data(left_img_path, right_img_path):
    """
    加载 KITTI 数据集中左右图像对。
    """
    img_left = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    img_right = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
    
    if img_left is None or img_right is None:
        raise ValueError(f"Failed to load images from {left_img_path} and {right_img_path}")
    
    return img_left, img_right

def load_ground_truth(depth_path):
    """
    加载深度图。
    """
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Failed to load depth map from {depth_path}")
    return depth_map
