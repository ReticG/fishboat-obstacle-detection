import cv2
import numpy as np

def compute_depth_map(img_left, img_right):
    """
    计算深度图。
    使用 StereoSGBM 进行视差图计算。
    """
    stereo_matcher = cv2.StereoSGBM_create(
        minDisparity=0, 
        numDisparities=16, 
        blockSize=5
    )
    
    disparity = stereo_matcher.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # 假设 Q 矩阵已知，可以通过相机标定得到
    Q = np.array([[1, 0, 0, -img_left.shape[1] / 2],
                  [0, 1, 0, -img_left.shape[0] / 2],
                  [0, 0, 0, 300],  # 距离缩放因子
                  [0, 0, -1 / 0.5, 0]])  # 基线 / 焦距
    
    depth_map = cv2.reprojectImageTo3D(disparity, Q)
    
    return depth_map
