import cv2
import numpy as np

def compute_depth(left_image, right_image):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity

def visualize_depth_map(depth_map):
    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_map_color

