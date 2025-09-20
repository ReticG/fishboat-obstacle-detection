import cv2
import os
from src.data_loader import load_kitti_data, load_ground_truth
from src.stereo.stereo_matching import compute_depth_map
from src.detector.yolo_detector import load_yolo_model, detect_obstacles
from src.tracker.tracker import track_objects

# 路径设置
left_img_path = 'data/kitti2015/training/image_2/000000_10.png'  # 替换为存在的图像路径
right_img_path = 'data/kitti2015/training/image_3/000000_10.png'  # 替换为存在的图像路径
depth_path = 'data/kitti2015/training/disp_noc_0/000000_10.png'  # 示例深度图路径

# 加载图像数据
img_left, img_right = load_kitti_data(left_img_path, right_img_path)

# 计算深度图
depth_map = compute_depth_map(img_left, img_right)

# 加载 YOLOv8 模型（假设已经有权重文件）
yolo_model = load_yolo_model('yolov8s.pt')  # 使用 YOLOv8 的小模型权重文件 yolov8s.pt

# 检测障碍物
detections, img_with_boxes = detect_obstacles(img_left, yolo_model)

# 显示深度图和检测到的障碍物
cv2.imshow('Depth Map', depth_map)
cv2.imshow('Obstacle Detection', img_with_boxes)

# 等待用户按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
