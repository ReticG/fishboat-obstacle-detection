import cv2
import os
import numpy as np

class StereoDataLoader:
    def __init__(self, left_video_path, right_video_path):
        """
        初始化数据加载器
        :param left_video_path: 左摄像头视频路径
        :param right_video_path: 右摄像头视频路径
        """
        self.left_cap = cv2.VideoCapture(left_video_path)
        self.right_cap = cv2.VideoCapture(right_video_path)

    def load_frames(self):
        """
        从双目视频中加载一帧数据
        :return: 左右摄像头的图像对
        """
        ret_left, left_frame = self.left_cap.read()
        ret_right, right_frame = self.right_cap.read()

        if not ret_left or not ret_right:
            return None, None

        # 进行一些基本的预处理，如调整尺寸、转换为灰度图等
        left_frame = cv2.resize(left_frame, (640, 480))
        right_frame = cv2.resize(right_frame, (640, 480))

        return left_frame, right_frame

    def release(self):
        """释放视频资源"""
        self.left_cap.release()
        self.right_cap.release()

