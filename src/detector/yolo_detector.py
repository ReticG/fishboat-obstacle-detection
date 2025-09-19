import torch
import cv2

class YOLOv5Detector:
    def __init__(self, model_name="yolov5s"):
        """
        初始化YOLOv5检测模型
        :param model_name: 使用的模型版本（默认为yolov5s）
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)

    def detect(self, frame):
        """
        使用YOLOv5进行物体检测
        :param frame: 输入的图像
        :return: 检测结果图像
        """
        results = self.model(frame)
        results.render()  # 在图像上渲染检测框
        return results.imgs[0]  # 返回渲染后的图像

