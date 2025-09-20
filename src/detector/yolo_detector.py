from ultralytics import YOLO
import torch
import cv2

def load_yolo_model(weights_path):
    """
    加载 YOLOv8 模型
    """
    # 加载 YOLOv8 模型（可以指定使用的设备，'cpu' 或 'cuda:0'）
    model = YOLO(weights_path)  # 直接使用 YOLOv8 的加载方式
    return model


def detect_obstacles(img, model):
    """
    使用 YOLOv8 模型检测障碍物
    """
    # 推理（检测）
    results = model(img)

    # YOLOv8 输出是一个列表，我们需要获取第一个（和唯一一个）结果对象
    result = results[0]  # 获取推理的第一个结果

    # 获取检测框的坐标（xywh 格式），置信度（confidence），类别（class）
    detections = result.boxes.xywh.cpu().numpy()  # 获取边界框坐标，格式为 [x_center, y_center, width, height]
    confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
    classes = result.boxes.cls.cpu().numpy()  # 获取类别 ID

    # 如果是 `xywh` 格式，我们需要转换为 `xyxy` 格式
    # 转换为左上角 (x1, y1) 和右下角 (x2, y2)
    for i, detection in enumerate(detections):
        x_center, y_center, width, height = detection
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 获取当前框的类别和置信度
        conf = confidences[i]
        cls = int(classes[i])

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 使用类名和置信度
        label = f'{model.names[cls]} {conf:.2f}'  # 类别名 + 置信度
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detections, img





