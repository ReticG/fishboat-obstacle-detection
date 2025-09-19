from src.data_loader import StereoDataLoader
from src.detector.yolo_detector import YOLOv5Detector
import cv2

def main():
    # 输入你的视频文件路径
    left_video_path = "path/to/left_video.mp4"
    right_video_path = "path/to/right_video.mp4"

    # 创建数据加载器和检测器
    data_loader = StereoDataLoader(left_video_path, right_video_path)
    detector = YOLOv5Detector()

    while True:
        left_frame, right_frame = data_loader.load_frames()

        if left_frame is None or right_frame is None:
            print("视频读取完毕")
            break

        # 使用YOLOv5进行物体检测
        left_detected = detector.detect(left_frame)
        right_detected = detector.detect(right_frame)

        # 显示检测结果
        cv2.imshow("Left Frame", left_detected)
        cv2.imshow("Right Frame", right_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    data_loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
