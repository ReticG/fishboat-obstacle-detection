# src/auto_label.py
import os, cv2, tqdm, argparse
from ultralytics import YOLO
from supervision import Detections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8n.pt', help='COCO pretrained or your own .pt')
    ap.add_argument('--imgs',    default='data/frames', help='folder of frames')
    ap.add_argument('--labels',  default='data/labels', help='output YOLO txt folder')
    ap.add_argument('--conf',    type=float, default=0.25, help='confidence threshold')
    ap.add_argument('--classes', nargs='+', type=int,
                    default=[0, 2, 5, 7],          # COCO: person, car, truck, boat
                    help='COCO class ids to keep')
    args = ap.parse_args()

    os.makedirs(args.labels, exist_ok=True)
    model = YOLO(args.weights)
    frames = sorted([f for f in os.listdir(args.imgs) if f.lower().endswith(('.jpg','.jpeg','.png'))])

    for fname in tqdm.tqdm(frames, desc='Auto-label'):
        im_path = os.path.join(args.imgs, fname)
        txt_path = os.path.join(args.labels, os.path.splitext(fname)[0] + '.txt')

        results = model(im_path, conf=args.conf, verbose=False)[0]
        det = Detections.from_ultralytics(results)

        # 只保留指定类别
        mask = [cls in args.classes for cls in det.class_id]
        det = det[mask]

        # 保存 YOLO 格式
        h, w = results.orig_shape
        with open(txt_path, 'w') as f:
            for xyxy, cls_id in zip(det.xyxy, det.class_id):
                x_c = (xyxy[0] + xyxy[2]) / 2 / w
                y_c = (xyxy[1] + xyxy[3]) / 2 / h
                bw  = (xyxy[2] - xyxy[0]) / w
                bh  = (xyxy[3] - xyxy[1]) / h
                f.write(f'{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n')

    print(f'✅ 自动标注完成 → {args.labels}')
    print('下一步：人工纠偏 + 添加缺失类别')

if __name__ == '__main__':
    main()
