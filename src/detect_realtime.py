from ultralytics import YOLO
import supervision as sv
import cv2, argparse

def main(src, weights):
    model = YOLO(weights)
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(src)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)[0]
        dets = sv.Detections.from_ultralytics(results)
        dets = tracker.update_with_detections(dets)
        ann = sv.BoxAnnotator().annotate(scene=frame.copy(), detections=dets)
        cv2.imshow("Fishboat", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or camera id")
    ap.add_argument("--weights", required=True, help="YOLO .pt file")
    args = ap.parse_args()
    main(args.source, args.weights)
