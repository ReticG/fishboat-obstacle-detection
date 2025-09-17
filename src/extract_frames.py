import cv2, os, argparse, hashlib

def extract_frames(video_path, output_dir, diff_thr=30, max_fps=2):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    prev, count, fps = None, 0, int(cap.get(cv2.CAP_PROP_FPS))
    interval = fps // max_fps
    while True:
        ret, frame = cap.read()
        if not ret: break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval: continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (640, 360))
        if prev is not None and cv2.countNonZero(cv2.absdiff(gray, prev)) < diff_thr*1000: continue
        prev = gray
        cv2.imwrite(os.path.join(output_dir, f"{count:06d}.jpg"), frame)
        count += 1
    print(f"Saved {count} frames to {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to input video")
    ap.add_argument("--out", required=True, help="path to output frames folder")
    args = ap.parse_args()
    extract_frames(args.video, args.out)
