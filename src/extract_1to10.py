import cv2, os, sys

SRC_DIR  = 'data/raw_video'   # 1.mp4 … 10.mp4
OUT_DIR  = 'data/frames'
INTERVAL = 60                 # 秒

os.makedirs(OUT_DIR, exist_ok=True)

for idx in range(1, 11):
    video_path = os.path.join(SRC_DIR, f'{idx}.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[WARN] 无法打开 {video_path}')
        continue
    fps   = cap.get(cv2.CAP_PROP_FPS)
    step  = max(1, int(fps * INTERVAL))
    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if (frame_id - 1) % step == 0:
            out_name = os.path.join(OUT_DIR, f'{idx}_{count:04d}.jpg')
            cv2.imwrite(out_name, frame)
            count += 1
    cap.release()
    print(f'[INFO] {idx}.mp4 → {count-1} frames')
print('✅ 全部抽帧完成')
