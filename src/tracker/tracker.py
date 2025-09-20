import cv2

def track_objects(img_left, prev_points):
    """
    使用光流法追踪物体位置。
    """
    gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
    
    return next_points
