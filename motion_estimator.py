import cv2
import numpy as np

# Function for detecting motion direction/magnitude of detected objects
class MotionEstimator:
    def __init__(self, smoothing_window=30):
        
        self.prev_gray = None
        self.transforms = []
        self.smoothing_window = smoothing_window
        
        # Feature detection parameters (ORB)
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        # Optical flow parameters
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process_frame(self, frame):
        # Compares motion to the previous frame to calculate a dx and dy
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return 0, 0
            
        prev_pts = self.detector.detect(self.prev_gray)
        prev_pts = np.array([pt.pt for pt in prev_pts], dtype='float32').reshape(-1, 1, 2)
        
        if prev_pts.shape[0] < 50:
            self.prev_gray = curr_gray
            return 0, 0

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None, **self.lk_params)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 20:
            self.prev_gray = curr_gray
            return 0, 0
        
        # Estimate dominant motion
        m = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)[0]

        if m is None:
            self.prev_gray = curr_gray
            return 0, 0
        
        dx = m[0, 2]
        dy = m[1, 2]
        
        self.transforms.append((dx, dy))

        if len(self.transforms) > self.smoothing_window:
            self.transforms.pop(0)

        self.prev_gray = curr_gray
        
        return dx, dy