import cv2
import numpy as np

class MotionEstimator:
    def __init__(self):
        # Increased maxCorners significantly
        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.01,
                                   minDistance=15, 
                                   blockSize=3)
        
        # Improved Lucas-Kanade params for better stability
        self.lk_params = dict(winSize=(21, 21), 
                              maxLevel=3,       
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.prev_gray = None
        self.prev_pts = None
        
        # Grid settings for uniform distribution
        self.grid_rows = 4
        self.grid_cols = 4

    def _get_distributed_features(self, frame_gray):
        """
        Detects features in a grid pattern to ensure coverage across the entire image.
        This prevents the tracker from getting 'stuck' on just one high-contrast area.
        """
        h, w = frame_gray.shape
        step_h = h // self.grid_rows
        step_w = w // self.grid_cols
        
        all_points = []
        
        # Loop through each cell in the 4x4 grid
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Define the box for this grid cell
                r_start, r_end = r * step_h, (r + 1) * step_h
                c_start, c_end = c * step_w, (c + 1) * step_w
                
                # Create a mask that only allows features in this specific cell
                mask = np.zeros_like(frame_gray)
                mask[r_start:r_end, c_start:c_end] = 255
                
                # Calculate how many points we want per cell (Total / 16)
                cell_max_corners = int(self.feature_params['maxCorners'] / (self.grid_rows * self.grid_cols))
                
                cell_params = self.feature_params.copy()
                cell_params['maxCorners'] = cell_max_corners
                
                # Detect features in this cell
                pts = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **cell_params)
                
                if pts is not None:
                    all_points.append(pts)
        
        # Combine all grid points into one big list
        if all_points:
            return np.vstack(all_points)
        return None

    def process_frame(self, frame_color):
        """
        Calculates the global camera motion (dx, dy) using distributed optical flow.
        """
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        # Initialize if first frame
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.prev_pts = self._get_distributed_features(self.prev_gray)
            return 0, 0

        # If we lost too many points (e.g., < 100), re-scan the whole grid
        if self.prev_pts is None or len(self.prev_pts) < 100: 
             self.prev_pts = self._get_distributed_features(self.prev_gray)
             if self.prev_pts is None:
                 self.prev_gray = frame_gray
                 return 0, 0

        # Calculate Optical Flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None, **self.lk_params)

        if next_pts is not None and status is not None:
            # Keep only the "Good" points that were successfully tracked
            good_new = next_pts[status == 1]
            good_old = self.prev_pts[status == 1]

            # If tracking failed for most points, reset
            if len(good_new) < 10:
                 self.prev_gray = frame_gray
                 self.prev_pts = self._get_distributed_features(frame_gray)
                 return 0, 0

            # Calculate movement vectors for all points
            movements = good_new - good_old
            
            # --- THE MAGIC ---
            # We take the MEDIAN. 
            # Cars move differently than the road. Trees move differently than the road.
            # But since the road (background) is the biggest thing, the Median value
            # will almost always represent the true Camera Movement.
            dx = np.median(movements[:, 0])
            dy = np.median(movements[:, 1])
            
            if np.isnan(dx): dx = 0
            if np.isnan(dy): dy = 0

            # Update for next frame
            self.prev_gray = frame_gray.copy()
            self.prev_pts = good_new.reshape(-1, 1, 2)
            
            return dx, dy

        # Fallback
        self.prev_gray = frame_gray
        return 0, 0