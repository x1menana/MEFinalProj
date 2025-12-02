# MEFinalProj

Drone-Based Vehicle Detection & Motion Tracking

This project implements a robust computer vision pipeline for detecting, tracking, and estimating the motion of vehicles in aerial drone footage. It features a fine-tuned YOLOv8 model for detection and a custom Optical Flow system to compensate for drone camera movement (ego-motion), ensuring that directional arrows are drawn only on vehicles that are actually moving, regardless of altitude or vehicle size.

Key Features

Fine-Tuned Detection: Utilizes a YOLOv8 Nano model trained specifically on aerial vehicle datasets for high-efficiency inference.

Ego-Motion Compensation: Implements a grid-based Lucas-Kanade Optical Flow algorithm to estimate and subtract drone movement from object tracking data. This stabilizes the footage coordinates mathematically, allowing for accurate differentiation between parked and moving cars.

Dynamic Motion Thresholding: Uses a clamped, size-relative thresholding formula. This ensures consistent tracking sensitivity for both large trucks at low altitudes and small cars at high altitudes, while filtering out detection jitter.

Visual Analytics: Annotates video with bounding boxes, confidence scores, tracking IDs, and directional velocity vectors (arrows).

Project Structure

main.py: The core inference script. Handles video I/O, runs the YOLO tracker, manages the history buffer for every vehicle, and draws analytics.

motion_estimator.py: A dedicated module for global motion estimation.

Algorithm: Uses Shi-Tomasi corner detection distributed across a 4x4 grid to find background features (road texturing, etc.).

Tracking: Applies Pyramidal Lucas-Kanade optical flow to track these features.

Result: Calculates the median vector of the background to derive the global camera shift (dx, dy).

train.py: Utility script for training the YOLO model. Configured to load yolov8n.pt and fine-tune using the dataset defined in data.yaml.

data.yaml: Dataset configuration file defining the root paths for training/validation images and class names (Vehicle).

Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

pip install ultralytics opencv-python numpy


Usage

1. Training (Optional)

If you wish to retrain or fine-tune the model on new data:

Verify dataset paths in data.yaml.

Run the training script:

python train.py


Weights are saved to runs/detect/trainX/weights/best.pt.

2. Inference (Running the Pipeline)

To run detection on your video dataset:

Place your video files in the target directory (default: 30m_elevation_study_dkr).

Update video_directories in main.py if needed.

Run the main script:

python main.py


Controls:

The processed video will be saved to test_output/.

Press q to safely stop processing, save the current video, and exit.

Technical Methodology

Motion Estimation & Stabilization

Raw bounding box tracking is insufficient for drone footage because the camera itself is moving. A stationary car appears to move in pixel space if the drone pans.

Solution: We calculate the "Camera Shift" every frame.

Math: Real_Velocity = Observed_Pixel_Velocity - Camera_Shift.

This allows the system to maintain a "Real World" coordinate history for every unique Object ID.

Clamped Dynamic Thresholding

To avoid false positives from detection "jitter" (bounding box noise), we apply a dynamic filter before drawing direction arrows.

Formula: Threshold = max(5px, min(Object_Size * 0.2, 20px))

Logic:

Floor (5px): Ignores micro-jitters on tiny objects.

Ceiling (20px): Ensures large vehicles (trucks) don't need excessive speed to be detected.

Scaling: The threshold adapts to 20% of the vehicle's size, making the system robust across different altitudes (30m vs 100m).