# Drone-Based Vehicle Detection & Motion Tracking

This project implements a computer vision pipeline for detecting, tracking, and estimating the true motion of vehicles in aerial drone footage. It leverages a fine-tuned **YOLOv8** model for object detection and a custom **Grid-Based Optical Flow** system to compensate for drone ego-motion. This allows the system to distinguish between moving vehicles and stationary ones, even while the drone camera is panning or moving.

## Key Features

- **Fine-Tuned Detection:** Uses a custom-trained YOLOv8 Nano model optimized for aerial vehicle detection.
- **Ego-Motion Compensation:** Calculates global camera movement using a grid-distributed Lucas-Kanade Optical Flow algorithm. This subtracts background motion from tracked objects to derive real-world velocity.
- **Size-Relative Thresholding:** Implements dynamic sensitivity based on the bounding box size (20% of vehicle length). This ensures consistent tracking for both large trucks at low altitudes and small cars at high altitudes.
- **Visual Analytics:** Outputs video annotated with bounding boxes, confidence scores, unique tracking IDs, and velocity vectors.

## Project Structure

- `main.py`: The primary inference script. Handles video processing, YOLO tracking, history buffering, motion subtraction, and annotation.
- `motion_estimator.py`: A dedicated module for global motion estimation.
  - **Method:** Divides the frame into a 4x4 grid to ensure uniform feature distribution.
  - **Tracking:** Uses Pyramidal Lucas-Kanade flow on Shi-Tomasi features.
  - **Logic:** Computes the median vector of background features to determine camera shift.
- `train.py`: Training utility to fine-tune `yolov8n.pt` using the configuration in `data.yaml`.
- `data.yaml`: Dataset configuration defining class names and directory paths.

## Installation

Ensure Python 3.8+ is installed. Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

## Usage

### 1. Training (Optional)

To train the model on a new dataset:

1. Verify image paths in `data.yaml`.
2. Run the training script:

```bash
python train.py
```

### 2. Inference

To run the detection and tracking pipeline:

1. Place target videos in the `30m_elevation_study_dkr` directory (or update `video_directories` in `main.py`).
2. Run the main script:

```bash
python main.py
```

### 3. Controls:

- The annotated video is saved to `test_output/`.
- Press `q` to safely stop processing and save the current video.

## Technical Methodology

### Motion Stabilization

Standard tracking fails on drone footage because the camera's movement creates false velocity vectors for stationary objects. This project solves this by calculating the "Camera Shift" `(dx, dy)` for every frame and subtracting it from the object's observed movement.

$$ \vec{V}_{real} = \vec{V}_{observed} - \vec{V}_{camera} $$

### Dynamic Thresholding

To filter out detection jitter (noise) without suppressing slow-moving vehicles, the system calculates a dynamic movement threshold for every object. A vehicle is only marked as "moving" if its displacement exceeds **20% of its own bounding box size**. This scales automatically across different drone altitudes and vehicle types.