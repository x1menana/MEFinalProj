# MEFinalProj

**Drone-Based Vehicle Detection & Motion Tracking**

This project implements a robust computer vision pipeline for detecting, tracking, and estimating the motion of vehicles in aerial drone footage. It features a fine-tuned **YOLOv8** model for detection and a custom **Optical Flow** system to compensate for drone camera movement (ego-motion), ensuring that directional arrows are drawn only on vehicles that are actually moving, regardless of altitude or vehicle size.

## Key Features

* **Fine-Tuned Detection:** Utilizes a YOLOv8 Nano model trained specifically on aerial vehicle datasets for high-efficiency inference.

* **Ego-Motion Compensation:** Implements a grid-based **Lucas-Kanade Optical Flow** algorithm to estimate and subtract drone movement from object tracking data. This stabilizes the footage coordinates mathematically, allowing for accurate differentiation between parked and moving cars.

* **Dynamic Motion Thresholding:** Uses a clamped, size-relative thresholding formula. This ensures consistent tracking sensitivity for both large trucks at low altitudes and small cars at high altitudes, while filtering out detection jitter.

* **Visual Analytics:** Annotates video with bounding boxes, confidence scores, tracking IDs, and directional velocity vectors (arrows).

## Project Structure

* **`main.py`**: The core inference script. Handles video I/O, runs the YOLO tracker, manages the history buffer for every vehicle, and draws analytics.

* **`motion_estimator.py`**: A dedicated module for global motion estimation.

  * *Algorithm:* Uses Shi-Tomasi corner detection distributed across a 4x4 grid to find background features (road texturing, etc.).

  * *Tracking:* Applies Pyramidal Lucas-Kanade optical flow to track these features.

  * *Result:* Calculates the median vector of the background to derive the global camera shift `(dx, dy)`.

* **`train.py`**: Utility script for training the YOLO model. Configured to load `yolov8n.pt` and fine-tune using the dataset defined in `data.yaml`.

* **`data.yaml`**: Dataset configuration file defining the root paths for training/validation images and class names (`Vehicle`).

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy