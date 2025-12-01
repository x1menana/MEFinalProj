import cv2
import os
from ultralytics import YOLO
import random

# --- CONFIGURATION ---
# NOTE: Ensure these paths and files exist in your environment.
yolo_model_path = 'yolo_models/yolov8n.pt'
# Updated to use the 'test_vid_30m' directory
video_directories = ['30m_elevation_study_dkr'] # Directory containing video files.
output_directory = 'test_output' # NEW: Directory where output videos will be saved

# Detection settings
required_objects = ['Car'] # Objects to track (converted to lowercase below for robust matching)
min_conf = 0.4 # Minimum confidence threshold for drawing a box

# Initialize model and suppress output
model = YOLO(yolo_model_path)
model.verbose = False

# Convert required objects list to lowercase for robust comparison against YOLO's output names
required_objects_lower = [obj.lower() for obj in required_objects]

# Define a constant white color for visualization
WHITE_COLOR = (255, 255, 255)

def getColours(cls_num):
    """Generates a pseudo-random color based on the class number (not used for boxes/text anymore)."""
    random.seed(cls_num * 100) # Use a seed derived from class ID for consistent coloring
    return tuple(random.randint(0, 255) for _ in range(3))

# --- MAIN PROCESSING LOOP ---

# 1. Iterate over all defined video directories
for dir in video_directories:
    for filename in os.listdir(dir):
        # FIX: Check the lowercase version of the filename for case-insensitive extension matching.
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
             # Skip non-video files
             continue

        video_path = os.path.join(dir, filename)
        print(f'Processing: {video_path}')

        videoCap = cv2.VideoCapture(video_path)
        if not videoCap.isOpened():
            print(f'Invalid file or cannot open video: {video_path}')
            continue # Move to the next file

        # Get video properties for the output file
        fps = int(videoCap.get(cv2.CAP_PROP_FPS))
        width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output VideoWriter setup
        # Ensure the output directory exists before creating the file
        os.makedirs(output_directory, exist_ok=True) 
        
        # Updated output filename prefix
        base_output_filename = f"30m_bboxes_{filename}"
        # Combine directory path with the filename
        output_filename = os.path.join(output_directory, base_output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        print(f"-> Saving output to: {output_filename}")

        frame_count = 0
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break # End of video stream

            frame_count += 1
            # Run YOLO tracking. persist=True maintains tracking IDs between frames.
            results = model.track(frame, persist=True, verbose=False)

            # Process detection results
            if results and results[0].boxes:
                # Iterate through all detected bounding boxes in the current frame
                for box in results[0].boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    class_name = results[0].names[cls].lower() # Get class name and convert to lowercase

                    # 1. FILTERING: Check if the detected object is in the required list
                    if class_name not in required_objects_lower:
                        continue

                    # 2. FILTERING: Check if the confidence meets the minimum threshold
                    if conf >= min_conf:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check for tracking ID (if available, used for consistent tracking visualization)
                        track_id = int(box.id.item()) if box.id is not None else -1

                        # --- DRAWING ---
                        label = f"{class_name.capitalize()} {conf:.2f}"
                        if track_id != -1:
                            label += f" | ID {track_id}"

                        # Draw the bounding box (using WHITE_COLOR)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE_COLOR, 2)

                        # Draw the label text (using WHITE_COLOR)
                        cv2.putText(frame, label,
                                    (x1, max(y1 - 10, 20)), # Position the text slightly above the box
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, WHITE_COLOR, 2)

            # Write annotated frame to output video
            out.write(frame)

            # Optional: Live preview (showing only one window now)
            cv2.imshow(f'Live Preview: {filename}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting loop via 'q' key press.")
                break

        # Cleanup for the current video
        videoCap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f'Finished processing {filename}. Output saved as: {output_filename}')

print("--- All videos processed. ---")