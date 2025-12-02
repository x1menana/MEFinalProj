import cv2
import os
from ultralytics import YOLO
import random
import math
from motion_estimator import MotionEstimator

fine_tuned_model_path = 'runs/detect/train9/weights/best.pt'
# video_directories = ['30m_elevation_study_dkr', '100m_elevation_study_wateloo']
video_directories = ['30m_elevation_study_dkr'] # Test run
output_directory = 'test_output' # Where annotated videos are saved

# Detection settings
required_objects = ['Vehicle'] 
min_conf = 0.7 # Adjustable

# Initiate model
try:
    model = YOLO(fine_tuned_model_path) # Load model
    model.verbose = False
    CLASS_NAMES = model.names
    print(f'Fine-tuned model loaded with names: {CLASS_NAMES}')
except FileNotFoundError:
    print(f"ERROR: Fine-tuned model not found at {fine_tuned_model_path}. Please check the path.")
    exit()

required_objects_lower = [obj.lower() for obj in required_objects]

# --- HELPER FUNCTION FROM FIRST SCRIPT ---
def getColours(cls_num):
    """Generates a pseudo-random color based on the class number (used for consistent box/text coloring)."""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

## Main function
# 1. Iterate over video directories (ex. 30m, 100m, etc)
for dir in video_directories:
    for filename in os.listdir(dir):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
              continue

        video_path = os.path.join(dir, filename)
        print(f'Processing: {video_path}')

        # Initialize tracking history dictionary
        track_history = {}
        motion_estimator = MotionEstimator()
        total_cam_dx = 0
        total_cam_dy = 0

        videoCap = cv2.VideoCapture(video_path)
        if not videoCap.isOpened():
            print(f'Invalid file or cannot open video: {video_path}')
            continue

        # Get video properties for the output file
        fps = int(videoCap.get(cv2.CAP_PROP_FPS))
        width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # # Output VideoWriter setup
        # os.makedirs(output_directory, exist_ok=True) 
        
        # # Updated output filename prefix
        # base_output_filename = f"bboxes_{dir}_{filename}"
        # output_filename = os.path.join(output_directory, base_output_filename)
        
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        # out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # print(f"-> Saving output to: {output_filename}")

        frame_count = 0
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break # End of video stream

            # Estimate camera motion
            cam_dx, cam_dy = motion_estimator.process_frame(frame)
            total_cam_dx += cam_dx
            total_cam_dy += cam_dy

            # VISUAL DEBUG: Display the accumulated camera shift
            cv2.putText(frame, f"Cam Shift: {int(total_cam_dx)}, {int(total_cam_dy)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_count += 1
            # Run YOLO tracking. persist=True maintains tracking IDs between frames.
            # NOTE: We use the 'model' which is the fine-tuned model here.
            results = model.track(frame, persist=True, verbose=False)

            # Process detection results
            if results and results[0].boxes:
                # Iterate through all detected bounding boxes in the current frame
                for box in results[0].boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    # 2. Use the CLASS_NAMES from the fine-tuned model
                    class_name_raw = CLASS_NAMES[cls]
                    class_name_lower = class_name_raw.lower() 

                    # 1. FILTERING: Check if the detected object is in the required list
                    if required_objects and class_name_lower not in required_objects_lower:
                        continue

                    # 2. FILTERING: Check if the confidence meets the minimum threshold
                    if conf >= min_conf:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate Center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        current_center = (center_x, center_y)

                        # Check for tracking ID (if available, used for consistent tracking visualization)
                        track_id = int(box.id.item()) if box.id is not None else -1

                        # Get the class-specific color
                        colour = getColours(cls)

                        # Arrow Drawing Logic
                        if track_id!= -1:
                            if track_id not in track_history:
                                track_history[track_id] = []
                            # Subtract camera motion
                            real_x = current_center[0] - total_cam_dx
                            real_y = current_center[1] - total_cam_dy
                            # Add current center to history
                            track_history[track_id].append((real_x, real_y))

                            # Keep only last 40 frames of history
                            if len(track_history[track_id]) > 15:
                                track_history[track_id].pop(0)
                            
                            # Compare current position with oldest position in history
                            if len(track_history[track_id]) > 10:
                                prev_real_x, prev_real_y = track_history[track_id][0]

                                dx = real_x - prev_real_x
                                dy = real_y - prev_real_y
                                magnitude = math.sqrt(dx**2 + dy**2)

                                box_size = max(x2 - x1, y2 - y1)

                                dynamic_threshold = box_size * 0.2

                                # Only draw arrow if movement is significant
                                if magnitude > dynamic_threshold:
                                    end_x = int(current_center[0] + dx)
                                    end_y = int(current_center[1] + dy)
                                    cv2.arrowedLine(frame, current_center, (end_x, end_y), (0, 0, 255), 4, tipLength=0.4)                            

                        # --- DRAWING ---
                        # Use the raw class name for display (e.g., 'Vehicle' instead of 'vehicle')
                        label = f"{class_name_raw} {conf:.2f}"
                        if track_id != -1:
                            label += f" | ID {track_id}"

                        # Draw the bounding box (using the class-specific color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        # Draw the label text (using the class-specific color)
                        cv2.putText(frame, label,
                                    (x1, max(y1 - 10, 20)), # Position the text slightly above the box
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, colour, 2)

            # # Write annotated frame to output video
            # out.write(frame)

            # Optional: Live preview 
            cv2.imshow(f'Live Preview: {filename}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                videoCap.release()
                cv2.destroyAllWindows()
                print("Exiting loop via 'q' key press.")
                exit()

        # Cleanup for the current video
        videoCap.release()
        # out.release()
        cv2.destroyAllWindows() 
        # print(f'Finished processing {filename}. Output saved as: {output_filename}\n')

print("All videos processed.")