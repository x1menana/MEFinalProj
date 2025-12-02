import cv2
import os
import csv
from ultralytics import YOLO
import random
import math
from motion_estimator import MotionEstimator
from collections import defaultdict

# Select CV model
fine_tuned_model_path = 'runs/detect/train9/weights/best.pt'
video_directories = ['30m_elevation_study_dkr', '100m_elevation_study_waterloo']
# video_directories = ['100m_elevation_study_waterloo'] # Test run
output_directory = 'test_output'

required_objects = ['Vehicle'] 
min_conf = 0.7 # Default minimum confidence

# Initiate model
try:
    model = YOLO(fine_tuned_model_path)
    model.verbose = False
    CLASS_NAMES = model.names
    print(f'Fine-tuned model loaded with names: {CLASS_NAMES}')
except FileNotFoundError:
    print(f"ERROR: Fine-tuned model not found at {fine_tuned_model_path}. Please check the path.")
    exit()

required_objects_lower = [obj.lower() for obj in required_objects] # ex. to read both ".mp4" and ".MP4" files

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# Retrieve timestamps for .csv output
def get_time_ranges(frames, fps):
    if not frames:
        return "N/A"
    
    frames.sort()
    
    ranges = []
    current_start = frames[0]
    current_end = frames[0]
    
    gap_threshold = fps + 1 
    
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] > gap_threshold:
            ranges.append((current_start, current_end))
            current_start = frames[i]
        current_end = frames[i]
        
    ranges.append((current_start, current_end)) # Add the last range

    # Format ranges into string
    time_ranges_str = []
    for start_frame, end_frame in ranges:
        start_time = start_frame / fps
        end_time = end_frame / fps
        time_ranges_str.append(f"{start_time:.2f}s - {end_time:.2f}s")
        
    return " | ".join(time_ranges_str)

# Write .csv results file
def write_csv_report(id_data, fps, output_filepath):
    report_data = []
    
    for track_id, data in id_data.items():
        if not data['confidences'] or not data['frames']:
            continue
            
        # Calculates average confidence
        avg_conf = sum(data['confidences']) / len(data['confidences'])
        
        # Collects timestamps and timestamp ranges
        frame_ranges_str = get_time_ranges(data['frames'], fps)
        
        report_data.append({
            'ID': track_id,
            'Class': data['class_name'],
            'Average_Confidence': f"{avg_conf:.4f}",
            'Timestamp_Ranges': frame_ranges_str,
            'Total_Frames_Tracked': len(data['frames'])
        })

    # Sort by detection ID
    report_data.sort(key=lambda x: x['ID'])
    
    csv_fields = ['ID', 'Class', 'Average_Confidence', 'Total_Frames_Tracked', 'Timestamp_Ranges']
    
    try:
        with open(output_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(report_data)
        print(f".csv report saved to: {output_filepath}")
    except Exception as e:
        print(f"ERROR writing .csv file: {e}")

# Main annotation function
for dir in video_directories:
    min_conf = 0.4 if dir == '100m_elevation_study_waterloo' else 0.7
    
    for filename in os.listdir(dir):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(dir, filename)
        print(f'Processing: {video_path}')

        # Data entries for .csv file
        id_data = defaultdict(lambda: {'class_name': '', 'confidences': [], 'frames': []})

        # Initialize motion estimator
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
        
        os.makedirs(output_directory, exist_ok=True) 
        
        # Export paths
        base_name_no_ext = os.path.splitext(filename)[0]
        output_video_filename = os.path.join(output_directory, f"bboxes_{dir}_{base_name_no_ext}.mp4")
        output_csv_filename = os.path.join(output_directory, f"summary_{dir}_{base_name_no_ext}.csv")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
        out = cv2.VideoWriter(output_video_filename, fourcc, fps, (width, height))

        print(f"Saving video output to: {output_video_filename}")

        frame_count = 0
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break

            frame_count += 1
            
            cam_dx, cam_dy = motion_estimator.process_frame(frame)
            total_cam_dx += cam_dx
            total_cam_dy += cam_dy

            results = model.track(frame, persist=True, verbose=False)

            # Process detection results
            if results and results[0].boxes:
                for box in results[0].boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    track_id = int(box.id.item()) if box.id is not None else -1
                    
                    class_name_raw = CLASS_NAMES[cls]
                    class_name_lower = class_name_raw.lower() 

                    if required_objects and class_name_lower not in required_objects_lower:
                        continue

                    if conf >= min_conf:
                        # Initialize annotation
                        if track_id != -1:
                            data = id_data[track_id]
                            data['class_name'] = class_name_raw
                            data['confidences'].append(conf)
                            data['frames'].append(frame_count)
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        current_center = (center_x, center_y)

                        colour = getColours(cls)

                        if track_id != -1:
                            if track_id not in track_history:
                                track_history[track_id] = []
                                
                            # Handles camera motion
                            real_x = current_center[0] - total_cam_dx
                            real_y = current_center[1] - total_cam_dy
                            track_history[track_id].append((real_x, real_y))

                            if len(track_history[track_id]) > 15:
                                track_history[track_id].pop(0)
                            
                            if len(track_history[track_id]) > 10:
                                prev_real_x, prev_real_y = track_history[track_id][0]

                                dx = real_x - prev_real_x
                                dy = real_y - prev_real_y
                                magnitude = math.sqrt(dx**2 + dy**2)

                                box_size = max(x2 - x1, y2 - y1)
                                dynamic_threshold = box_size * 0.2

                                # Arrow is drawn only for significant movement
                                if magnitude > dynamic_threshold:
                                    end_x = int(current_center[0] + dx)
                                    end_y = int(current_center[1] + dy)
                                    cv2.arrowedLine(frame, current_center, (end_x, end_y), (0, 0, 255), 4, tipLength=0.4)

                        # Drawing
                        label = f"{class_name_raw} {conf:.2f}"
                        if track_id != -1:
                            label += f" | ID {track_id}"

                        # Annotate bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        cv2.putText(frame, label,
                                    (x1, max(y1 - 10, 20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, colour, 2)

            out.write(frame)

            # OPTIONAL: Live preview
            cv2.imshow(f'Live Preview: {filename}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        videoCap.release()
        out.release()
        cv2.destroyAllWindows() 
        print(f'Finished processing {filename}')

        write_csv_report(id_data, fps, output_csv_filename)

print("All videos processed.")