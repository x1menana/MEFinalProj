import cv2
import os
from ultralytics import YOLO
import random
print(cv2.__version__)

yolo_model_path = 'yolo_models/yolov8n.pt'
video_directories = ['30m_elevation_study_dkr', '100m_elevation_study_waterloo', 'drone_gimbal_gammarc', 'varying_elevation_study_dkr']
### Declare models here
fine_tuned_model = YOLO('runs/detect/train3/weights/best.pt')
print(f'fine tuned model names: {fine_tuned_model.names}')
model = YOLO(yolo_model_path)

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

frame_count = 0
min_conf = 0.4
required_objects = ['Car']
for dir in video_directories:
    for filename in os.listdir(dir):
        video_path = os.path.join(dir, filename)
        print(f'extracting frames from current video: {video_path}')

        # get vid capture
        videoCap = cv2.VideoCapture(video_path)
        if not videoCap.isOpened():
            print(f'file_path not valid: {video_path}')
            break
        # get frame:
        while True:
            ret, frame = videoCap.read()
            if not ret:
                break
            ### TODO HERE: feed in frame to model ###
            # results = model.predict(source=frame, conf=0.1, verbose=True)  # lower conf for debugging
            # res = results[0]
            # print('num detections:', len(res.boxes))
            results = model.track(frame, stream=True)
            for result in results:
                # print(f'result: {result}')
                class_names = result.names
                # print(f'class_names: {class_names}')
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    # print(f' curr class_name: {class_name}')
                    if class_name not in required_objects: # if we dont detect a car or other req objects
                        continue
                    if box.conf[0] > min_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        colour = getColours(cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}",
                                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, colour, 2)
                        
            cv2.imshow(f'video: {filename}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

videoCap.release()
cv2.destroyAllWindows()