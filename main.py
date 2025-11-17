import cv2
import os
print(cv2.__version__)

video_directories = ['30m_elevation_study_dkr', '100m_elevation_study_waterloo', 'drone_gimbal_gammarc', 'varying_elevation_study_dkr']
# video_directories = ['30m_elevation_study_dkr']

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
            cv2.imshow(f'video: {filename}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ### TODO HERE: feed in frame to model ###

videoCap.release()
cv2.destroyAllWindows()