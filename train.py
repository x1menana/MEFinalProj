import cv2
import os
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
import json
# print(cv2.__version__)

# declare necessary dictionaries
training_img_dir = 'data/train/images'
training_labels = 'data/train/labels'
yolo_model_path = 'yolo_models/yolov8n.pt'

# check dataset
info = check_det_dataset('data.yaml')
print(info)

def main():
    # obtain dictionary for classes

    # declare yolo model to finetune
    model = YOLO('yolov8n.pt')

    # fine tune TODO: increase epochs
    model.train(data='data.yaml', epochs=50) # enable gpu
    pass

if __name__ == "__main__":
    main()